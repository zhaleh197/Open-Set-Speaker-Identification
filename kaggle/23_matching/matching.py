"""
One-to-one matching between test groups and known speakers.

The reframing. This is not open-set recognition. There are exactly 999 people;
446 carry identity labels and 553 are pooled under one label. No genuinely unseen
speaker exists. The eval set is ~3,604 files over those same 999 people, about
3.6 files each -- so it is ~999 groups, of which exactly 446 are known speakers
and the rest are not.

That makes a hard constraint available that independent argmax throws away:

    each known speaker is one person, so it owns AT MOST ONE group

Right now two different groups can both claim speaker X (one of them is
necessarily wrong) while speaker Y receives nothing at all and scores F1 = 0.
A one-to-one assignment repairs both failures in a single step.

Why this is not the Sinkhorn that failed. Sinkhorn imposed a balance on FILES --
that every known speaker receives an equal share of the test set. That was false,
speakers contribute different numbers of files, and it cost 0.023. This imposes
one-to-one matching on GROUPS, which follows from each speaker being one person.
The earlier constraint was invented; this one is a fact about the data.

Grouping was already measured at 98% purity covering 86% of files, in the
notebook where averaging those groups turned out to be useless. The grouping was
sound; only the use of it was wrong.
"""
import json, collections, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score

O2, O5 = (sys.argv[1:3] + ["_out", "_out5"])[:2]
COHORT, NFOLDS, TAU = 300, 12, 0.25
GROUP_THR = 0.30

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]
KU = len(classes) - 1

l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
E = {"ecapa": l2(np.load(O2 + "/emb_file.npy").astype("float64")),
     "resnet": l2(np.load(O5 + "/emb_resnet.npy").astype("float64"))}

by = collections.defaultdict(list)
for i, l in enumerate(labels):
    if valid[i]:
        by[l].append(i)
unk = np.array(by["unknown"])


def asnorm(S, Et, Ev, n=COHORT):
    Cv, Ct = Ev @ Et.T, Et @ Et.T
    np.fill_diagonal(Ct, -np.inf)
    g = lambda M: (np.sort(M, 1)[:, -n:].mean(1, keepdims=True),
                   np.sort(M, 1)[:, -n:].std(1, keepdims=True) + 1e-9)
    mv, sv = g(Cv); mt, st = g(Ct)
    return 0.5 * ((S - mv) / sv + (S - mt.T) / st.T)


def make_fold(f, nh=3, share=0.50):
    """3 held-out files per speaker: the only protocol that reproduces the eval's
    several-files-per-person structure, which this method depends on."""
    rng = np.random.RandomState(8000 + f)
    val, trn = [], []
    for s in known:
        idx = list(by[s]); rng.shuffle(idx)
        k = min(nh, max(len(idx) - 2, 0))
        if k <= 0:
            trn += idx; continue
        val += [(i, s) for i in idx[:k]]; trn += idx[k:]
    n_unk = min(int(round(len(val) * share / (1 - share))), len(unk) - 400)
    p = rng.permutation(len(unk))
    val += [(unk[j], "unknown") for j in p[:n_unk]]
    trn += list(unk[p[n_unk:]])
    trn = np.array(trn)
    vi = np.array([i for i, _ in val])
    S = sum(0.5 * asnorm(E[m][vi] @ E[m][trn].T, E[m][trn], E[m][vi])
            for m in ["ecapa", "resnet"])
    tl = np.array([labels[j] for j in trn])
    C = np.full((len(vi), len(classes)), -9e9)
    for ci, c in enumerate(classes):
        cols = np.flatnonzero(tl == c)
        if len(cols):
            C[:, ci] = S[:, cols].max(1)
    A = l2(np.concatenate([E[m][vi] for m in ["ecapa", "resnet"]], axis=1))
    grp = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="complete",
                                  distance_threshold=GROUP_THR).fit_predict(A)
    return val, vi, C, grp


macro = lambda val, yp: f1_score([s for _, s in val], yp, average="macro",
                                 labels=classes, zero_division=0)


def baseline(vi, C):
    sk = C[:, :KU].max(1); su = C[:, KU]; b = C[:, :KU].argmax(1)
    return [classes[b[r]] if sk[r] - su[r] > TAU else "unknown" for r in range(len(vi))]


def matched(vi, C, grp, margin):
    """Aggregate to groups, then solve the assignment once for the whole set."""
    gids = np.unique(grp)
    G = np.zeros((len(gids), len(classes)))
    for gi, g in enumerate(gids):
        m = np.flatnonzero(grp == g)
        G[gi] = C[m].mean(0)                      # a group's evidence for each class

    # profit of giving group gi to known speaker ci, relative to calling it unknown
    profit = G[:, :KU] - (G[:, KU] + margin)[:, None]
    cost = -np.where(profit > 0, profit, 0.0)     # never force a negative match
    rows, cols = linear_sum_assignment(cost)

    yp = ["unknown"] * len(vi)
    for gi, ci in zip(rows, cols):
        if profit[gi, ci] <= 0:
            continue
        for r in np.flatnonzero(grp == gids[gi]):
            yp[r] = classes[ci]
    return yp


print("building %d folds ..." % NFOLDS, flush=True)
FOLDS = [make_fold(f) for f in range(NFOLDS)]
ng = [len(np.unique(g)) for *_, g in FOLDS]
nv = [len(vi) for _, vi, _, _ in FOLDS]
print("  %d files -> %d groups per fold (%.1f files per group)"
      % (int(np.mean(nv)), int(np.mean(ng)), np.mean(nv) / np.mean(ng)))
print("  known speakers to place: %d\n" % len(known))

base = np.array([macro(v, baseline(i, C)) for v, i, C, _ in FOLDS])
print("=" * 74)
print("baseline (independent argmax per file)   %.4f +/-%.4f"
      % (base.mean(), base.std(ddof=1) / np.sqrt(len(base))))
print("=" * 74)
for margin in [0.0, 0.10, 0.20, 0.25, 0.35, 0.50]:
    v = np.array([macro(vv, matched(ii, C, g, margin)) for vv, ii, C, g in FOLDS])
    d = v - base
    sem = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / sem if sem > 0 else 0
    print("  one-to-one matching, margin %.2f    %.4f   %+.4f (t=%+.1f)%s"
          % (margin, v.mean(), d.mean(), t, "  SIGNIFICANT" if abs(t) > 2.5 else ""))

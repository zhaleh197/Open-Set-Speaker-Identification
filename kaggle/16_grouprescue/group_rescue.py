"""
Group-aware rescue of starved speakers.

Two findings that are weak alone and strong together.

  1. macro-F1 pays absurdly well for rescuing a speaker that currently receives
     zero predictions, and charges almost nothing when the rescue misses.
  2. Test files of one speaker can be grouped at 98% purity, covering 86% of the
     set -- but averaging their embeddings (tested separately) buys nothing.

So use the groups for the rescue instead of for the query. For a speaker with no
predictions, hand it a whole GROUP rather than a single file:

     single file, 1 of ~3.6 true   ->  F1 = 2*1/(1+3.6) = 0.36
     whole group, 3 of ~3.6 true   ->  F1 = 2*3/(3+3.6) = 0.91

Two and a half times the payoff for the same bet, because a pure group is right
or wrong together. Failure still costs almost nothing: the class stays at zero
and 'unknown' loses three hits out of roughly eighteen hundred.

Evaluated on folds holding out 3 files per speaker, the only protocol I have that
reproduces the eval's multiple-files-per-speaker structure.
"""
import json, collections, sys
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score

O2, O5 = (sys.argv[1:3] + ["_out", "_out5"])[:2]
COHORT, NFOLDS = 300, 12
MODELS, WEIGHTS, TAU = ["ecapa", "resnet"], [1, 3], 0.20
GROUP_THR, MAX_GROUP = 0.30, 5

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]
KU = len(classes) - 1

l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
E = {"ecapa": l2(np.load(O2 + "/emb_file.npy").astype("float64")),
     "resnet": l2(np.load(O5 + "/emb_resnet.npy").astype("float64"))}
WN = np.array(WEIGHTS, float); WN /= WN.sum()

by = collections.defaultdict(list)
for i, l in enumerate(labels):
    if valid[i]:
        by[l].append(i)
unk = np.array(by["unknown"])
kp = np.flatnonzero(valid)
G0 = E["ecapa"][kp] @ E["ecapa"][kp].T
np.fill_diagonal(G0, 0)
DEGEN = set(kp[(G0 > 0.9999).sum(1) > 0].tolist())


def asnorm(S, Et, Ev, n=COHORT):
    Cv, Ct = Ev @ Et.T, Et @ Et.T
    np.fill_diagonal(Ct, -np.inf)
    g = lambda M: (np.sort(M, 1)[:, -n:].mean(1, keepdims=True),
                   np.sort(M, 1)[:, -n:].std(1, keepdims=True) + 1e-9)
    mv, sv = g(Cv); mt, st = g(Ct)
    return 0.5 * ((S - mv) / sv + (S - mt.T) / st.T)


def make_fold(f, nh=3, share=0.50):
    rng = np.random.RandomState(9000 + f)
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
    S = sum(a * asnorm(E[m][vi] @ E[m][trn].T, E[m][trn], E[m][vi])
            for a, m in zip(WN, MODELS))
    tl = np.array([labels[j] for j in trn])
    C = np.full((len(vi), len(classes)), -9e9)
    for ci, c in enumerate(classes):
        cols = np.flatnonzero(tl == c)
        if len(cols):
            C[:, ci] = S[:, cols].max(1)
    A = l2(np.concatenate([E[m][vi] for m in MODELS], axis=1))
    grp = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="complete",
                                  distance_threshold=GROUP_THR).fit_predict(A)
    return val, vi, C, grp


def base_labels(vi, C, bias=None):
    A = C if bias is None else C + bias[None, :]
    sk = A[:, :KU].max(1); su = A[:, KU]; b = A[:, :KU].argmax(1)
    return [classes[b[r]] if sk[r] - su[r] > TAU else "unknown" for r in range(len(vi))]


def rescue(yp, C, grp, mode, k=2):
    """mode 'file': give a starved speaker its k best loose files.
       mode 'group': give it the best whole group of currently-unknown files."""
    got = collections.Counter(yp)
    free = {r for r in range(len(yp)) if yp[r] == "unknown"}
    if mode == "group":
        members = collections.defaultdict(list)
        for r in free:
            members[grp[r]].append(r)
        blocks = [v for v in members.values() if 1 <= len(v) <= MAX_GROUP]
    for ci, c in enumerate(known):
        if got[c] > 0 or not free:
            continue
        if mode == "file":
            cand = sorted([r for r in free], key=lambda r: -C[r, ci])[:k]
        else:
            avail = [b for b in blocks if all(r in free for r in b)]
            if not avail:
                continue
            cand = max(avail, key=lambda b: np.mean([C[r, ci] for r in b]))
        for r in cand:
            yp[r] = c
            free.discard(r)
    return yp


def guard(val, yp):
    for j, (i, _) in enumerate(val):
        if i in DEGEN:
            yp[j] = "unknown"
    return yp


macro = lambda val, yp: f1_score([s for _, s in val], yp, average="macro",
                                 labels=classes, zero_division=0)

print("building %d folds (3 held-out files per speaker) ..." % NFOLDS, flush=True)
FOLDS = [make_fold(f) for f in range(NFOLDS)]

sizes = collections.Counter()
for _, _, _, grp in FOLDS:
    sizes.update(collections.Counter(collections.Counter(grp).values()))
print("group sizes across folds:", dict(sorted(sizes.items())[:6]))

base = np.array([macro(v, guard(v, base_labels(i, C))) for v, i, C, _ in FOLDS])
print("\n" + "=" * 76)
print("baseline                                  %.4f +/-%.4f"
      % (base.mean(), base.std(ddof=1) / np.sqrt(len(base))))
print("=" * 76)


def report(name, fn):
    v = np.array([macro(vv, fn(vv, ii, C, g)) for vv, ii, C, g in FOLDS])
    d = v - base
    sem = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / sem if sem > 0 else 0
    print("  %-40s %.4f   %+.4f (t=%+.1f)%s"
          % (name, v.mean(), d.mean(), t, "  SIGNIFICANT" if abs(t) > 2.5 else ""))
    return v.mean()


for k in [1, 2, 3]:
    report("file rescue, k=%d" % k,
           lambda v, i, C, g, k=k: guard(v, rescue(base_labels(i, C), C, g, "file", k)))

report("GROUP rescue (whole group at once)",
       lambda v, i, C, g: guard(v, rescue(base_labels(i, C), C, g, "group")))

# stack the per-speaker bias on top
acc = np.zeros(len(classes))
for val, vi, C, g in FOLDS:
    acc += f1_score([s for _, s in val], guard(val, base_labels(vi, C)),
                    average=None, labels=classes, zero_division=0)
F = acc / len(FOLDS)
for alpha in [0.2, 0.35]:
    B = np.zeros(len(classes)); B[:KU] = -alpha * np.log(np.clip(F[:KU], 0.05, 1.0))
    report("bias a=%.2f + GROUP rescue" % alpha,
           lambda v, i, C, g, B=B: guard(v, rescue(base_labels(i, C, B), C, g, "group")))

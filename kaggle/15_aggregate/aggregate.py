"""
Test-time query aggregation -- and a protocol that can actually see it.

The real eval hands us ~3.6 recordings of every known speaker. Averaging three
embeddings of the same person cuts the noise by about sqrt(3), which is exactly
the medicine for the short files that carry 46% of our errors. Nothing in the
pipeline currently exploits this: every test file is answered alone.

Why my earlier transductive attempt failed, and why this is not that. That one
clustered test files and copied the most confident LABEL across each cluster;
clusters merged different speakers and the wrong label spread. This never copies
a label. It averages the EMBEDDINGS of files that look like the same recording
session, producing one cleaner query, and then asks the question once. A wrong
grouping degrades a query; it cannot broadcast a wrong answer.

The protocol point that matters. Every fold I have built so far holds out 1-3
files per speaker, so there was barely anything to group and this idea was
invisible -- untestable, not untested. These folds hold out 3 per speaker, which
mirrors the real eval's ~3.6 on the query side. The prototype side is thinner
than reality (2 instead of 5), so the absolute numbers here are pessimistic; the
comparison between with and without aggregation is what counts.
"""
import json, collections, sys
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score

O2, O5 = (sys.argv[1:3] + ["_out", "_out5"])[:2]
COHORT = 300
MODELS, WEIGHTS, TAU = ["ecapa", "resnet"], [1, 3], 0.20
NFOLDS = 12

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
G = E["ecapa"][kp] @ E["ecapa"][kp].T
np.fill_diagonal(G, 0)
DEGEN = set(kp[(G > 0.9999).sum(1) > 0].tolist())


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
    return val, np.array([i for i, _ in val]), np.array(trn)


def class_matrix(Q, trn):
    """Q: query vectors per model. Returns class-collapsed fused scores."""
    S = 0.0
    for a, m in zip(WN, MODELS):
        S = S + a * asnorm(Q[m] @ E[m][trn].T, E[m][trn], Q[m])
    tl = np.array([labels[j] for j in trn])
    C = np.full((S.shape[0], len(classes)), -9e9)
    for ci, c in enumerate(classes):
        cols = np.flatnonzero(tl == c)
        if len(cols):
            C[:, ci] = S[:, cols].max(1)
    return C


def decide(C):
    sk = C[:, :KU].max(1); su = C[:, KU]; b = C[:, :KU].argmax(1)
    return [classes[b[r]] if sk[r] - su[r] > TAU else "unknown" for r in range(len(sk))]


def guard(val, yp):
    for j, (i, _) in enumerate(val):
        if i in DEGEN:
            yp[j] = "unknown"
    return yp


def run(val, vi, trn, thr=None, max_group=4):
    if thr is None:
        Q = {m: E[m][vi] for m in MODELS}
        return guard(val, decide(class_matrix(Q, trn)))

    # group query files that look like the same recording session
    A = l2(np.concatenate([E[m][vi] for m in MODELS], axis=1))
    cl = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="complete",
                                 distance_threshold=thr).fit_predict(A)
    Q = {m: E[m][vi].copy() for m in MODELS}
    sizes = collections.Counter(cl)
    for c, n in sizes.items():
        if n < 2 or n > max_group:
            continue                      # a huge group is a merge error, not a speaker
        idx = np.flatnonzero(cl == c)
        for m in MODELS:
            Q[m][idx] = l2(E[m][vi][idx].mean(0))[None, :]
    return guard(val, decide(class_matrix(Q, trn)))


macro = lambda val, yp: f1_score([s for _, s in val], yp, average="macro",
                                 labels=classes, zero_division=0)

print("folds hold out 3 files per speaker so grouping has something to find\n")
FOLDS = [make_fold(f) for f in range(NFOLDS)]

# how well could grouping possibly do here?
pur, cov = [], []
for val, vi, trn in FOLDS:
    A = l2(np.concatenate([E[m][vi] for m in MODELS], axis=1))
    cl = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="complete",
                                 distance_threshold=0.30).fit_predict(A)
    yt = [s for _, s in val]
    g = collections.defaultdict(list)
    for r, c in enumerate(cl):
        g[c].append(r)
    multi = [v for v in g.values() if len(v) > 1]
    pure = [v for v in multi if len({yt[r] for r in v}) == 1]
    pur.append(len(pure) / max(len(multi), 1))
    cov.append(sum(len(v) for v in multi) / len(val))
print("at threshold 0.30: %.0f%% of multi-file groups are single-speaker, "
      "%.0f%% of files land in a group" % (np.mean(pur) * 100, np.mean(cov) * 100))

base = np.array([macro(v, run(v, i, t)) for v, i, t in FOLDS])
print("\n" + "=" * 74)
print("baseline (each file answered alone)   %.4f +/-%.4f"
      % (base.mean(), base.std(ddof=1) / np.sqrt(len(base))))
print("=" * 74)
for thr in [0.15, 0.20, 0.25, 0.30, 0.35, 0.45]:
    v = np.array([macro(vv, run(vv, ii, tt, thr)) for vv, ii, tt in FOLDS])
    d = v - base
    sem = d.std(ddof=1) / np.sqrt(len(d))
    print("  group at cosine distance < %.2f   %.4f   %+.4f (t=%+.1f)%s"
          % (thr, v.mean(), d.mean(), d.mean() / sem if sem > 0 else 0,
             "  SIGNIFICANT" if sem > 0 and abs(d.mean() / sem) > 2.5 else ""))

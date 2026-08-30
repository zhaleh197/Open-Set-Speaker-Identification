"""
Method lab. Runs on cached embeddings only -- no GPU, no Kaggle, seconds per idea.

Two protocols, because they answer different questions:

  holdout-1  1 val file per known speaker. Matches notebook 02/03 so numbers stay
             comparable with the 0.9600 leaderboard result.
  holdout-2  2 val files per known speaker, prototypes built from the other 3.
             The real eval has ~3.6 files per known speaker, so any method that
             exploits multiple test files of the same speaker (transductive
             clustering) is INVISIBLE under holdout-1 and only shows up here.
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score

OUT = sys.argv[1] if len(sys.argv) > 1 else "_out"
SEED = 0

emb = np.load(OUT + "/emb_file.npy")
meta = json.load(open(OUT + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]

by = collections.defaultdict(list)
for i, l in enumerate(labels):
    if valid[i]:
        by[l].append(i)
unk = np.array(by["unknown"])

E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

# the 34 non-speech files whose embeddings collapse onto one point
keepall = np.flatnonzero(valid)
_P = E[keepall]
_G = _P @ _P.T
np.fill_diagonal(_G, 0)
DEGEN = set(keepall[(_G > 0.9999).sum(1) > 0].tolist())


def folds(n_hold_known, n_folds=5):
    for f in range(n_folds):
        rng = np.random.RandomState(SEED + f)
        val, trn = [], []
        for s in known:
            idx = list(by[s])
            if len(idx) <= n_hold_known:
                trn += idx
                continue
            rng.shuffle(idx)
            val += [(i, s) for i in idx[:n_hold_known]]
            trn += idx[n_hold_known:]
        p = rng.permutation(len(unk))
        nh = min(len(val), len(unk) // 2)
        val += [(unk[j], "unknown") for j in p[:nh]]
        trn += list(unk[p[nh:]])
        yield val, np.array(trn)


def macro(val, yp):
    return f1_score([s for _, s in val], yp, average="macro",
                    labels=classes, zero_division=0)


def asnorm(S, Etrn, Eval, cohort_n=300):
    """Adaptive score normalisation. Standard in speaker verification: rescale
    every trial by the score distribution each side produces against an imposter
    cohort, so one 'magnetic' prototype cannot win every comparison."""
    Cv = Eval @ Etrn.T                       # val vs cohort (= the training set)
    Ct = Etrn @ Etrn.T
    np.fill_diagonal(Ct, -np.inf)

    def stats(M):
        top = np.sort(M, axis=1)[:, -cohort_n:]
        return top.mean(1, keepdims=True), top.std(1, keepdims=True) + 1e-9

    mv, sv = stats(Cv)
    mt, st = stats(Ct)
    return 0.5 * ((S - mv) / sv + (S - mt.T) / st.T)


def predict(val, trn, use_asnorm=False, transductive=0, guard_degen=False):
    vi = np.array([i for i, _ in val])
    Ev, Et = E[vi], E[trn]
    S = Ev @ Et.T
    if use_asnorm:
        S = asnorm(S, Et, Ev)
    nn = S.argmax(1)
    yp = [labels[trn[j]] for j in nn]
    conf = S[np.arange(len(vi)), nn]

    if transductive > 0:
        # Test files of one speaker sit close together. Group them, then let the
        # most confident member label the whole group. Fixes the weak files by
        # borrowing evidence from the strong ones -- only possible because the
        # eval set holds several files per speaker.
        from sklearn.cluster import AgglomerativeClustering
        cl = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                     linkage="average",
                                     distance_threshold=transductive).fit_predict(Ev)
        for c in np.unique(cl):
            m = np.flatnonzero(cl == c)
            if len(m) < 2:
                continue
            best = m[conf[m].argmax()]
            for j in m:
                yp[j] = yp[best]

    if guard_degen:
        for j, (i, _) in enumerate(val):
            if i in DEGEN:
                yp[j] = "unknown"
    return yp


def run(name, n_hold, **kw):
    sc = [macro(v, predict(v, t, **kw)) for v, t in folds(n_hold)]
    print("  %-46s %.4f  (+/- %.4f)" % (name, np.mean(sc), np.std(sc)))
    return float(np.mean(sc))


print("=" * 72)
print("PROTOCOL A: holdout-1  (comparable to the 0.9600 submission)")
print("=" * 72)
base1 = run("1-NN baseline (what scored 0.9600)", 1)
run("+ degenerate guard", 1, guard_degen=True)
run("+ AS-norm", 1, use_asnorm=True)
run("+ AS-norm + degenerate guard", 1, use_asnorm=True, guard_degen=True)

print()
print("=" * 72)
print("PROTOCOL B: holdout-2  (2 val files/speaker -- lets transductive show)")
print("=" * 72)
base2 = run("1-NN baseline", 2)
run("+ AS-norm", 2, use_asnorm=True)
run("+ degenerate guard", 2, guard_degen=True)
for th in [0.15, 0.25, 0.35, 0.45]:
    run("+ transductive (dist<%.2f)" % th, 2, transductive=th)
run("+ AS-norm + transductive 0.25 + guard", 2,
    use_asnorm=True, transductive=0.25, guard_degen=True)

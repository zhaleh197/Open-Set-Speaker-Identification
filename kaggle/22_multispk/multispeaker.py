"""
Is "one file = one speaker" actually true?

Every version of this pipeline averages the embeddings of a file's chunks into
one vector. That is correct only if the file contains one person. If some of
these 60-second recordings hold two speakers, the average lands between them and
resembles neither -- which is exactly the unexplained anomaly the rescue
diagnostic turned up: a starved speaker's own files do not score for that
speaker.

Nobody checks this because the assumption is invisible. It has never been tested
here either, and the data to test it is already on disk: 32,412 chunk-level
embeddings with their file ownership.

Three questions:

  1. Split each file's chunks into two clusters. How far apart are they? A
     single-speaker recording gives two nearly identical halves; a two-speaker
     recording gives a real gap.
  2. Do files with a large internal gap actually get misclassified more often?
     If the split is real but harmless, it is a curiosity, not a cause.
  3. Does replacing the mean with the DOMINANT cluster's mean fix them? That is
     the whole point -- take the majority speaker instead of the blend.
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score

O2, O5 = (sys.argv[1:3] + ["_out", "_out5"])[:2]
COHORT, NFOLDS, TAU = 300, 12, 0.25

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
dur = np.array(meta["durations"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]
KU = len(classes) - 1
N = len(labels)

l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
C = l2(np.load(O2 + "/emb_chunk.npy").astype("float64"))
own = np.load(O2 + "/chunk_owner.npy")
chunks_of = collections.defaultdict(list)
for ci, f in enumerate(own):
    chunks_of[int(f)].append(ci)

# ---------------------------------------------------------------- question 1
def two_way_split(X):
    """Cheapest honest 2-means on the unit sphere: seed with the two most
    dissimilar chunks, then a few assignment/update rounds."""
    if len(X) < 4:
        return None
    G = X @ X.T
    i, j = np.unravel_index(np.argmin(G), G.shape)
    a, b = X[i].copy(), X[j].copy()
    for _ in range(8):
        asg = (X @ a) < (X @ b)
        if asg.all() or (~asg).all():
            return None
        a = l2(X[~asg].mean(0)); b = l2(X[asg].mean(0))
    return float(a @ b), int((~asg).sum()), int(asg.sum()), asg


gap, n_major, coh = np.full(N, np.nan), np.zeros(N, int), np.full(N, np.nan)
DOM = {}
for f, idx in chunks_of.items():
    X = C[idx]
    m = X.mean(0)
    coh[f] = np.linalg.norm(m)
    r = two_way_split(X)
    if r is None:
        continue
    sim, na, nb, asg = r
    gap[f] = sim
    big = ~asg if na >= nb else asg
    n_major[f] = int(big.sum())
    DOM[f] = l2(X[big].mean(0))

ok = ~np.isnan(gap)
print("files with enough chunks to split: %d" % ok.sum())
print("\ncosine between a file's two internal clusters:")
for q in [1, 5, 10, 25, 50]:
    print("   p%-3d %.4f" % (q, np.percentile(gap[ok], q)))
print("   a single-speaker recording should sit near 1.0;")
print("   a genuinely two-speaker file would sit far below it")

for thr in [0.5, 0.6, 0.7, 0.8]:
    n = int((gap[ok] < thr).sum())
    print("   files with internal gap below %.1f : %4d  (%.1f%%)"
          % (thr, n, n / ok.sum() * 100))

# ---------------------------------------------------------------- question 2
print("\n" + "=" * 74)
print("do split files actually fail more often?")
print("=" * 74)
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


def make_fold(f, share=0.50):
    rng = np.random.RandomState(6000 + f)
    val, trn = [], []
    for s in known:
        idx = list(by[s]); rng.shuffle(idx)
        nh = min(rng.randint(1, 4), max(len(idx) - 2, 0))
        if nh <= 0:
            trn += idx; continue
        val += [(i, s) for i in idx[:nh]]; trn += idx[nh:]
    n_unk = min(int(round(len(val) * share / (1 - share))), len(unk) - 400)
    p = rng.permutation(len(unk))
    val += [(unk[j], "unknown") for j in p[:n_unk]]
    trn += list(unk[p[n_unk:]])
    return val, np.array([i for i, _ in val]), np.array(trn)


FOLDS = [make_fold(f) for f in range(NFOLDS)]


def run(vec_ecapa):
    """vec_ecapa: the per-file ECAPA vector to use (mean, or dominant cluster)."""
    B = {"ecapa": vec_ecapa, "resnet": E["resnet"]}
    per_file_err = collections.Counter()
    per_file_seen = collections.Counter()
    scores = []
    for val, vi, trn in FOLDS:
        S = sum(0.5 * asnorm(B[m][vi] @ B[m][trn].T, B[m][trn], B[m][vi])
                for m in ["ecapa", "resnet"])
        tl = np.array([labels[j] for j in trn])
        Cm = np.full((len(vi), len(classes)), -9e9)
        for ci, c in enumerate(classes):
            cols = np.flatnonzero(tl == c)
            if len(cols):
                Cm[:, ci] = S[:, cols].max(1)
        sk = Cm[:, :KU].max(1); su = Cm[:, KU]; b = Cm[:, :KU].argmax(1)
        yt = [s for _, s in val]
        yp = [classes[b[r]] if sk[r] - su[r] > TAU else "unknown" for r in range(len(vi))]
        for r, i in enumerate(vi):
            per_file_seen[i] += 1
            if yp[r] != yt[r]:
                per_file_err[i] += 1
        scores.append(f1_score(yt, yp, average="macro", labels=classes, zero_division=0))
    return np.array(scores), per_file_err, per_file_seen


base, err, seen = run(E["ecapa"])
rate = {i: err[i] / seen[i] for i in seen if seen[i] >= 3}
split = [i for i in rate if ok[i] and gap[i] < 0.7]
whole = [i for i in rate if ok[i] and gap[i] >= 0.9]
print("  error rate, files with an internal gap < 0.7 : %.1f%%  (n=%d)"
      % (100 * np.mean([rate[i] for i in split]), len(split)))
print("  error rate, files with an internal gap >= 0.9: %.1f%%  (n=%d)"
      % (100 * np.mean([rate[i] for i in whole]), len(whole)))
print("  overall                                      : %.1f%%"
      % (100 * np.mean(list(rate.values()))))

# ---------------------------------------------------------------- question 3
print("\n" + "=" * 74)
print("replace the mean with the dominant cluster (only where the gap is real)")
print("=" * 74)
print("  baseline (mean over all chunks)      %.4f +/-%.4f"
      % (base.mean(), base.std(ddof=1) / np.sqrt(len(base))))
for thr in [0.5, 0.6, 0.7, 0.8]:
    V = E["ecapa"].copy()
    n = 0
    for f, d in DOM.items():
        if gap[f] < thr:
            V[f] = d; n += 1
    v, _, _ = run(V)
    d = v - base
    sem = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / sem if sem > 0 else 0
    print("  gap < %.1f  (%4d files switched)      %.4f   %+.4f (t=%+.1f)%s"
          % (thr, n, v.mean(), d.mean(), t, "  SIGNIFICANT" if abs(t) > 2.5 else ""))

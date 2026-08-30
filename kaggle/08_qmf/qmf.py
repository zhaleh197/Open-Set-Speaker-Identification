"""
Two literature-grounded upgrades over the v2 backend, both measurable offline.

1. WCCN backend (Within-Class Covariance Normalisation)
   Raw cosine treats every embedding direction as equally informative. WCCN
   whitens out the directions along which the SAME speaker varies (channel,
   room, phonetic content), so the remaining geometry is mostly identity.
   Standard speaker-recognition backend; we can estimate it because we have 446
   labelled speakers with several recordings each.

2. QMF calibration (Quality Measure Functions)
   v2 uses one global margin threshold tau. But the reliability of a score
   depends on the recording: 46% of our errors are on files under 10 s. QMF
   replaces the fixed tau with a logistic regression over the score PLUS
   quality features, so a 60 s file and a 4 s file get different effective
   thresholds. Used by VoxSRC-winning systems.

   Quality features available to us:
     log duration of the test file and of the matched prototype
     chunk coherence = || mean of unit chunk embeddings ||
        near 1.0 when a file's chunks agree with each other, small when the
        recording is inconsistent or mostly non-speech. A free per-file
        confidence signal that falls straight out of the extraction we already
        ran, and the closest analogue to the embedding-magnitude quality term
        used in the literature.
     number of chunks

   The calibrator is fit ONLY on leave-one-out trials among the fold's training
   prototypes, then applied to the held-out files, so no validation data leaks
   into it.
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

O2 = sys.argv[1] if len(sys.argv) > 1 else "_out"
O5 = sys.argv[2] if len(sys.argv) > 2 else "_out5"
TAU_V2, COHORT = 0.20, 300
W = {"ecapa": 1.0, "resnet": 3.0}

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
dur = np.array(meta["durations"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]
N = len(labels)

l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
RAW = {"ecapa": np.load(O2 + "/emb_file.npy").astype("float64"),
       "resnet": np.load(O5 + "/emb_resnet.npy").astype("float64")}

# ---- quality features from the cached chunk embeddings ---------------------
C = l2(np.load(O2 + "/emb_chunk.npy").astype("float64"))
own = np.load(O2 + "/chunk_owner.npy")
coh = np.zeros(N); nch = np.zeros(N)
bucket = collections.defaultdict(list)
for ci, f in enumerate(own):
    bucket[int(f)].append(ci)
for f, idx in bucket.items():
    m = C[idx].mean(0)
    coh[f] = np.linalg.norm(m)          # 1.0 = chunks perfectly agree
    nch[f] = len(idx)
print("chunk coherence: med %.4f  p10 %.4f  (files with >=1 chunk: %d)"
      % (np.median(coh[coh > 0]), np.percentile(coh[coh > 0], 10), (coh > 0).sum()))

by = collections.defaultdict(list)
for i, l in enumerate(labels):
    if valid[i]:
        by[l].append(i)
unk = np.array(by["unknown"])

kp = np.flatnonzero(valid)
_E = l2(RAW["ecapa"])
G = _E[kp] @ _E[kp].T
np.fill_diagonal(G, 0)
DEGEN = set(kp[(G > 0.9999).sum(1) > 0].tolist())


def folds(nh, n=5):
    for f in range(n):
        rng = np.random.RandomState(f)
        val, trn = [], []
        for s in known:
            idx = list(by[s]); rng.shuffle(idx)
            if len(idx) <= nh:
                trn += idx; continue
            val += [(i, s) for i in idx[:nh]]; trn += idx[nh:]
        p = rng.permutation(len(unk)); m = min(len(val), len(unk) // 2)
        val += [(unk[j], "unknown") for j in p[:m]]; trn += list(unk[p[m:]])
        yield val, np.array(trn)


def wccn(E, idx, alpha=0.15):
    """Whitening matrix from within-speaker scatter, with shrinkage: 446 classes
    of ~4 files each is thin for a full covariance, so regularise toward I."""
    d = E.shape[1]
    grp = collections.defaultdict(list)
    for i in idx:
        if labels[i] != "unknown":
            grp[labels[i]].append(i)
    Wm = np.zeros((d, d)); n = 0
    for s, ii in grp.items():
        if len(ii) < 2:
            continue
        X = E[ii] - E[ii].mean(0)
        Wm += X.T @ X; n += len(ii)
    if n == 0:
        return np.eye(d)
    Wm /= n
    Wm = (1 - alpha) * Wm + alpha * np.trace(Wm) / d * np.eye(d)
    ev, V = np.linalg.eigh(Wm)
    return V @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-8))) @ V.T


def asnorm(S, Et, Ev, n=COHORT):
    Cv, Ct = Ev @ Et.T, Et @ Et.T
    np.fill_diagonal(Ct, -np.inf)
    g = lambda M: (np.sort(M, 1)[:, -n:].mean(1, keepdims=True),
                   np.sort(M, 1)[:, -n:].std(1, keepdims=True) + 1e-9)
    mv, sv = g(Cv); mt, st = g(Ct)
    return 0.5 * ((S - mv) / sv + (S - mt.T) / st.T)


def fused(vi, trn, use_wccn):
    S = 0.0
    tot = sum(W.values())
    for name, w in W.items():
        E = RAW[name]
        if use_wccn:
            E = E @ wccn(l2(E), trn)
        E = l2(E)
        S = S + (w / tot) * asnorm(E[vi] @ E[trn].T, E[trn], E[vi])
    return S


def split_scores(S, trn):
    tl = np.array([labels[j] for j in trn])
    isu = tl == "unknown"
    Sk = np.where(isu[None, :], -9e9, S)
    Su = np.where(isu[None, :], S, -9e9)
    return tl, Sk.max(1), Su.max(1), Sk.argmax(1)


def feats(vi, trn, sk, su, best):
    q = lambda i: [np.log1p(dur[i]), coh[i], np.log1p(nch[i])]
    return np.array([[sk[r], sk[r] - su[r], su[r]] + q(vi[r]) + q(trn[best[r]])
                     for r in range(len(vi))])


def fit_qmf(trn, use_wccn):
    """Leave-one-out trials among the training prototypes only."""
    S = fused(trn, trn, use_wccn)
    np.fill_diagonal(S, -9e9)                       # never match a file to itself
    tl, sk, su, best = split_scores(S, trn)
    X = feats(trn, trn, sk, su, best)
    y = np.array([1 if (labels[trn[r]] != "unknown" and tl[best[r]] == labels[trn[r]])
                  else 0 for r in range(len(trn))])
    if y.min() == y.max():
        return None
    return LogisticRegression(max_iter=2000, C=1.0).fit(X, y)


def evaluate(nh, mode, tau=TAU_V2, thr=0.5):
    use_wccn = "wccn" in mode
    use_qmf = "qmf" in mode
    sc = []
    for val, trn in folds(nh):
        vi = np.array([i for i, _ in val])
        cal = fit_qmf(trn, use_wccn) if use_qmf else None
        S = fused(vi, trn, use_wccn)
        tl, sk, su, best = split_scores(S, trn)
        if cal is not None:
            p = cal.predict_proba(feats(vi, trn, sk, su, best))[:, 1]
            yp = [tl[best[r]] if p[r] > thr else "unknown" for r in range(len(vi))]
        else:
            yp = [tl[best[r]] if sk[r] - su[r] > tau else "unknown" for r in range(len(vi))]
        for j, (i, _) in enumerate(val):
            if i in DEGEN:
                yp[j] = "unknown"
        sc.append(f1_score([s for _, s in val], yp, average="macro",
                           labels=classes, zero_division=0))
    return float(np.mean(sc)), float(np.std(sc))


print()
print("=" * 74)
print("%-26s %-22s %s" % ("config", "holdout-1", "holdout-2"))
print("=" * 74)
for mode in ["base", "wccn", "qmf", "wccn+qmf"]:
    if "qmf" in mode:
        # the operating point of a calibrated probability still needs choosing;
        # sweep it the same way tau was swept, and report the robust choice
        rows = [(t, evaluate(1, mode, thr=t), evaluate(2, mode, thr=t))
                for t in [0.30, 0.40, 0.50, 0.60, 0.70]]
        t, a, b = max(rows, key=lambda r: min(r[1][0], r[2][0]))
        print("  %-24s %.4f +/-%.4f   %.4f +/-%.4f   (p>%.2f)"
              % (mode, a[0], a[1], b[0], b[1], t))
    else:
        a, b = evaluate(1, mode), evaluate(2, mode)
        print("  %-24s %.4f +/-%.4f   %.4f +/-%.4f" % (mode, a[0], a[1], b[0], b[1]))
print()
print("reference: v2 shipped = base, and scored 0.9660 on the leaderboard")

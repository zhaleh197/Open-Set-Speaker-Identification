"""
Does the Sinkhorn gain survive a realistic test set, or is it an artifact?

The +0.0094 was measured on folds I built to be exactly 50/50 known/unknown with
exactly 2 held-out files per known speaker. Sinkhorn's assumption was therefore
perfectly true by construction, which is precisely the condition that flatters
it. The real eval is only approximately balanced.

This rebuilds the folds to be deliberately UNfaithful to the assumption:

  * each known speaker contributes a random 1-3 held-out files, not a fixed 2,
    so the known marginal is genuinely uneven
  * the true unknown share is swept over 42-58%, while Sinkhorn is always told
    50% and always assumes the known speakers are uniform

If the gain holds up under that mismatch it is real. If it collapses, the method
was reading my fold construction rather than the data, and does not ship.
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score

O2, O5, O9 = (sys.argv[1:4] + ["_out", "_out5", "_out9"])[:3]
COHORT, NFOLDS = 300, 15
MODELS, WEIGHTS, TAU = ["ecapa", "resnet", "eres2netv2"], [1, 2, 1], 0.25
ASSUMED_UNK = 0.50                      # what the shipped code would believe

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]
KU = len(classes) - 1

l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
E = {"ecapa": l2(np.load(O2 + "/emb_file.npy").astype("float64")),
     "resnet": l2(np.load(O5 + "/emb_resnet.npy").astype("float64")),
     "eres2netv2": l2(np.load(O9 + "/emb_eres2netv2.npy").astype("float64"))}

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


def make_fold(f, true_unk_share):
    """Uneven known counts, and an unknown block sized to hit the target share."""
    rng = np.random.RandomState(5000 + f)
    val, trn = [], []
    for s in known:
        idx = list(by[s]); rng.shuffle(idx)
        nh = min(rng.randint(1, 4), max(len(idx) - 2, 0))     # 1-3, keep >=2 prototypes
        if nh <= 0:
            trn += idx; continue
        val += [(i, s) for i in idx[:nh]]; trn += idx[nh:]
    n_known = len(val)
    n_unk = int(round(n_known * true_unk_share / (1 - true_unk_share)))
    p = rng.permutation(len(unk))
    n_unk = min(n_unk, len(unk) - 400)                        # leave prototypes behind
    val += [(unk[j], "unknown") for j in p[:n_unk]]
    trn += list(unk[p[n_unk:]])
    trn = np.array(trn)
    vi = np.array([i for i, _ in val])
    w = np.array(WEIGHTS, float); w /= w.sum()
    S = sum(a * asnorm(E[m][vi] @ E[m][trn].T, E[m][trn], E[m][vi])
            for a, m in zip(w, MODELS))
    tl = np.array([labels[j] for j in trn])
    C = np.full((len(vi), len(classes)), -9e9)
    for ci, c in enumerate(classes):
        cols = np.flatnonzero(tl == c)
        if len(cols):
            C[:, ci] = S[:, cols].max(1)
    actual = n_unk / len(val)
    return val, vi, C, actual


def baseline(val, vi, C):
    sk = C[:, :KU].max(1); su = C[:, KU]; b = C[:, :KU].argmax(1)
    yp = [classes[b[r]] if sk[r] - su[r] > TAU else "unknown" for r in range(len(vi))]
    for j, (i, _) in enumerate(val):
        if i in DEGEN:
            yp[j] = "unknown"
    return yp


def sinkhorn(val, vi, C, eps, iters, unk_share=ASSUMED_UNK):
    L = C.copy() / eps
    L[:, KU] += TAU / eps
    n, k = L.shape
    target = np.full(k, (1.0 - unk_share) / (k - 1))
    target[KU] = unk_share
    logc = np.log(target * n)
    for _ in range(iters):
        L -= L.max(1, keepdims=True)
        L -= np.log(np.exp(L).sum(1, keepdims=True))
        L += logc[None, :] - np.log(np.exp(L).sum(0, keepdims=True) + 1e-30)
    yp = [classes[j] for j in L.argmax(1)]
    for j, (i, _) in enumerate(val):
        if i in DEGEN:
            yp[j] = "unknown"
    return yp


macro = lambda val, yp: f1_score([s for _, s in val], yp, average="macro",
                                 labels=classes, zero_division=0)

print("Sinkhorn always assumes %.0f%% unknown and uniform known speakers." % (ASSUMED_UNK * 100))
print("Folds have UNEVEN known counts (1-3 files/speaker) and a true share that moves.\n")
print("=" * 82)
print("%-14s %-9s %-20s %-20s %s" % ("true unknown", "actual", "baseline", "sinkhorn", "paired delta"))
print("=" * 82)

for share in [0.42, 0.46, 0.50, 0.54, 0.58]:
    folds = [make_fold(f, share) for f in range(NFOLDS)]
    act = np.mean([a for *_, a in folds])
    b = np.array([macro(v, baseline(v, i, C)) for v, i, C, _ in folds])
    s = np.array([macro(v, sinkhorn(v, i, C, 0.5, 10)) for v, i, C, _ in folds])
    d = s - b
    sem = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / sem if sem > 0 else 0
    print("  %-12.0f%% %-9.1f%% %.4f +/-%.4f    %.4f +/-%.4f    %+.4f (t=%+.1f)%s"
          % (share * 100, act * 100,
             b.mean(), b.std(ddof=1) / np.sqrt(len(b)),
             s.mean(), s.std(ddof=1) / np.sqrt(len(s)),
             d.mean(), t, "  SIGNIFICANT" if abs(t) > 2.5 else ""))

print()
print("=" * 82)
print("epsilon / iteration sweep at the realistic setting (true share 50%, uneven counts)")
print("=" * 82)
folds = [make_fold(f, 0.50) for f in range(NFOLDS)]
b = np.array([macro(v, baseline(v, i, C)) for v, i, C, _ in folds])
print("  baseline %.4f +/-%.4f" % (b.mean(), b.std(ddof=1) / np.sqrt(len(b))))
for eps in [0.25, 0.35, 0.50, 0.75, 1.0]:
    row = []
    for iters in [3, 10, 25]:
        s = np.array([macro(v, sinkhorn(v, i, C, eps, iters)) for v, i, C, _ in folds])
        row.append("it%-3d %.4f (%+.4f)" % (iters, s.mean(), s.mean() - b.mean()))
    print("  eps=%-5.2f  %s" % (eps, "   ".join(row)))

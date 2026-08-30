"""
Keep the half of the assumption that survived.

Full Sinkhorn imposes TWO constraints: the overall known/unknown split, and
uniformity across the 446 known speakers. The robustness test showed the second
one is false -- real speakers contribute 1-3 test files, not an equal share --
and it is what destroyed the score.

The first constraint is well founded: the contest split every person's audio
roughly 50/50, so about half the eval set is unknown. And it is exactly the thing
our error analysis says is broken (208 false alarms against 101 misses).

So: three variants, all evaluated on the realistic folds.

  quantile  pick tau per test set so the predicted unknown share hits the target.
            One scalar, no per-speaker assumption at all.
  mild      full Sinkhorn but only 3 iterations, a nudge rather than a constraint.
  unkonly   Sinkhorn on a two-column problem (known-mass vs unknown-mass), which
            corrects the split without ever touching the balance among speakers.
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score

O2, O5, O9 = (sys.argv[1:4] + ["_out", "_out5", "_out9"])[:3]
COHORT, NFOLDS = 300, 15
MODELS, WEIGHTS, TAU = ["ecapa", "resnet", "eres2netv2"], [1, 2, 1], 0.25
TARGET_UNK = 0.50

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


def make_fold(f, true_share):
    rng = np.random.RandomState(5000 + f)
    val, trn = [], []
    for s in known:
        idx = list(by[s]); rng.shuffle(idx)
        nh = min(rng.randint(1, 4), max(len(idx) - 2, 0))
        if nh <= 0:
            trn += idx; continue
        val += [(i, s) for i in idx[:nh]]; trn += idx[nh:]
    n_known = len(val)
    n_unk = min(int(round(n_known * true_share / (1 - true_share))), len(unk) - 400)
    p = rng.permutation(len(unk))
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
    return val, vi, C


def guard(val, yp):
    for j, (i, _) in enumerate(val):
        if i in DEGEN:
            yp[j] = "unknown"
    return yp


def decide(val, vi, C, tau):
    sk = C[:, :KU].max(1); su = C[:, KU]; b = C[:, :KU].argmax(1)
    return guard(val, [classes[b[r]] if sk[r] - su[r] > tau else "unknown"
                       for r in range(len(vi))])


def v_baseline(val, vi, C):
    return decide(val, vi, C, TAU)


def v_quantile(val, vi, C, target=TARGET_UNK):
    """Choose tau on this test set so the predicted unknown share hits target."""
    m = C[:, :KU].max(1) - C[:, KU]
    tau = float(np.quantile(m, target))
    return decide(val, vi, C, tau)


def v_mild(val, vi, C, eps=1.0, iters=3, share=TARGET_UNK):
    L = C.copy() / eps
    L[:, KU] += TAU / eps
    n, k = L.shape
    t = np.full(k, (1 - share) / (k - 1)); t[KU] = share
    logc = np.log(t * n)
    for _ in range(iters):
        L -= L.max(1, keepdims=True)
        L -= np.log(np.exp(L).sum(1, keepdims=True))
        L += logc[None, :] - np.log(np.exp(L).sum(0, keepdims=True) + 1e-30)
    return guard(val, [classes[j] for j in L.argmax(1)])


def v_unkonly(val, vi, C, eps=1.0, iters=10, share=TARGET_UNK):
    """Two-column transport: known-mass vs unknown-mass. Shifts the split without
    ever equalising the speakers."""
    sk = C[:, :KU].max(1); su = C[:, KU] + TAU
    L = np.stack([sk, su], 1) / eps
    logc = np.log(np.array([1 - share, share]) * len(sk))
    for _ in range(iters):
        L -= L.max(1, keepdims=True)
        L -= np.log(np.exp(L).sum(1, keepdims=True))
        L += logc[None, :] - np.log(np.exp(L).sum(0, keepdims=True) + 1e-30)
    b = C[:, :KU].argmax(1)
    return guard(val, [classes[b[r]] if L[r, 0] > L[r, 1] else "unknown"
                       for r in range(len(sk))])


macro = lambda val, yp: f1_score([s for _, s in val], yp, average="macro",
                                 labels=classes, zero_division=0)

VARIANTS = {"quantile": v_quantile, "mild sinkhorn (it=3)": v_mild,
            "unknown-mass only": v_unkonly}

print("Realistic folds: 1-3 held-out files per known speaker, true share swept.")
print("Every variant is told 50%% and never told the true share.\n")
print("=" * 88)
print("%-8s %-20s %s" % ("true", "baseline", "paired delta per variant"))
print("=" * 88)

for share in [0.42, 0.46, 0.50, 0.54, 0.58]:
    folds = [make_fold(f, share) for f in range(NFOLDS)]
    b = np.array([macro(v, v_baseline(v, i, C)) for v, i, C in folds])
    line = "  %-6.0f%% %.4f +/-%.4f  " % (share * 100, b.mean(),
                                          b.std(ddof=1) / np.sqrt(len(b)))
    for name, fn in VARIANTS.items():
        s = np.array([macro(v, fn(v, i, C)) for v, i, C in folds])
        d = s - b
        sem = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / sem if sem > 0 else 0
        line += " | %s %+.4f(t=%+.0f)" % (name.split()[0], d.mean(), t)
    print(line, flush=True)

print()
print("=" * 88)
print("If a variant is positive at EVERY true share, the assumption it relies on is safe.")
print("=" * 88)

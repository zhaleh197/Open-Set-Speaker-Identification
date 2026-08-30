"""
Does 3 s chunking beat 6 s, and where does tau move?

Two things to settle, and the second one matters as much as the first.

  1. Is the 3 s representation better, alone or fused with the 6 s one? The two
     share encoders but see the audio at different granularity, so they are also
     a candidate ensemble.

  2. tau = 0.32 was tuned against the LEADERBOARD for the 6 s embeddings, and CV
     is known to place it too low (CV said 0.20, the real optimum is 0.303). If
     3 s chunks shift the margin distribution, shipping 0.32 unchanged would
     confound the comparison. CV cannot give the absolute value, but the SHIFT
     between two representations measured on identical folds is exactly the kind
     of relative quantity it can still be trusted for:

         tau_3s_real  ~=  0.303 + (tau_3s_cv - tau_6s_cv)
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score

O2, O5, O19 = (sys.argv[1:4] + ["_out", "_out5", "_out19"])[:3]
COHORT, NFOLDS = 300, 15
TAUS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
LB_TAU_6S = 0.303          # leaderboard-fitted optimum for the 6 s pipeline

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]
KU = len(classes) - 1

l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
E = {
    "c6_ecapa":  l2(np.load(O2 + "/emb_file.npy").astype("float64")),
    "c6_resnet": l2(np.load(O5 + "/emb_resnet.npy").astype("float64")),
    "c3_ecapa":  l2(np.load(O19 + "/emb_c3_ecapa.npy").astype("float64")),
    "c3_resnet": l2(np.load(O19 + "/emb_c3_resnet.npy").astype("float64")),
}

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
    return (0.5 * ((S - mv) / sv + (S - mt.T) / st.T)).astype("float32")


def make_fold(f, share=0.50):
    """Realistic: uneven held-out counts, matching the eval's structure."""
    rng = np.random.RandomState(4000 + f)
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


print("caching score matrices for %d folds ..." % NFOLDS, flush=True)
FOLDS = [make_fold(f) for f in range(NFOLDS)]
CACHE = {n: [asnorm(v[vi] @ v[trn].T, v[trn], v[vi]) for _, vi, trn in FOLDS]
         for n, v in E.items()}
print("done\n")


def curve(recipe, weights=None):
    w = np.array(weights if weights else [1] * len(recipe), float); w /= w.sum()
    out = {t: [] for t in TAUS}
    for fi, (val, vi, trn) in enumerate(FOLDS):
        S = sum(a * CACHE[n][fi] for a, n in zip(w, recipe))
        tl = np.array([labels[j] for j in trn])
        C = np.full((len(vi), len(classes)), -9e9)
        for ci, c in enumerate(classes):
            cols = np.flatnonzero(tl == c)
            if len(cols):
                C[:, ci] = S[:, cols].max(1)
        sk = C[:, :KU].max(1); su = C[:, KU]; b = C[:, :KU].argmax(1)
        yt = [s for _, s in val]
        for t in TAUS:
            yp = [classes[b[r]] if sk[r] - su[r] > t else "unknown" for r in range(len(vi))]
            out[t].append(f1_score(yt, yp, average="macro", labels=classes, zero_division=0))
    return {t: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))
            for t, v in out.items()}


RECIPES = {
    "6 s  (shipped)":      (["c6_ecapa", "c6_resnet"], [1, 1]),
    "3 s":                 (["c3_ecapa", "c3_resnet"], [1, 1]),
    "3 s, resnet only":    (["c3_resnet"], [1]),
    "6 s + 3 s, all four": (["c6_ecapa", "c6_resnet", "c3_ecapa", "c3_resnet"], [1, 1, 1, 1]),
}

print("=" * 82)
print("%-24s %-14s %-22s" % ("recipe", "best CV tau", "macro-F1 at that tau"))
print("=" * 82)
res = {}
for name, (rec, w) in RECIPES.items():
    c = curve(rec, w)
    bt = max(c, key=lambda t: c[t][0])
    res[name] = (c, bt)
    print("  %-22s %.2f           %.4f +/-%.4f" % (name, bt, c[bt][0], c[bt][1]))

base_c, base_t = res["6 s  (shipped)"]
print()
print("=" * 82)
print("paired against the 6 s pipeline, each at its own best CV tau")
print("=" * 82)
for name, (c, bt) in res.items():
    if name == "6 s  (shipped)":
        continue
    d = np.array(c[bt][0]) - np.array(base_c[base_t][0])
    print("  %-22s %+.4f" % (name, d))

print()
print("=" * 82)
print("tau transfer")
print("=" * 82)
print("  CV optimum, 6 s : %.2f      leaderboard optimum, 6 s : %.3f" % (base_t, LB_TAU_6S))
print("  CV bias         : %+.3f" % (LB_TAU_6S - base_t))
for name, (c, bt) in res.items():
    if name == "6 s  (shipped)":
        continue
    print("  %-22s CV optimum %.2f  ->  ship tau %.3f"
          % (name, bt, LB_TAU_6S + (bt - base_t)))

print()
print("full tau curves (mean macro-F1):")
print("  tau    " + "".join("%-12s" % n[:11] for n in RECIPES))
for t in TAUS:
    print("  %.2f   " % t + "".join("%-12.4f" % res[n][0][t][0] for n in RECIPES))

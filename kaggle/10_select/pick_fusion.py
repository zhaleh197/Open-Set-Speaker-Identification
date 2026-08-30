"""
Evaluate all five encoders under the real protocol, then pick a fusion by greedy
forward selection.

Raw cosine separation is not a usable ranking signal -- WavLM looked terrible by
that measure and was terrible, but the 3D-Speaker models also look worse than
ECAPA by it while sitting in a more anisotropic space that AS-norm is designed to
fix. So every model is scored through the full v2 recipe (AS-norm, margin
threshold, degenerate guard) and judged on macro-F1.

Greedy forward selection with tau re-tuned at each step, scored on the WORSE of
the two holdout protocols so a combo cannot win on one protocol's noise.
"""
import json, collections, itertools, sys
import numpy as np
from sklearn.metrics import f1_score

O2 = sys.argv[1] if len(sys.argv) > 1 else "_out"
O5 = sys.argv[2] if len(sys.argv) > 2 else "_out5"
O9 = sys.argv[3] if len(sys.argv) > 3 else "_out9"
COHORT = 300
TAUS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]

l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
E = {
    "ecapa":      l2(np.load(O2 + "/emb_file.npy").astype("float64")),
    "resnet":     l2(np.load(O5 + "/emb_resnet.npy").astype("float64")),
    "campplus":   l2(np.load(O9 + "/emb_campplus.npy").astype("float64")),
    "eres2netv2": l2(np.load(O9 + "/emb_eres2netv2.npy").astype("float64")),
    "eres2net":   l2(np.load(O9 + "/emb_eres2net.npy").astype("float64")),
}
NAMES = list(E)

by = collections.defaultdict(list)
for i, l in enumerate(labels):
    if valid[i]:
        by[l].append(i)
unk = np.array(by["unknown"])
kp = np.flatnonzero(valid)
G = E["ecapa"][kp] @ E["ecapa"][kp].T
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


def asnorm(S, Et, Ev, n=COHORT):
    Cv, Ct = Ev @ Et.T, Et @ Et.T
    np.fill_diagonal(Ct, -np.inf)
    g = lambda M: (np.sort(M, 1)[:, -n:].mean(1, keepdims=True),
                   np.sort(M, 1)[:, -n:].std(1, keepdims=True) + 1e-9)
    mv, sv = g(Cv); mt, st = g(Ct)
    return (0.5 * ((S - mv) / sv + (S - mt.T) / st.T)).astype("float32")


print("caching AS-norm score matrices ...", flush=True)
CACHE = {}
for nh in (1, 2):
    CACHE[nh] = []
    for val, trn in folds(nh):
        vi = np.array([i for i, _ in val])
        mats = {m: asnorm(E[m][vi] @ E[m][trn].T, E[m][trn], E[m][vi]) for m in NAMES}
        tl = np.array([labels[j] for j in trn])
        CACHE[nh].append((val, vi, tl, tl == "unknown", mats))
    print("  holdout-%d cached" % nh, flush=True)


def score(models, weights, nh, tau):
    w = np.array(weights, dtype=float); w /= w.sum()
    out = []
    for val, vi, tl, isu, mats in CACHE[nh]:
        S = sum(wi * mats[m] for wi, m in zip(w, models))
        Sk = np.where(isu[None, :], -9e9, S)
        Su = np.where(isu[None, :], S, -9e9)
        sk, su, b = Sk.max(1), Su.max(1), Sk.argmax(1)
        yp = [tl[b[r]] if sk[r] - su[r] > tau else "unknown" for r in range(len(vi))]
        for j, (i, _) in enumerate(val):
            if i in DEGEN:
                yp[j] = "unknown"
        out.append(f1_score([s for _, s in val], yp, average="macro",
                            labels=classes, zero_division=0))
    return float(np.mean(out))


def best(models, weights):
    """Return (worse-protocol score, h1, h2, tau) at the best tau."""
    rows = [(min(score(models, weights, 1, t), score(models, weights, 2, t)),
             score(models, weights, 1, t), score(models, weights, 2, t), t)
            for t in TAUS]
    return max(rows, key=lambda r: r[0])


print()
print("=" * 74)
print("SINGLE MODELS  (full v2 recipe, tau tuned)")
print("=" * 74)
singles = {}
for m in NAMES:
    w, h1, h2, t = best([m], [1])
    singles[m] = w
    print("  %-12s worse %.4f   h1 %.4f   h2 %.4f   tau %.2f" % (m, w, h1, h2, t))

print()
print("=" * 74)
print("GREEDY FORWARD SELECTION")
print("=" * 74)
cur = [max(singles, key=singles.get)]
curw = [1.0]
curbest = best(cur, curw)
print("  start: %-22s worse %.4f  (h1 %.4f h2 %.4f, tau %.2f)"
      % ("+".join(cur), curbest[0], curbest[1], curbest[2], curbest[3]))

while len(cur) < len(NAMES):
    cand = None
    for m in NAMES:
        if m in cur:
            continue
        for wnew in [1.0, 2.0, 3.0]:
            r = best(cur + [m], curw + [wnew])
            if cand is None or r[0] > cand[0][0]:
                cand = (r, m, wnew)
    r, m, wnew = cand
    if r[0] <= curbest[0] + 1e-4:
        print("  no further gain -- stopping")
        break
    cur, curw, curbest = cur + [m], curw + [wnew], r
    print("  + %-10s w=%.0f -> %-30s worse %.4f  (h1 %.4f h2 %.4f, tau %.2f)"
          % (m, wnew, "+".join(cur), r[0], r[1], r[2], r[3]))

print()
print("=" * 74)
print("  selected : %s" % " + ".join("%s(w=%.0f)" % (m, w) for m, w in zip(cur, curw)))
print("  tau      : %.2f" % curbest[3])
print("  holdout-1: %.4f     holdout-2: %.4f" % (curbest[1], curbest[2]))
print("  v2 shipped (ecapa+resnet 1:3, tau 0.20): h1 0.9480  h2 0.9544 -> LB 0.9660")
print("=" * 74)
json.dump({"models": cur, "weights": curw, "tau": curbest[3],
           "holdout1": curbest[1], "holdout2": curbest[2], "cohort": COHORT},
          open("selected_config.json", "w"), indent=2)
print("wrote selected_config.json")

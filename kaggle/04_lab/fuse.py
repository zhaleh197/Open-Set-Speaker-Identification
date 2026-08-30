"""
Score-level fusion of the three encoders, evaluated under the holdout-2 protocol.

Each model gets its own AS-norm before fusion. That matters: WavLM's raw cosines
sit in a narrow band near 0.95 (an anisotropic space), so averaging raw scores
would let ECAPA and ResNet drown it out. Normalising first puts all three on a
comparable scale and is what makes the fusion meaningful.
"""
import json, collections, itertools, sys
import numpy as np
from sklearn.metrics import f1_score

OUT2 = sys.argv[1] if len(sys.argv) > 1 else "_out"
OUT5 = sys.argv[2] if len(sys.argv) > 2 else "_out5"

meta = json.load(open(OUT2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]

RAW = {
    "ecapa":  np.load(OUT2 + "/emb_file.npy"),
    "resnet": np.load(OUT5 + "/emb_resnet.npy"),
    "wavlm":  np.load(OUT5 + "/emb_wavlm.npy"),
}


def prep(E, center):
    E = E.astype("float64")
    if center:                                   # kill the anisotropic mean direction
        E = E - E[valid].mean(0, keepdims=True)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


by = collections.defaultdict(list)
for i, l in enumerate(labels):
    if valid[i]:
        by[l].append(i)
unk = np.array(by["unknown"])

# degenerate-embedding guard, derived from ECAPA
_P = prep(RAW["ecapa"], False)[np.flatnonzero(valid)]
_G = _P @ _P.T
np.fill_diagonal(_G, 0)
DEGEN = set(np.flatnonzero(valid)[(_G > 0.9999).sum(1) > 0].tolist())


def folds(nh=2, n=5):
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


def asnorm(S, Et, Ev, n=300):
    Cv, Ct = Ev @ Et.T, Et @ Et.T
    np.fill_diagonal(Ct, -np.inf)
    f = lambda M: (np.sort(M, 1)[:, -n:].mean(1, keepdims=True),
                   np.sort(M, 1)[:, -n:].std(1, keepdims=True) + 1e-9)
    mv, sv = f(Cv); mt, st = f(Ct)
    return 0.5 * ((S - mv) / sv + (S - mt.T) / st.T)


def score_matrices(models, center, use_asnorm, val, trn):
    vi = np.array([i for i, _ in val])
    mats = []
    for name in models:
        E = prep(RAW[name], center.get(name, False))
        S = E[vi] @ E[trn].T
        mats.append(asnorm(S, E[trn], E[vi]) if use_asnorm else S)
    return vi, mats


def run(models, weights=None, center=None, use_asnorm=True, tau=0.0, guard=True):
    center = center or {}
    w = np.array(weights if weights else [1.0] * len(models), dtype=float)
    w = w / w.sum()
    sc = []
    for val, trn in folds():
        vi, mats = score_matrices(models, center, use_asnorm, val, trn)
        S = sum(wi * M for wi, M in zip(w, mats))
        tl = np.array([labels[j] for j in trn])
        isu = tl == "unknown"
        Sk = np.where(isu[None, :], -9e9, S)
        Su = np.where(isu[None, :], S, -9e9)
        sk, su = Sk.max(1), Su.max(1)
        best = Sk.argmax(1)
        yp = [tl[best[r]] if sk[r] - su[r] > tau else "unknown" for r in range(len(vi))]
        if guard:
            for j, (i, _) in enumerate(val):
                if i in DEGEN:
                    yp[j] = "unknown"
        sc.append(f1_score([s for _, s in val], yp, average="macro",
                           labels=classes, zero_division=0))
    return float(np.mean(sc)), float(np.std(sc))


def best_tau(models, taus, **kw):
    out = max(((run(models, tau=t, **kw), t) for t in taus), key=lambda x: x[0][0])
    return out[0][0], out[0][1], out[1]


print("=" * 74)
print("SINGLE MODELS  (AS-norm on, degenerate guard on, tau tuned per model)")
print("=" * 74)
RAWTAUS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
for m in ["ecapa", "resnet", "wavlm"]:
    for cen in [False, True]:
        s, sd, t = best_tau([m], RAWTAUS, center={m: cen})
        print("  %-8s center=%-5s  macro-F1 = %.4f (+/- %.4f)  tau=%.2f"
              % (m, cen, s, sd, t))

print()
print("=" * 74)
print("FUSION")
print("=" * 74)
combos = [("ecapa", "resnet"), ("ecapa", "wavlm"), ("resnet", "wavlm"),
          ("ecapa", "resnet", "wavlm")]
best = (0, None)
for c in combos:
    s, sd, t = best_tau(list(c), RAWTAUS, center={"wavlm": True})
    print("  %-28s macro-F1 = %.4f (+/- %.4f)  tau=%.2f" % ("+".join(c), s, sd, t))
    if s > best[0]:
        best = (s, c, t)

print()
print("weight search on the best combo (%s):" % "+".join(best[1]))
for w in [[2, 1], [1, 2], [3, 1], [1, 3], [2, 1, 1], [1, 1, 2], [2, 2, 1], [3, 2, 1]]:
    if len(w) != len(best[1]):
        continue
    s, sd, t = best_tau(list(best[1]), RAWTAUS, weights=w, center={"wavlm": True})
    print("   weights %-10s macro-F1 = %.4f (+/- %.4f)  tau=%.2f" % (str(w), s, sd, t))

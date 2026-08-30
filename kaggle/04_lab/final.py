"""
Lock the shipping configuration: ecapa+resnet 1:3, AS-norm, degenerate guard.

The margin threshold tau is swept on BOTH holdout protocols. Taking the argmax of
a single noisy curve does not transfer to the leaderboard, so the chosen tau is
the one that maximises the WORSE of the two protocols.
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score

O2 = sys.argv[1] if len(sys.argv) > 1 else "_out"
O5 = sys.argv[2] if len(sys.argv) > 2 else "_out5"

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]

RAW = {"ecapa": np.load(O2 + "/emb_file.npy"),
       "resnet": np.load(O5 + "/emb_resnet.npy")}
P = {k: (lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9))(v.astype("float64"))
     for k, v in RAW.items()}

by = collections.defaultdict(list)
for i, l in enumerate(labels):
    if valid[i]:
        by[l].append(i)
unk = np.array(by["unknown"])

kp = np.flatnonzero(valid)
G = P["ecapa"][kp] @ P["ecapa"][kp].T
np.fill_diagonal(G, 0)
DEGEN = set(kp[(G > 0.9999).sum(1) > 0].tolist())
print("degenerate files guarded:", len(DEGEN))


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


def asnorm(S, Et, Ev, n=300):
    Cv, Ct = Ev @ Et.T, Et @ Et.T
    np.fill_diagonal(Ct, -np.inf)
    g = lambda M: (np.sort(M, 1)[:, -n:].mean(1, keepdims=True),
                   np.sort(M, 1)[:, -n:].std(1, keepdims=True) + 1e-9)
    mv, sv = g(Cv); mt, st = g(Ct)
    return 0.5 * ((S - mv) / sv + (S - mt.T) / st.T)


def run(nh, tau, w=(1, 3)):
    ws = np.array(w, dtype=float); ws = ws / ws.sum()
    sc = []
    for val, trn in folds(nh):
        vi = np.array([i for i, _ in val])
        S = 0
        for wi, m in zip(ws, ["ecapa", "resnet"]):
            S = S + wi * asnorm(P[m][vi] @ P[m][trn].T, P[m][trn], P[m][vi])
        tl = np.array([labels[j] for j in trn])
        isu = tl == "unknown"
        Sk = np.where(isu[None, :], -9e9, S)
        Su = np.where(isu[None, :], S, -9e9)
        sk, su = Sk.max(1), Su.max(1)
        b = Sk.argmax(1)
        yp = [tl[b[r]] if sk[r] - su[r] > tau else "unknown" for r in range(len(vi))]
        for j, (i, _) in enumerate(val):
            if i in DEGEN:
                yp[j] = "unknown"
        sc.append(f1_score([s for _, s in val], yp, average="macro",
                           labels=classes, zero_division=0))
    return float(np.mean(sc)), float(np.std(sc))


print()
print("   tau      holdout-1            holdout-2")
rows = []
for t in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55]:
    a = run(1, t); b = run(2, t)
    rows.append((t, a[0], b[0]))
    print("   %.2f   %.4f +/-%.4f   %.4f +/-%.4f" % (t, a[0], a[1], b[0], b[1]), flush=True)

best = max(rows, key=lambda r: min(r[1], r[2]))
print()
print("   most robust tau = %.2f  ->  holdout-1 %.4f   holdout-2 %.4f"
      % (best[0], best[1], best[2]))
json.dump({"tau": best[0], "weights": {"ecapa": 1, "resnet": 3},
           "holdout1": best[1], "holdout2": best[2], "asnorm_cohort": 300},
          open("final_config.json", "w"), indent=2)
print("   wrote final_config.json")

"""
High-power comparison of the shortlisted configurations.

Every remaining difference is around 0.003, while 5 folds give a standard error
of the mean of roughly the same size. That means the last few decisions were
being made on noise. This runs 20 folds per protocol and reports the standard
ERROR (not the spread), plus a paired test against the shipped v2 config so the
comparison uses the same folds for both arms rather than two independent draws.

Paired comparison matters here: fold-to-fold variation is large but shared, so
the difference between two configs on the SAME fold is far less noisy than the
difference of their averages.
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score

O2, O5, O9 = (sys.argv[1:4] + ["_out", "_out5", "_out9"])[:3]
COHORT, NFOLDS = 300, 20

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]

l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
E = {"ecapa": l2(np.load(O2 + "/emb_file.npy").astype("float64")),
     "resnet": l2(np.load(O5 + "/emb_resnet.npy").astype("float64")),
     "campplus": l2(np.load(O9 + "/emb_campplus.npy").astype("float64")),
     "eres2netv2": l2(np.load(O9 + "/emb_eres2netv2.npy").astype("float64")),
     "eres2net": l2(np.load(O9 + "/emb_eres2net.npy").astype("float64"))}

by = collections.defaultdict(list)
for i, l in enumerate(labels):
    if valid[i]:
        by[l].append(i)
unk = np.array(by["unknown"])
kp = np.flatnonzero(valid)
G = E["ecapa"][kp] @ E["ecapa"][kp].T
np.fill_diagonal(G, 0)
DEGEN = set(kp[(G > 0.9999).sum(1) > 0].tolist())

CONFIGS = {
    "v2 shipped  (eca1+res3)":        (["ecapa", "resnet"], [1, 3], 0.20),
    "greedy4     (eca+e2v2+res+e2n)": (["ecapa", "eres2netv2", "resnet", "eres2net"],
                                       [1, 1, 1, 1], 0.40),
    "eca+e2v2":                       (["ecapa", "eres2netv2"], [1, 1], 0.30),
    "eca+res+e2v2":                   (["ecapa", "resnet", "eres2netv2"], [1, 2, 1], 0.25),
    "all five":                       (list(E), [1, 1, 1, 1, 1], 0.40),
}


def asnorm(S, Et, Ev, n=COHORT):
    Cv, Ct = Ev @ Et.T, Et @ Et.T
    np.fill_diagonal(Ct, -np.inf)
    g = lambda M: (np.sort(M, 1)[:, -n:].mean(1, keepdims=True),
                   np.sort(M, 1)[:, -n:].std(1, keepdims=True) + 1e-9)
    mv, sv = g(Cv); mt, st = g(Ct)
    return (0.5 * ((S - mv) / sv + (S - mt.T) / st.T)).astype("float32")


def one_fold(f, nh):
    rng = np.random.RandomState(1000 + f)
    val, trn = [], []
    for s in known:
        idx = list(by[s]); rng.shuffle(idx)
        if len(idx) <= nh:
            trn += idx; continue
        val += [(i, s) for i in idx[:nh]]; trn += idx[nh:]
    p = rng.permutation(len(unk)); m = min(len(val), len(unk) // 2)
    val += [(unk[j], "unknown") for j in p[:m]]; trn += list(unk[p[m:]])
    trn = np.array(trn)
    vi = np.array([i for i, _ in val])
    mats = {k: asnorm(E[k][vi] @ E[k][trn].T, E[k][trn], E[k][vi]) for k in E}
    tl = np.array([labels[j] for j in trn])
    return val, vi, tl, tl == "unknown", mats


def apply(cfg, fold):
    models, w, tau = cfg
    val, vi, tl, isu, mats = fold
    ws = np.array(w, float); ws /= ws.sum()
    S = sum(a * mats[m] for a, m in zip(ws, models))
    Sk = np.where(isu[None, :], -9e9, S)
    Su = np.where(isu[None, :], S, -9e9)
    sk, su, b = Sk.max(1), Su.max(1), Sk.argmax(1)
    yp = [tl[b[r]] if sk[r] - su[r] > tau else "unknown" for r in range(len(vi))]
    for j, (i, _) in enumerate(val):
        if i in DEGEN:
            yp[j] = "unknown"
    return f1_score([s for _, s in val], yp, average="macro",
                    labels=classes, zero_division=0)


for nh in (1, 2):
    print("\n" + "=" * 78)
    print("HOLDOUT-%d   %d folds   (mean +/- standard error of the mean)" % (nh, NFOLDS))
    print("=" * 78)
    per = {k: [] for k in CONFIGS}
    for f in range(NFOLDS):
        fold = one_fold(f, nh)
        for k, cfg in CONFIGS.items():
            per[k].append(apply(cfg, fold))
        print("   fold %2d/%d" % (f + 1, NFOLDS), end="\r", flush=True)
    base = np.array(per["v2 shipped  (eca1+res3)"])
    print(" " * 30, end="\r")
    for k, v in per.items():
        v = np.array(v)
        sem = v.std(ddof=1) / np.sqrt(len(v))
        d = v - base
        dsem = d.std(ddof=1) / np.sqrt(len(d)) if k != "v2 shipped  (eca1+res3)" else 0
        tag = ""
        if k != "v2 shipped  (eca1+res3)":
            t = d.mean() / dsem if dsem > 0 else 0
            tag = "   paired delta %+.4f +/-%.4f  (t=%+.1f)%s" % (
                d.mean(), dsem, t, "  SIGNIFICANT" if abs(t) > 2.5 else "")
        print("  %-32s %.4f +/-%.4f%s" % (k, v.mean(), sem, tag))

"""
Which short-file policy is right?

194 files (4.3%) have under 6 s of speech, median 2.67 s. The extraction run
confirmed the policies genuinely disagree about them, while leaving every long
file bit-identical:

    pad      cosine to tile  median 0.776   p10 0.263   min -0.010
    reflect                  median 0.958
    raw                      median 0.988

Zero-padding produces an essentially unrelated vector for some of these files --
one pair is orthogonal. So the choice is not cosmetic. This decides it on the
metric, and separately on the short files alone, because 4.3% of the set can only
move macro-F1 so far and the global number alone would be too noisy to read.
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score

O25 = sys.argv[1] if len(sys.argv) > 1 else "_out25"
O2 = sys.argv[2] if len(sys.argv) > 2 else "_out"
COHORT, NFOLDS = 300, 15
TAUS = [0.10, 0.20, 0.25, 0.32, 0.40]
POLICIES = ["tile", "pad", "reflect", "raw"]

meta = json.load(open(O2 + "/meta.json"))
labels = meta["labels"]
short = np.array(json.load(open(O25 + "/meta_short.json"))["is_short"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]
KU = len(classes) - 1

l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
BANK = {p: {m: l2(np.load("%s/emb_%s_%s.npy" % (O25, p, m)).astype("float64"))
            for m in ["ecapa", "resnet"]} for p in POLICIES}
valid = np.abs(BANK["tile"]["ecapa"]).sum(1) > 1e-6
print("usable files: %d   short: %d" % (valid.sum(), int((short & valid).sum())))

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
    rng = np.random.RandomState(1500 + f)
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
print("folds built\n")


def evaluate(policy, tau):
    E = BANK[policy]
    macro, short_hit, short_n = [], 0, 0
    for val, vi, trn in FOLDS:
        S = sum(0.5 * asnorm(E[m][vi] @ E[m][trn].T, E[m][trn], E[m][vi])
                for m in ["ecapa", "resnet"])
        tl = np.array([labels[j] for j in trn])
        C = np.full((len(vi), len(classes)), -9e9)
        for ci, c in enumerate(classes):
            cols = np.flatnonzero(tl == c)
            if len(cols):
                C[:, ci] = S[:, cols].max(1)
        sk = C[:, :KU].max(1); su = C[:, KU]; b = C[:, :KU].argmax(1)
        yt = [s for _, s in val]
        yp = [classes[b[r]] if sk[r] - su[r] > tau else "unknown" for r in range(len(vi))]
        macro.append(f1_score(yt, yp, average="macro", labels=classes, zero_division=0))
        for r, i in enumerate(vi):
            if short[i]:
                short_n += 1
                short_hit += (yp[r] == yt[r])
    return np.array(macro), short_hit / max(short_n, 1)


print("=" * 78)
print("%-10s %-28s %s" % ("policy", "macro-F1 (best tau)", "accuracy on SHORT files"))
print("=" * 78)
res = {}
for p in POLICIES:
    best = max(((evaluate(p, t), t) for t in TAUS), key=lambda x: x[0][0].mean())
    (mac, sacc), tau = best
    res[p] = mac
    print("  %-8s %.4f +/-%.4f  (tau %.2f)      %.1f%%"
          % (p, mac.mean(), mac.std(ddof=1) / np.sqrt(len(mac)), tau, sacc * 100))

base = res["tile"]
print()
print("=" * 78)
print("paired against tile, the policy that ships today")
print("=" * 78)
for p in POLICIES[1:]:
    d = res[p] - base
    sem = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / sem if sem > 0 else 0
    print("  %-8s %+.4f (t=%+.1f)%s"
          % (p, d.mean(), t, "  SIGNIFICANT" if abs(t) > 2.5 else ""))

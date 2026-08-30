"""
Balanced label assignment by Sinkhorn / optimal transport.

The idea. Right now every test file is labelled independently by argmax. But we
know something about the test set AS A WHOLE that argmax throws away: roughly
half of it is "unknown", and the known speakers appear in roughly equal numbers
(the contest split each person's audio ~50/50, and every known speaker has a
comparable amount). Independent argmax has no way to honour that, and the error
analysis shows it does not: 208 false alarms against 101 misses -- the predicted
label distribution is skewed toward claiming a known identity.

Sinkhorn turns the score matrix into a soft assignment that matches BOTH
marginals: every file gets one label (row sums), and the label totals match the
expected class proportions (column sums). It is the same machinery used for
balanced cluster assignment in self-supervised learning (SwAV) and for
transductive few-shot classification.

Why this is not the transductive clustering that already failed. That attempt
clustered test files and propagated one label across each cluster, which merged
different speakers and destroyed accuracy. This never groups test files together
and never moves a label from one file to another; it only reweights the existing
scores so the totals come out right.

Honest risk: it assumes the test marginals. If the real eval is not ~50% unknown
or the known speakers are not roughly balanced, this pushes the wrong way. The
epsilon sweep below covers that -- large epsilon is nearly a no-op, small epsilon
enforces the marginals hard, so the sweep shows how much the assumption is worth.
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score

O2, O5, O9 = (sys.argv[1:4] + ["_out", "_out5", "_out9"])[:3]
COHORT, NFOLDS = 300, 20
MODELS, WEIGHTS, TAU = ["ecapa", "resnet", "eres2netv2"], [1, 2, 1], 0.25

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]
cls_index = {c: i for i, c in enumerate(classes)}

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


def make_fold(f, nh=2):
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
    w = np.array(WEIGHTS, float); w /= w.sum()
    S = sum(a * asnorm(E[m][vi] @ E[m][trn].T, E[m][trn], E[m][vi])
            for a, m in zip(w, MODELS))
    tl = np.array([labels[j] for j in trn])

    # collapse prototypes to classes: best prototype per class
    C = np.full((len(vi), len(classes)), -9e9)
    for ci, c in enumerate(classes):
        cols = np.flatnonzero(tl == c)
        if len(cols):
            C[:, ci] = S[:, cols].max(1)
    return val, vi, C


def baseline(val, vi, C):
    """The v2 rule, expressed on the class-collapsed matrix."""
    ku = cls_index["unknown"]
    sk = C[:, :ku].max(1); su = C[:, ku]
    b = C[:, :ku].argmax(1)
    yp = [classes[b[r]] if sk[r] - su[r] > TAU else "unknown" for r in range(len(vi))]
    for j, (i, _) in enumerate(val):
        if i in DEGEN:
            yp[j] = "unknown"
    return yp


def sinkhorn(val, vi, C, eps, iters, unk_share, tau_shift=0.0):
    """Match row marginals (one label per file) and column marginals (expected
    class proportions) by alternating normalisation in log space."""
    ku = cls_index["unknown"]
    L = C.copy() / eps
    L[:, ku] += tau_shift / eps          # the margin threshold, folded into the cost
    n, k = L.shape
    target = np.full(k, (1.0 - unk_share) / (k - 1))
    target[ku] = unk_share
    logc = np.log(target * n)
    for _ in range(iters):
        L -= L.max(1, keepdims=True)
        L -= np.log(np.exp(L).sum(1, keepdims=True))          # rows sum to 1
        colsum = np.log(np.exp(L).sum(0, keepdims=True) + 1e-30)
        L += (logc[None, :] - colsum)                          # columns hit target
    yp = [classes[j] for j in L.argmax(1)]
    for j, (i, _) in enumerate(val):
        if i in DEGEN:
            yp[j] = "unknown"
    return yp


def macro(val, yp):
    return f1_score([s for _, s in val], yp, average="macro",
                    labels=classes, zero_division=0)


print("building %d folds ..." % NFOLDS, flush=True)
FOLDS = [make_fold(f) for f in range(NFOLDS)]
print("done\n")

base = np.array([macro(v, baseline(v, i, C)) for v, i, C in FOLDS])
print("=" * 78)
print("baseline (v2 rule, 3-model fusion)   %.4f +/-%.4f"
      % (base.mean(), base.std(ddof=1) / np.sqrt(len(base))))
print("=" * 78)
print("%-42s %-18s %s" % ("sinkhorn config", "macro-F1", "paired delta"))
print("-" * 78)

best = (0, None)
for eps in [0.5, 1.0, 2.0, 4.0]:
    for iters in [1, 3, 10]:
        v = np.array([macro(vv, sinkhorn(vv, ii, C, eps, iters, 0.5, TAU))
                      for vv, ii, C in FOLDS])
        d = v - base
        sem = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / sem if sem > 0 else 0
        flag = "  SIGNIFICANT" if abs(t) > 2.5 else ""
        print("  eps=%-5.1f iters=%-3d                        %.4f +/-%.4f   %+.4f (t=%+.1f)%s"
              % (eps, iters, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), d.mean(), t, flag))
        if v.mean() > best[0]:
            best = (v.mean(), (eps, iters))

print("-" * 78)
print("best: %s at macro-F1 %.4f" % (best[1], best[0]))

if best[1]:
    eps, iters = best[1]
    print("\nsensitivity to the assumed unknown share (the risky assumption):")
    for share in [0.40, 0.45, 0.50, 0.55, 0.60]:
        v = np.array([macro(vv, sinkhorn(vv, ii, C, eps, iters, share, TAU))
                      for vv, ii, C in FOLDS])
        print("   assumed unknown=%.0f%%   macro-F1 %.4f   (delta vs baseline %+.4f)"
              % (share * 100, v.mean(), v.mean() - base.mean()))

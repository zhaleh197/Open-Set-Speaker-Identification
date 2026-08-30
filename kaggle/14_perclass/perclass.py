"""
Per-speaker decision bias, derived from the structure of macro-F1 itself.

The idea nobody optimising embeddings will reach for.

Lipton & Elkan: for calibrated scores the F1-optimal decision threshold for a
class equals half that class's achievable F1. A class that can reach F1 0.9 should
be judged at 0.45; a class that can only reach 0.3 should be judged at 0.15. Weak
classes deserve MORE permissive thresholds, not less. We currently apply one
global margin tau to all 446 speakers, which is the wrong shape.

Why this is exploitable here specifically. Normally per-class thresholds are
useless out of sample -- you are fitting the validation split. But this contest
fixes the class set: the same 446 speakers appear in cross-validation and in the
real eval. "Speaker 173 is hard" is a property of that person's recordings, not
of my fold construction, so it transfers.

And macro-F1 makes the bet extremely asymmetric. A speaker with ~4 eval files
that currently receives zero predictions scores F1 = 0. Force one plausible file
onto it:
    right -> precision 1.0, recall 0.25, F1 0.40, worth +0.0009 macro
    wrong -> that class stays at 0, and 'unknown' loses 1 of ~1800 hits, ~0.00002
Roughly a 45:1 payoff. Rescuing ten starved speakers is the whole gap to first.

Protocol note: biases are estimated on folds 0-9 and applied to folds 10-19, so
nothing is evaluated on the folds that produced it.
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score

O2, O5 = (sys.argv[1:3] + ["_out", "_out5"])[:2]
COHORT = 300
MODELS, WEIGHTS, TAU = ["ecapa", "resnet"], [1, 3], 0.20
N_CAL, N_EVAL = 10, 10

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]
KU = len(classes) - 1

l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
E = {"ecapa": l2(np.load(O2 + "/emb_file.npy").astype("float64")),
     "resnet": l2(np.load(O5 + "/emb_resnet.npy").astype("float64"))}
WN = np.array(WEIGHTS, float); WN /= WN.sum()

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


def make_fold(f, share=0.50):
    """Realistic: uneven held-out counts per speaker, like the real eval."""
    rng = np.random.RandomState(7000 + f)
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
    trn = np.array(trn)
    vi = np.array([i for i, _ in val])
    S = sum(a * asnorm(E[m][vi] @ E[m][trn].T, E[m][trn], E[m][vi])
            for a, m in zip(WN, MODELS))
    tl = np.array([labels[j] for j in trn])
    C = np.full((len(vi), len(classes)), -9e9)
    for ci, c in enumerate(classes):
        cols = np.flatnonzero(tl == c)
        if len(cols):
            C[:, ci] = S[:, cols].max(1)
    return val, vi, C


def predict(val, vi, C, bias=None, rescue=0):
    B = np.zeros(len(classes)) if bias is None else bias
    A = C + B[None, :]
    sk = A[:, :KU].max(1); su = A[:, KU]; b = A[:, :KU].argmax(1)
    yp = [classes[b[r]] if sk[r] - su[r] > TAU else "unknown" for r in range(len(vi))]

    if rescue > 0:
        # Any known speaker with zero predictions scores F1 = 0 no matter what.
        # Hand it its best available candidate from the files currently called
        # unknown. If the guess is wrong the class stays at 0 and we lose one hit
        # out of ~1800 on 'unknown'; if it is right the class jumps to ~0.4.
        got = collections.Counter(yp)
        free = [r for r in range(len(vi)) if yp[r] == "unknown"]
        for ci, c in enumerate(known):
            if got[c] > 0 or not free:
                continue
            cand = sorted(free, key=lambda r: -C[r, ci])[:rescue]
            for r in cand:
                yp[r] = c
                free.remove(r)

    for j, (i, _) in enumerate(val):
        if i in DEGEN:
            yp[j] = "unknown"
    return yp


macro = lambda val, yp: f1_score([s for _, s in val], yp, average="macro",
                                 labels=classes, zero_division=0)


def per_class_f1(val, yp):
    yt = [s for _, s in val]
    return f1_score(yt, yp, average=None, labels=classes, zero_division=0)


print("building %d calibration + %d evaluation folds ..." % (N_CAL, N_EVAL), flush=True)
CAL = [make_fold(f) for f in range(N_CAL)]
EVA = [make_fold(100 + f) for f in range(N_EVAL)]

# ---- estimate each speaker's achievable F1 on the calibration folds only -----
acc = np.zeros(len(classes)); n = 0
for val, vi, C in CAL:
    acc += per_class_f1(val, predict(val, vi, C))
    n += 1
F = acc / n
print("\nper-speaker F1 estimated on calibration folds:")
kf = F[:KU]
print("  mean %.3f | median %.3f | p10 %.3f | speakers at 0.00: %d | below 0.5: %d"
      % (kf.mean(), np.median(kf), np.percentile(kf, 10), (kf == 0).sum(), (kf < 0.5).sum()))

base = np.array([macro(v, predict(v, i, C)) for v, i, C in EVA])
print("\n" + "=" * 78)
print("baseline (global tau, no per-class bias)   %.4f +/-%.4f"
      % (base.mean(), base.std(ddof=1) / np.sqrt(len(base))))
print("=" * 78)

print("\nA. per-speaker bias  b_c = -alpha * log(F_c)   [Lipton-Elkan shape]")
for alpha in [0.05, 0.1, 0.2, 0.35, 0.5, 0.8]:
    B = np.zeros(len(classes))
    B[:KU] = -alpha * np.log(np.clip(F[:KU], 0.05, 1.0))
    v = np.array([macro(vv, predict(vv, ii, C, bias=B)) for vv, ii, C in EVA])
    d = v - base
    sem = d.std(ddof=1) / np.sqrt(len(d))
    print("   alpha=%-5.2f  %.4f +/-%.4f   %+.4f (t=%+.1f)%s"
          % (alpha, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), d.mean(),
             d.mean() / sem if sem > 0 else 0,
             "  SIGNIFICANT" if sem > 0 and abs(d.mean() / sem) > 2.5 else ""))

print("\nB. starved-class rescue: force k files onto every speaker with none")
for k in [1, 2, 3]:
    v = np.array([macro(vv, predict(vv, ii, C, rescue=k)) for vv, ii, C in EVA])
    d = v - base
    sem = d.std(ddof=1) / np.sqrt(len(d))
    print("   k=%-2d  %.4f +/-%.4f   %+.4f (t=%+.1f)%s"
          % (k, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), d.mean(),
             d.mean() / sem if sem > 0 else 0,
             "  SIGNIFICANT" if sem > 0 and abs(d.mean() / sem) > 2.5 else ""))

print("\nC. both together")
for alpha in [0.1, 0.2, 0.35]:
    B = np.zeros(len(classes))
    B[:KU] = -alpha * np.log(np.clip(F[:KU], 0.05, 1.0))
    for k in [1, 2]:
        v = np.array([macro(vv, predict(vv, ii, C, bias=B, rescue=k)) for vv, ii, C in EVA])
        d = v - base
        sem = d.std(ddof=1) / np.sqrt(len(d))
        print("   alpha=%-5.2f k=%-2d  %.4f   %+.4f (t=%+.1f)%s"
              % (alpha, k, v.mean(), d.mean(), d.mean() / sem if sem > 0 else 0,
                 "  SIGNIFICANT" if sem > 0 and abs(d.mean() / sem) > 2.5 else ""))

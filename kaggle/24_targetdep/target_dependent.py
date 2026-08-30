"""
Target-dependent scoring: one operating point per speaker, not one for all 446.

The pipeline applies a single global margin tau to every known speaker. That
assumes all 446 voices are equally distinctive, which they are not. Some speakers
sit alone in the embedding space and could be accepted at a low threshold safely;
others live in a crowded neighbourhood and need a high one. A global tau is a
compromise that is wrong for both.

Everything needed to set a per-speaker operating point is in the training data
and involves no leakage:

    genuine  distribution  leave-one-out among that speaker's own 5 files
    impostor distribution  that speaker's prototypes against every other
                           training file

Three ways to use it, from mildest to most committed:

  z-norm     rescale each speaker's scores by its own impostor mean and spread.
             This is classic T-norm. AS-norm already normalises against a cohort,
             but by the PROTOTYPE's top-N neighbours, which is not the same as the
             speaker's full impostor distribution.
  midpoint   place each speaker's boundary halfway between its genuine and
             impostor means -- a per-speaker decision boundary rather than a
             per-speaker rescaling.
  d-prime    offset each speaker by how separable it actually is,
             (mu_gen - mu_imp) / sigma. A speaker that separates well earns a
             more permissive threshold; one that does not earns a stricter one.

Evaluated on realistic folds, with every statistic computed from the fold's
training portion only.
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score

O2, O5 = (sys.argv[1:3] + ["_out", "_out5"])[:2]
COHORT, NFOLDS = 300, 15
TAUS = [0.05, 0.15, 0.25, 0.35, 0.50]

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]
KU = len(classes) - 1

l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
E = {"ecapa": l2(np.load(O2 + "/emb_file.npy").astype("float64")),
     "resnet": l2(np.load(O5 + "/emb_resnet.npy").astype("float64"))}

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


def collapse(S, trn_labels):
    C = np.full((S.shape[0], len(classes)), -9e9)
    for ci, c in enumerate(classes):
        cols = np.flatnonzero(trn_labels == c)
        if len(cols):
            C[:, ci] = S[:, cols].max(1)
    return C


def make_fold(f, share=0.50):
    rng = np.random.RandomState(2500 + f)
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
    tl = np.array([labels[j] for j in trn])

    Sv = sum(0.5 * asnorm(E[m][vi] @ E[m][trn].T, E[m][trn], E[m][vi])
             for m in ["ecapa", "resnet"])
    Cv = collapse(Sv, tl)

    # training-only statistics: every training file scored against every class,
    # with a file's own prototype masked so nothing scores against itself
    St = sum(0.5 * asnorm(E[m][trn] @ E[m][trn].T, E[m][trn], E[m][trn])
             for m in ["ecapa", "resnet"])
    np.fill_diagonal(St, -9e9)
    Ct = collapse(St, tl)

    stats = np.zeros((len(classes), 4))          # mu_gen, mu_imp, sd_imp, dprime
    for ci, c in enumerate(known):
        own = tl == c
        gen = Ct[own, ci]
        imp = Ct[~own, ci]
        gen = gen[gen > -1e8]; imp = imp[imp > -1e8]
        if len(gen) == 0 or len(imp) < 10:
            stats[ci] = [0, 0, 1, 0]
            continue
        mg, mi, si = gen.mean(), imp.mean(), imp.std() + 1e-9
        sg = gen.std() + 1e-9
        stats[ci] = [mg, mi, si, (mg - mi) / np.sqrt(0.5 * (sg ** 2 + si ** 2))]
    return val, vi, Cv, stats


print("building %d folds (statistics from training only) ..." % NFOLDS, flush=True)
FOLDS = [make_fold(f) for f in range(NFOLDS)]
dp = np.concatenate([s[:KU, 3] for *_, s in FOLDS])
print("per-speaker separability (d-prime): p10 %.2f  median %.2f  p90 %.2f"
      % (np.percentile(dp, 10), np.median(dp), np.percentile(dp, 90)))
print("  a wide spread here is the whole premise: speakers are not equally easy\n")

macro = lambda val, yp: f1_score([s for _, s in val], yp, average="macro",
                                 labels=classes, zero_division=0)


def decide(val, vi, C, tau):
    sk = C[:, :KU].max(1); su = C[:, KU]; b = C[:, :KU].argmax(1)
    return [classes[b[r]] if sk[r] - su[r] > tau else "unknown" for r in range(len(vi))]


def variant(name, tau, alpha=0.0):
    out = []
    for val, vi, C, st in FOLDS:
        A = C.copy()
        if name == "znorm":
            A[:, :KU] = (A[:, :KU] - st[:KU, 1][None, :]) / st[:KU, 2][None, :]
        elif name == "midpoint":
            A[:, :KU] = A[:, :KU] - alpha * (0.5 * (st[:KU, 0] + st[:KU, 1]))[None, :]
        elif name == "dprime":
            d = st[:KU, 3]
            A[:, :KU] = A[:, :KU] + alpha * (d - d.mean())[None, :]
        out.append(macro(val, decide(val, vi, A, tau)))
    return np.array(out)


base = max((variant("base", t).mean(), t) for t in TAUS)
base_v = variant("base", base[1])
print("=" * 76)
print("baseline, global tau = %.2f          %.4f +/-%.4f"
      % (base[1], base_v.mean(), base_v.std(ddof=1) / np.sqrt(len(base_v))))
print("=" * 76)


def report(label, v):
    d = v - base_v
    sem = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / sem if sem > 0 else 0
    print("  %-44s %.4f   %+.4f (t=%+.1f)%s"
          % (label, v.mean(), d.mean(), t, "  SIGNIFICANT" if abs(t) > 2.5 else ""))


print("\nA. per-speaker impostor z-norm (T-norm on top of AS-norm)")
for t in [0.0, 0.25, 0.5, 1.0, 1.5]:
    report("znorm, tau=%.2f" % t, variant("znorm", t))

print("\nB. per-speaker genuine/impostor midpoint boundary")
for a in [0.25, 0.5, 1.0]:
    best = max((variant("midpoint", t, a).mean(), t) for t in TAUS)
    report("midpoint alpha=%.2f (tau=%.2f)" % (a, best[1]),
           variant("midpoint", best[1], a))

print("\nC. per-speaker d-prime offset: separable speakers get a looser threshold")
for a in [0.05, 0.1, 0.2, 0.4]:
    best = max((variant("dprime", t, a).mean(), t) for t in TAUS)
    report("dprime alpha=%.2f (tau=%.2f)" % (a, best[1]),
           variant("dprime", best[1], a))

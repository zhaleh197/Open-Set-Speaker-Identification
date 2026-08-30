"""
Judge the new front-end on the metric, not on separation.

Three questions:

  1. Does the neural VAD beat the energy rule outright?
  2. The two front-ends share encoders but differ in what audio reaches them, so
     they make different mistakes. Does fusing them beat either alone? This is
     ensemble diversity for free -- no new model, just a second reading of the
     same audio.
  3. The VAD flags 83 files as almost pure non-speech. That is a principled
     replacement for the 34-file cosine hack, which only ever caught the extreme
     cases. Does routing those files straight to "unknown" help more?
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score

O2, O5, O17 = (sys.argv[1:4] + ["_out", "_out5", "_out17"])[:3]
COHORT, NFOLDS, TAU = 300, 15, 0.20

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]
KU = len(classes) - 1

vmeta = json.load(open(O17 + "/meta_vad.json"))
speech_s = np.array(vmeta["speech_seconds"])
vad_prob = np.array(vmeta["vad_prob"])

l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
BANK = {
    "old_ecapa":  l2(np.load(O2 + "/emb_file.npy").astype("float64")),
    "old_resnet": l2(np.load(O5 + "/emb_resnet.npy").astype("float64")),
    "vad_ecapa":  l2(np.load(O17 + "/emb_vad_ecapa.npy").astype("float64")),
    "vad_resnet": l2(np.load(O17 + "/emb_vad_resnet.npy").astype("float64")),
}
# a file is usable only if every encoder in a recipe produced something
USABLE = {k: (np.abs(v).sum(1) > 1e-6) for k, v in BANK.items()}

by = collections.defaultdict(list)
for i, l in enumerate(labels):
    if valid[i]:
        by[l].append(i)
unk = np.array(by["unknown"])

kp = np.flatnonzero(valid)
G = BANK["old_ecapa"][kp] @ BANK["old_ecapa"][kp].T
np.fill_diagonal(G, 0)
DEGEN = set(kp[(G > 0.9999).sum(1) > 0].tolist())
NONSPEECH = set(np.flatnonzero((vad_prob >= 0) & (vad_prob < 0.2)).tolist())
print("guards: cosine-hack flags %d files, VAD flags %d as non-speech, overlap %d"
      % (len(DEGEN), len(NONSPEECH), len(DEGEN & NONSPEECH)))


def asnorm(S, Et, Ev, n=COHORT):
    Cv, Ct = Ev @ Et.T, Et @ Et.T
    np.fill_diagonal(Ct, -np.inf)
    g = lambda M: (np.sort(M, 1)[:, -n:].mean(1, keepdims=True),
                   np.sort(M, 1)[:, -n:].std(1, keepdims=True) + 1e-9)
    mv, sv = g(Cv); mt, st = g(Ct)
    return 0.5 * ((S - mv) / sv + (S - mt.T) / st.T)


def make_fold(f, share=0.50):
    rng = np.random.RandomState(3000 + f)
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
CACHE = {}
for name, E in BANK.items():
    CACHE[name] = [asnorm(E[vi] @ E[trn].T, E[trn], E[vi]) for _, vi, trn in FOLDS]
print("score matrices cached\n")


def evaluate(recipe, guard_set):
    """recipe: {embedding name: weight}"""
    w = np.array(list(recipe.values()), float); w /= w.sum()
    out = []
    for fi, (val, vi, trn) in enumerate(FOLDS):
        S = sum(a * CACHE[n][fi] for a, n in zip(w, recipe))
        tl = np.array([labels[j] for j in trn])
        C = np.full((len(vi), len(classes)), -9e9)
        for ci, c in enumerate(classes):
            cols = np.flatnonzero(tl == c)
            if len(cols):
                C[:, ci] = S[:, cols].max(1)
        sk = C[:, :KU].max(1); su = C[:, KU]; b = C[:, :KU].argmax(1)
        yp = [classes[b[r]] if sk[r] - su[r] > TAU else "unknown" for r in range(len(vi))]
        for j, (i, _) in enumerate(val):
            if i in guard_set:
                yp[j] = "unknown"
        out.append(f1_score([s for _, s in val], yp, average="macro",
                            labels=classes, zero_division=0))
    return np.array(out)


RECIPES = {
    "v2 shipped: old ecapa+resnet 1:3": {"old_ecapa": 1, "old_resnet": 3},
    "VAD only:   vad ecapa+resnet 1:3": {"vad_ecapa": 1, "vad_resnet": 3},
    "both front-ends, all four":        {"old_ecapa": 1, "old_resnet": 3,
                                         "vad_ecapa": 1, "vad_resnet": 3},
    "both, resnet-heavy":               {"old_resnet": 2, "vad_resnet": 2,
                                         "old_ecapa": 1, "vad_ecapa": 1},
}

base = evaluate(RECIPES["v2 shipped: old ecapa+resnet 1:3"], DEGEN)
print("=" * 80)
print("%-40s %-20s %s" % ("recipe (guard: cosine hack)", "macro-F1", "paired vs v2"))
print("=" * 80)
for name, rec in RECIPES.items():
    v = evaluate(rec, DEGEN)
    d = v - base
    sem = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / sem if sem > 0 else 0
    print("  %-38s %.4f +/-%.4f  %+.4f (t=%+.1f)%s"
          % (name, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), d.mean(), t,
             "  SIGNIFICANT" if abs(t) > 2.5 else ""))

print()
print("=" * 80)
print("guard comparison, on the best recipe")
print("=" * 80)
best_name = max(RECIPES, key=lambda k: evaluate(RECIPES[k], DEGEN).mean())
rec = RECIPES[best_name]
print("  using: %s\n" % best_name)
for gname, g in [("cosine hack (34 files)", DEGEN),
                 ("VAD non-speech (83 files)", NONSPEECH),
                 ("both", DEGEN | NONSPEECH),
                 ("none", set())]:
    v = evaluate(rec, g)
    d = v - base
    sem = d.std(ddof=1) / np.sqrt(len(d))
    print("  %-28s %.4f +/-%.4f  %+.4f (t=%+.1f)"
          % (gname, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), d.mean(),
             d.mean() / sem if sem > 0 else 0))

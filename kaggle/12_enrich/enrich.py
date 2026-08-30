"""
Prototype enrichment: let the test set enrol itself.

Each known speaker has 5 training files. The eval set adds roughly 3-4 more
recordings of that same person. Right now those test files are only ever queries
-- once one is confidently identified, the information is thrown away.

Two passes instead. Pass 1 labels everything as usual. Any test file identified
as a known speaker with a large margin is then ADDED to that speaker's prototype
set, growing enrolment by up to 40% with in-domain, same-session material. Pass 2
re-scores everything against the enriched set, which should mostly help the weak
files -- the short ones that are 46% of our errors -- because they now have a
closer, same-recording-condition neighbour to match against.

This is not the transductive clustering that failed, and not the marginal
matching that half-failed. Nothing is assumed about the label distribution, and
no label is ever propagated between test files. A test file only ever gets added
to a speaker the model was already confident about.

The obvious risk is error propagation: a confident mistake becomes a permanent
bad prototype. That is what the confidence sweep below is for -- if the gain only
appears at reckless thresholds, it is not a real gain.

Evaluated on the REALISTIC folds (1-3 held-out files per speaker, true unknown
share swept), because that is the setting that exposed the last idea as an
artifact.
"""
import json, collections, sys
import numpy as np
from sklearn.metrics import f1_score

O2, O5, O9 = (sys.argv[1:4] + ["_out", "_out5", "_out9"])[:3]
COHORT, NFOLDS = 300, 15
MODELS, WEIGHTS, TAU = ["ecapa", "resnet", "eres2netv2"], [1, 2, 1], 0.25

meta = json.load(open(O2 + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]
KU = len(classes) - 1

l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
E = {"ecapa": l2(np.load(O2 + "/emb_file.npy").astype("float64")),
     "resnet": l2(np.load(O5 + "/emb_resnet.npy").astype("float64")),
     "eres2netv2": l2(np.load(O9 + "/emb_eres2netv2.npy").astype("float64"))}
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


def class_scores(vi, proto_idx, proto_lab):
    """Fused, AS-normed scores collapsed to one column per class.

    A test file that has been added to the prototype set must never score against
    itself -- cosine 1.0 would make it trivially 'correct' and the whole
    experiment meaningless. Those cells are masked out before the collapse.
    """
    proto_idx = np.asarray(proto_idx)
    S = 0.0
    for a, m in zip(WN, MODELS):
        S = S + a * asnorm(E[m][vi] @ E[m][proto_idx].T, E[m][proto_idx], E[m][vi])
    pos = {int(p): c for c, p in enumerate(proto_idx)}
    for r, i in enumerate(vi):
        c = pos.get(int(i))
        if c is not None:
            S[r, c] = -9e9
    C = np.full((len(vi), len(classes)), -9e9)
    pl = np.asarray(proto_lab)
    for ci, c in enumerate(classes):
        cols = np.flatnonzero(pl == c)
        if len(cols):
            C[:, ci] = S[:, cols].max(1)
    return C


def make_fold(f, true_share):
    rng = np.random.RandomState(5000 + f)
    val, trn = [], []
    for s in known:
        idx = list(by[s]); rng.shuffle(idx)
        nh = min(rng.randint(1, 4), max(len(idx) - 2, 0))
        if nh <= 0:
            trn += idx; continue
        val += [(i, s) for i in idx[:nh]]; trn += idx[nh:]
    n_known = len(val)
    n_unk = min(int(round(n_known * true_share / (1 - true_share))), len(unk) - 400)
    p = rng.permutation(len(unk))
    val += [(unk[j], "unknown") for j in p[:n_unk]]
    trn += list(unk[p[n_unk:]])
    return val, np.array([i for i, _ in val]), np.array(trn)


def guard(val, yp):
    for j, (i, _) in enumerate(val):
        if i in DEGEN:
            yp[j] = "unknown"
    return yp


def rule(C, tau=TAU):
    sk = C[:, :KU].max(1); su = C[:, KU]; b = C[:, :KU].argmax(1)
    return [classes[b[r]] if sk[r] - su[r] > tau else "unknown"
            for r in range(len(sk))], sk - su


def mild_sinkhorn(C, eps=1.0, iters=3, share=0.50):
    L = C / eps
    L[:, KU] += TAU / eps
    n, k = L.shape
    t = np.full(k, (1 - share) / (k - 1)); t[KU] = share
    logc = np.log(t * n)
    for _ in range(iters):
        L -= L.max(1, keepdims=True)
        L -= np.log(np.exp(L).sum(1, keepdims=True))
        L += logc[None, :] - np.log(np.exp(L).sum(0, keepdims=True) + 1e-30)
    return [classes[j] for j in L.argmax(1)]


macro = lambda val, yp: f1_score([s for _, s in val], yp, average="macro",
                                 labels=classes, zero_division=0)


def run(val, vi, trn, conf, use_sinkhorn):
    proto_idx = list(trn)
    proto_lab = [labels[j] for j in trn]
    C = class_scores(vi, np.array(proto_idx), proto_lab)
    yp, margin = rule(C)

    if conf is not None:
        add_i, add_l = [], []
        for r in range(len(vi)):
            if yp[r] != "unknown" and margin[r] > conf and vi[r] not in DEGEN:
                add_i.append(int(vi[r])); add_l.append(yp[r])
        if add_i:
            # class_scores masks the self-match, so pass 2 is a fair re-scoring
            proto_idx = np.array(list(trn) + add_i)
            proto_lab = proto_lab + add_l
            C = class_scores(vi, proto_idx, proto_lab)

    yp = mild_sinkhorn(C) if use_sinkhorn else rule(C)[0]
    return guard(val, list(yp))


print("Realistic folds. 'enrich' adds confidently-identified test files as extra")
print("prototypes, then re-scores. Confidence is the pass-1 margin.\n")
print("=" * 92)
hdr = "%-8s %-20s %-20s %s" % ("true", "baseline", "+sinkhorn", "+sinkhorn+enrich (by confidence)")
print(hdr)
print("=" * 92)

for share in [0.46, 0.50, 0.54]:
    folds = [make_fold(f, share) for f in range(NFOLDS)]
    b = np.array([macro(v, run(v, i, t, None, False)) for v, i, t in folds])
    s = np.array([macro(v, run(v, i, t, None, True)) for v, i, t in folds])
    line = "  %-6.0f%% %.4f +/-%.4f  %.4f (%+.4f)  " % (
        share * 100, b.mean(), b.std(ddof=1) / np.sqrt(len(b)), s.mean(), s.mean() - b.mean())
    for conf in [1.5, 1.0, 0.6]:
        e = np.array([macro(v, run(v, i, t, conf, True)) for v, i, t in folds])
        d = e - s
        sem = d.std(ddof=1) / np.sqrt(len(d))
        line += " c=%.1f %+.4f(t=%+.0f)" % (conf, d.mean(), d.mean() / sem if sem > 0 else 0)
    print(line, flush=True)

print()
print("enrich is only worth shipping if it is positive at every share AND at the")
print("conservative confidence levels -- a gain that needs a reckless threshold is")
print("error propagation waiting to happen on the real test set.")

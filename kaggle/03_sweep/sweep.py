"""
Runs entirely on the cached embeddings from notebook 02 -- no GPU, no Kaggle.

Two questions the k=446 result left open:
  1. the k sweep never plateaued -- how far does it actually go?
  2. is a centroid per speaker the right scorer, or is nearest-neighbour better?
"""
import json, collections, sys
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score

SEED = 0
OUT = sys.argv[1] if len(sys.argv) > 1 else "_out"

emb = np.load(OUT + "/emb_file.npy")
meta = json.load(open(OUT + "/meta.json"))
labels, valid = meta["labels"], np.array(meta["valid"])
known = sorted({l for l in labels if l != "unknown"})
classes = known + ["unknown"]

by_spk = collections.defaultdict(list)
for i, l in enumerate(labels):
    if valid[i]:
        by_spk[l].append(i)
unk_idx = np.array(by_spk["unknown"])
print("known speakers: %d   usable unknown files: %d" % (len(known), len(unk_idx)))


def l2(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def folds(n_folds=5):
    """Same protocol as notebook 02: hold out one file per known speaker plus an
    equal number of unknown files, so validation is ~50/50 like the real set."""
    for fold in range(n_folds):
        rng = np.random.RandomState(SEED + fold)
        val, trn_known = [], collections.defaultdict(list)
        for s in known:
            idx = by_spk[s]
            if len(idx) < 2:
                trn_known[s] = idx
                continue
            h = idx[fold % len(idx)]
            val.append((h, s))
            trn_known[s] = [j for j in idx if j != h]
        perm = rng.permutation(len(unk_idx))
        n_hold = min(len(val), len(unk_idx) // 2)
        val += [(unk_idx[j], "unknown") for j in perm[:n_hold]]
        yield val, trn_known, unk_idx[perm[n_hold:]]


def score(val, yp):
    return f1_score([s for _, s in val], yp, average="macro",
                    labels=classes, zero_division=0)


def run(k, known_mode="centroid", n_folds=5):
    out = []
    for val, trn_known, trn_unk in folds(n_folds):
        proto, plab = [], []
        for s in known:
            E = emb[trn_known[s]]
            if not len(E):
                continue
            if known_mode == "centroid":
                proto.append(E.mean(0)); plab.append(s)
            else:                                   # every training file is a prototype
                for e in E:
                    proto.append(e); plab.append(s)
        if k >= len(trn_unk):                       # degenerate case = 1-NN over unknown
            for j in trn_unk:
                proto.append(emb[j]); plab.append("unknown")
        elif k <= 1:
            proto.append(emb[trn_unk].mean(0)); plab.append("unknown")
        else:
            cl = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                         linkage="average").fit_predict(emb[trn_unk])
            for c in range(k):
                m = trn_unk[cl == c]
                if len(m):
                    proto.append(emb[m].mean(0)); plab.append("unknown")
        C = l2(np.stack(proto))
        vi = np.array([i for i, _ in val])
        yp = [plab[j] for j in (l2(emb[vi]) @ C.T).argmax(1)]
        out.append(score(val, yp))
    return float(np.mean(out)), float(np.std(out))


print("\n=== k sweep, known speakers scored by centroid ===")
best = (0, None)
res = {}
for k in [446, 553, 700, 900, 1200, 1600, 10 ** 9]:
    m, s = run(k)
    tag = "1-NN (every unknown file)" if k > 10 ** 6 else str(k)
    res[tag] = m
    print("  k=%-26s macro-F1 = %.4f  (+/- %.4f)" % (tag, m, s))
    if m > best[0]:
        best = (m, tag)
print("  -> best: k=%s at %.4f" % (best[1], best[0]))

print("\n=== does 1-NN beat centroids for the KNOWN speakers too? ===")
for mode in ["centroid", "allfiles"]:
    m, s = run(553, known_mode=mode)
    print("  known=%-9s k=553  macro-F1 = %.4f  (+/- %.4f)" % (mode, m, s))

json.dump(res, open(OUT + "/sweep_scores.json", "w"), indent=2)

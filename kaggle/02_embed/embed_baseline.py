"""
IAAA 3rd Contest - Speaker Identification
Notebook 02: ECAPA embedding extraction + centroid baseline

Facts established locally (no need to re-derive):
  - audio files are WAV (PCM_16, 16 kHz, stereo with IDENTICAL channels)
    despite the .mp3 extension  -> read with soundfile, take channel 0
  - 4529 files, 446 known speakers + "unknown" (2275 files, 50.2%)
  - ~140 files are corrupt/near-empty (smallest is 48 bytes = header only)
  - metric is macro-F1 over 447 classes

Outputs to /kaggle/working/:
  emb_file.npy     (N, 192)  L2-normalised mean embedding per file
  emb_chunk.npy    (M, 192)  L2-normalised per-chunk embeddings
  chunk_owner.npy  (M,)      index into files[] for each chunk
  meta.json                  file order, labels, durations, skip reasons

Run on Kaggle with GPU + Internet.
"""
import os, sys, csv, json, time, random, collections, subprocess, traceback
from pathlib import Path

import numpy as np


def ensure_speechbrain():
    """Kaggle images do not ship speechbrain. Installing it the normal way drags
    in a fresh torch wheel that has no kernels for this GPU -> every forward pass
    dies with cudaErrorNoKernelImageForDevice. So: pure-python deps normally,
    speechbrain itself with --no-deps, leaving Kaggle's CUDA-matched torch alone."""
    import torch
    before = torch.__version__
    try:
        import speechbrain  # noqa: F401
    except ImportError:
        print("installing speechbrain (--no-deps, preserving torch %s) ..." % before, flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hyperpyyaml"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "speechbrain"], check=True)
    # never reload torch -- re-running its module init re-registers TORCH_LIBRARY
    # and blows up. Compare the on-disk distribution against the loaded module instead.
    from importlib.metadata import version
    on_disk = version("torch")
    if not before.startswith(on_disk):
        print("!! torch on disk (%s) no longer matches the loaded module (%s); "
              "the install clobbered it" % (on_disk, before))
    else:
        print("torch intact:", before)


ALLOW_CPU = False   # 73h of audio on CPU takes most of a session -- not worth the quota


def pick_device():
    """Never trust torch.cuda.is_available() alone -- actually run a kernel.

    Kaggle's P100 is Pascal (sm_60) and the torch cu128 wheels no longer ship
    Pascal kernels, so on a P100 the GPU is visible but every kernel launch
    fails with cudaErrorNoKernelImageForDevice. Run this notebook on T4 (sm_75).
    """
    import torch
    reason = None
    if not torch.cuda.is_available():
        reason = "no CUDA device visible"
    else:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print("gpu: %s  compute capability sm_%d%d  torch %s"
              % (name, cap[0], cap[1], torch.__version__))
        try:
            a = torch.randn(64, 64, device="cuda")
            torch.mm(a, a).sum().item()
            print("GPU kernels OK")
            return "cuda"
        except Exception as ex:
            reason = "GPU present but no usable kernels (%s): %s" % (name, str(ex).splitlines()[0])

    msg = ("%s.\n   This run would fall back to CPU, which takes hours for 73h of "
           "audio.\n   Switch the accelerator to T4 (sm_75) -- P100 is sm_60 and the "
           "torch cu128\n   wheels dropped Pascal support." % reason)
    if not ALLOW_CPU:
        raise RuntimeError(msg)
    print("!! " + msg + "\n   ALLOW_CPU is set, continuing on CPU anyway.")
    return "cpu"

SR          = 16000
CHUNK_S     = 6.0            # seconds per chunk fed to ECAPA
MAX_CHUNKS  = 10             # cap per file -> bounds compute at ~60s/file
MIN_SPEECH  = 1.0            # files with less than this much speech are skipped
OUT         = Path("/kaggle/working")
SEED        = 0

random.seed(SEED)
np.random.seed(SEED)


def hr(t=""):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72, flush=True)


# --------------------------------------------------------------------------
# locate the data
# --------------------------------------------------------------------------
def find_data():
    root = Path("/kaggle/input")
    labels = list(root.rglob("labels.csv"))
    if not labels:
        raise SystemExit("labels.csv not found under /kaggle/input")
    lab_path = labels[0]
    audio_dir = lab_path.parent
    print("labels.csv :", lab_path)
    print("audio dir  :", audio_dir)
    return lab_path, audio_dir


# --------------------------------------------------------------------------
# io: these are WAV files wearing an .mp3 extension
# --------------------------------------------------------------------------
def read_mono(path, max_s=90.0):
    """Read channel 0 as float32. Handles the real (WAV) case fast, and
    falls back to librosa in case the eval set really is mp3."""
    import soundfile as sf
    try:
        with sf.SoundFile(path) as f:
            n = min(len(f), int(max_s * f.samplerate))
            y = f.read(frames=n, dtype="float32", always_2d=True)
            sr = f.samplerate
        y = y[:, 0]
    except Exception:
        import librosa
        y, sr = librosa.load(path, sr=None, mono=True, duration=max_s)
        y = np.asarray(y, dtype="float32")
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def speech_only(y, sr=SR, frame=400, hop=160, top_db=35.0):
    """Cheap energy VAD: keep frames within top_db of the file's loudest frame.
    Median silence in this corpus is only ~3%, so this is about dropping the
    pathological files, not squeezing out every pause."""
    if len(y) < frame:
        return y
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt((y[idx] ** 2).mean(axis=1) + 1e-12)
    keep = rms > (rms.max() * (10 ** (-top_db / 20.0)))
    if not keep.any():
        return y
    mask = np.zeros(len(y), dtype=bool)
    for s in np.flatnonzero(keep):
        mask[s * hop: s * hop + frame] = True
    return y[mask]


def to_chunks(y, sr=SR):
    """Fixed-length chunks. Short files are tiled rather than zero-padded so
    ECAPA never sees a block of digital silence."""
    L = int(CHUNK_S * sr)
    if len(y) < L:
        reps = int(np.ceil(L / max(len(y), 1)))
        y = np.tile(y, reps)[:L]
        return np.stack([y])
    n = min(MAX_CHUNKS, len(y) // L)
    if n == 0:
        return np.stack([y[:L]])
    # spread the chunks across the whole file rather than taking a prefix
    starts = np.linspace(0, len(y) - L, n).astype(int)
    return np.stack([y[s: s + L] for s in starts])


# --------------------------------------------------------------------------
# embedding extraction
# --------------------------------------------------------------------------
def extract(files, audio_dir):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    device = pick_device()
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(OUT / "ecapa"),
        run_opts={"device": device},
    )
    enc.eval()

    # prove the model itself runs before committing to 4529 files
    with torch.no_grad():
        probe = enc.encode_batch(torch.zeros(1, SR, device=device) + 1e-3)
    print("model smoke test OK, embedding dim =", probe.shape[-1], flush=True)

    emb_file, emb_chunk, owner = [], [], []
    durations, skipped = [], {}
    t0, n_err = time.time(), 0

    for i, fn in enumerate(files):
        dur = 0.0
        try:
            y = read_mono(audio_dir / fn)
            dur = len(y) / SR
            s = speech_only(y)
            if len(s) < MIN_SPEECH * SR:
                skipped[fn] = "no_speech(%.2fs)" % (len(s) / SR)
                emb_file.append(np.zeros(192, dtype="float32"))
            else:
                ch = to_chunks(s)
                t = torch.from_numpy(ch).to(device)
                with torch.no_grad():
                    e = enc.encode_batch(t).squeeze(1).cpu().numpy()
                e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
                m = e.mean(0)
                m = m / (np.linalg.norm(m) + 1e-9)
                emb_file.append(m.astype("float32"))
                emb_chunk.append(e.astype("float32"))
                owner.extend([i] * len(e))
        except Exception as ex:
            skipped[fn] = "error: %s" % str(ex).splitlines()[0]
            emb_file.append(np.zeros(192, dtype="float32"))
            n_err += 1
            if n_err <= 3:
                print("  error on %s: %s" % (fn, str(ex).splitlines()[0]), flush=True)
            if n_err == 25:
                raise RuntimeError(
                    "25 files failed to embed -- aborting rather than burning "
                    "the rest of the run. First reason: %s" % skipped[fn])
        durations.append(dur)

        if (i + 1) % 250 == 0:
            el = time.time() - t0
            print("  %5d/%d  %.1fs elapsed  eta %.1f min"
                  % (i + 1, len(files), el, el / (i + 1) * (len(files) - i - 1) / 60),
                  flush=True)

    return (np.stack(emb_file),
            np.concatenate(emb_chunk) if emb_chunk else np.zeros((0, 192), "float32"),
            np.array(owner, dtype="int32"),
            durations, skipped)


# --------------------------------------------------------------------------
# centroid baseline, scored the way the contest scores it
# --------------------------------------------------------------------------
def l2(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def cluster_unknown(E, k):
    """Split the heterogeneous 'unknown' bucket into k pseudo-identities."""
    from sklearn.cluster import AgglomerativeClustering
    try:
        ac = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    except TypeError:                                    # sklearn < 1.4
        ac = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
    return ac.fit_predict(E)


def evaluate(emb, labels, valid, n_folds=5, k_list=(1, 50, 200, 446)):
    """Leave-one-file-out style CV that mimics the real eval composition:
    ~50% unknown files, known speakers seen in training but with held-out files."""
    from sklearn.metrics import f1_score

    known = sorted({l for l in labels if l != "unknown"})
    classes = known + ["unknown"]

    by_spk = collections.defaultdict(list)
    for i, l in enumerate(labels):
        if valid[i]:
            by_spk[l].append(i)
    unk_idx = np.array(by_spk["unknown"])

    results = {}
    for k in k_list:
        scores = []
        for fold in range(n_folds):
            rng = np.random.RandomState(SEED + fold)

            # hold out one file per known speaker
            val, trn_known = [], collections.defaultdict(list)
            for s in known:
                idx = by_spk[s]
                if len(idx) < 2:
                    trn_known[s] = idx
                    continue
                h = idx[fold % len(idx)]
                val.append((h, s))
                trn_known[s] = [j for j in idx if j != h]

            # hold out the same number of unknown files -> 50/50, like the real set
            perm = rng.permutation(len(unk_idx))
            n_hold = min(len(val), len(unk_idx) // 2)
            val += [(unk_idx[j], "unknown") for j in perm[:n_hold]]
            trn_unk = unk_idx[perm[n_hold:]]

            # known centroids
            cents, cent_lab = [], []
            for s in known:
                if trn_known[s]:
                    cents.append(emb[trn_known[s]].mean(0))
                    cent_lab.append(s)

            # unknown centroids
            if k <= 1:
                cents.append(emb[trn_unk].mean(0))
                cent_lab.append("unknown")
            else:
                kk = min(k, len(trn_unk))
                cl = cluster_unknown(emb[trn_unk], kk)
                for c in range(kk):
                    m = trn_unk[cl == c]
                    if len(m):
                        cents.append(emb[m].mean(0))
                        cent_lab.append("unknown")

            C = l2(np.stack(cents))
            vi = np.array([i for i, _ in val])
            yt = [s for _, s in val]
            yp = [cent_lab[j] for j in (l2(emb[vi]) @ C.T).argmax(1)]
            scores.append(f1_score(yt, yp, average="macro", labels=classes, zero_division=0))

        results[k] = (float(np.mean(scores)), float(np.std(scores)))
        print("  unknown clusters k=%-4d  macro-F1 = %.4f  (+/- %.4f)"
              % (k, results[k][0], results[k][1]), flush=True)
    return results


# --------------------------------------------------------------------------
def main():
    hr("0. SETUP")
    ensure_speechbrain()
    import torch, speechbrain, sklearn, soundfile
    print("python %s | torch %s | speechbrain %s | sklearn %s | soundfile %s"
          % (sys.version.split()[0], torch.__version__, speechbrain.__version__,
             sklearn.__version__, soundfile.__version__))
    print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

    lab_path, audio_dir = find_data()
    rows = list(csv.DictReader(open(lab_path, encoding="utf-8")))
    files = [r["audio_file"] for r in rows]
    labels = [r["speaker_id"] for r in rows]
    print("files: %d   known speakers: %d   unknown files: %d"
          % (len(files), len({l for l in labels if l != 'unknown'}),
             sum(l == 'unknown' for l in labels)))

    hr("1. EMBEDDING EXTRACTION")
    emb, chunks, owner, durations, skipped = extract(files, audio_dir)
    print("\nfile embeddings :", emb.shape)
    print("chunk embeddings:", chunks.shape)
    print("skipped files   :", len(skipped))
    for fn, why in list(skipped.items())[:10]:
        print("   ", fn, "->", why)

    valid = np.array([np.abs(e).sum() > 0 for e in emb])
    print("usable files    : %d / %d" % (valid.sum(), len(files)))
    d = np.array(durations)
    print("duration  med %.1fs  min %.1fs  max %.1fs" % (np.median(d), d.min(), d.max()))

    OUT.mkdir(exist_ok=True, parents=True)
    np.save(OUT / "emb_file.npy", emb)
    np.save(OUT / "emb_chunk.npy", chunks)
    np.save(OUT / "chunk_owner.npy", owner)
    json.dump({"files": files, "labels": labels, "durations": durations,
               "valid": valid.tolist(), "skipped": skipped,
               "chunk_s": CHUNK_S, "max_chunks": MAX_CHUNKS},
              open(OUT / "meta.json", "w"))
    print("saved to", OUT)

    hr("2. SANITY: does the embedding space separate speakers at all?")
    kn = [i for i in range(len(files)) if valid[i] and labels[i] != "unknown"]
    by = collections.defaultdict(list)
    for i in kn:
        by[labels[i]].append(i)
    same, diff = [], []
    rng = np.random.RandomState(SEED)
    for s, idx in by.items():
        if len(idx) >= 2:
            a, b = rng.choice(idx, 2, replace=False)
            same.append(float(emb[a] @ emb[b]))
    for _ in range(3000):
        a, b = rng.choice(kn, 2, replace=False)
        if labels[a] != labels[b]:
            diff.append(float(emb[a] @ emb[b]))
    print("  cosine, SAME speaker      : %.4f" % np.mean(same))
    print("  cosine, DIFFERENT speaker : %.4f" % np.mean(diff))
    print("  separation                : %.4f" % (np.mean(same) - np.mean(diff)))
    print("  (a healthy gap here means the pretrained model transfers; a tiny")
    print("   gap means this domain needs fine-tuning before anything else)")

    hr("3. CENTROID BASELINE (macro-F1, 5 folds)")
    print("  k=1 is the naive 'one blob for unknown' approach.")
    print("  k>1 splits the unknown bucket into pseudo-identities.\n")
    res = evaluate(emb, labels, valid)
    best = max(res, key=lambda k: res[k][0])
    print("\n  best k = %d  ->  macro-F1 %.4f" % (best, res[best][0]))
    print("  gain from clustering: %+.4f" % (res[best][0] - res[1][0]))
    json.dump({str(k): v for k, v in res.items()}, open(OUT / "baseline_scores.json", "w"))

    hr("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

"""
IAAA 3rd Contest - Speaker Identification
Notebook 19: 3-second chunks instead of 6.

CHUNK_S = 6 and MAX_CHUNKS = 10 were picked on day one and never questioned. They
are the last untested axis, and unlike everything tried since 0.9660 they sit on
the representation side rather than the decision side, so they are not obviously
on the same plateau.

Why 3 s might matter here. ECAPA's statistics pooling is length-sensitive, and
46% of our errors are on files under 10 seconds. A 4-second file currently yields
ONE chunk, tiled to fill 6 s -- the model sees the same audio twice and the
"average over chunks" averages nothing. At 3 s the same file yields a real
chunk plus a partial one, and a 60 s file yields 20 independent views instead of
10, halving the variance of the mean.

Total coverage is held constant: 20 x 3 s = 60 s, same as 10 x 6 s. Only the
granularity changes.

Also measures how the score scale shifts. tau = 0.32 was tuned against the
leaderboard for the 6 s embeddings; if 3 s chunks move the margin distribution,
that number has to move with it or the comparison is confounded.
"""
import os, sys, csv, json, time, collections, subprocess, traceback
from pathlib import Path

import numpy as np

SR = 16000
CHUNK_S, MAX_CHUNKS = 3.0, 20          # 6.0 / 10 in the shipped pipeline
MIN_SPEECH = 1.0
OUT = Path("/kaggle/working")


def hr(t=""):
    print("\n" + "=" * 72); print(t); print("=" * 72, flush=True)


def ensure_speechbrain():
    import torch
    before = torch.__version__
    try:
        import speechbrain  # noqa: F401
    except ImportError:
        print("installing speechbrain (--no-deps, preserving torch %s)" % before, flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hyperpyyaml"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "speechbrain"], check=True)
    from importlib.metadata import version
    if not before.startswith(version("torch")):
        raise RuntimeError("torch clobbered: %s -> %s" % (before, version("torch")))
    print("torch intact:", before)


def pick_device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    cap = torch.cuda.get_device_capability(0)
    print("gpu: %s sm_%d%d" % (torch.cuda.get_device_name(0), cap[0], cap[1]))
    a = torch.randn(64, 64, device="cuda"); torch.mm(a, a).sum().item()
    return "cuda:0"


def read_mono(path, max_s=90.0):
    import soundfile as sf
    try:
        with sf.SoundFile(str(path)) as f:
            n = min(len(f), int(max_s * f.samplerate))
            y = f.read(frames=n, dtype="float32", always_2d=True)[:, 0]
            sr = f.samplerate
    except Exception:
        import librosa
        y, sr = librosa.load(str(path), sr=None, mono=True, duration=max_s)
        y = np.asarray(y, dtype="float32")
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def speech_only(y, frame=400, hop=160, top_db=35.0):
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


def to_chunks(y):
    L = int(CHUNK_S * SR)
    if len(y) < L:
        reps = int(np.ceil(L / max(len(y), 1)))
        return np.stack([np.tile(y, reps)[:L]])
    n = min(MAX_CHUNKS, len(y) // L)
    if n == 0:
        return np.stack([y[:L]])
    starts = np.linspace(0, len(y) - L, n).astype(int)
    return np.stack([y[s: s + L] for s in starts])


def main():
    hr("0. SETUP")
    ensure_speechbrain()
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    device = pick_device()
    print("chunk config: %.0f s x up to %d  (shipped: 6 s x 10)" % (CHUNK_S, MAX_CHUNKS))

    lab = list(Path("/kaggle/input").rglob("labels.csv"))[0]
    audio_dir = lab.parent
    rows = list(csv.DictReader(open(lab, encoding="utf-8")))
    files = [r["audio_file"] for r in rows]
    labels = [r["speaker_id"] for r in rows]
    print("files:", len(files))

    encoders, dims = {}, {"ecapa": 192, "resnet": 256}
    for name, src in [("ecapa", "speechbrain/spkrec-ecapa-voxceleb"),
                      ("resnet", "speechbrain/spkrec-resnet-voxceleb")]:
        e = EncoderClassifier.from_hparams(source=src, savedir=str(OUT / name),
                                           run_opts={"device": device})
        e.eval()
        encoders[name] = e

    hr("1. EXTRACT")
    emb = {k: np.zeros((len(files), d), "float32") for k, d in dims.items()}
    n_chunks = np.zeros(len(files), dtype=int)
    skipped, t0, n_err = {}, time.time(), 0
    for i, fn in enumerate(files):
        try:
            s = speech_only(read_mono(audio_dir / fn))
            if len(s) < MIN_SPEECH * SR:
                skipped[fn] = "no_speech"
                continue
            ch = to_chunks(s)
            n_chunks[i] = len(ch)
            t = torch.from_numpy(ch).to(device)
            for k, enc in encoders.items():
                with torch.no_grad():
                    v = enc.encode_batch(t).squeeze(1).cpu().numpy()
                v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
                m = v.mean(0)
                emb[k][i] = (m / (np.linalg.norm(m) + 1e-9)).astype("float32")
        except Exception as ex:
            skipped[fn] = "error: %s" % str(ex).splitlines()[0]
            n_err += 1
            if n_err <= 3:
                print("   error %s: %s" % (fn, str(ex).splitlines()[0]), flush=True)
            if n_err == 25:
                raise RuntimeError("25 failures, aborting")
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print("   %5d/%d  %.0fs  eta %.1f min"
                  % (i + 1, len(files), el, el / (i + 1) * (len(files) - i - 1) / 60), flush=True)

    for k in emb:
        np.save(OUT / ("emb_c3_%s.npy" % k), emb[k])
    json.dump({"chunk_s": CHUNK_S, "max_chunks": MAX_CHUNKS,
               "n_chunks": n_chunks.tolist(), "skipped": skipped},
              open(OUT / "meta_c3.json", "w"))
    ok = n_chunks > 0
    print("\nsaved. skipped %d" % len(skipped))
    print("chunks per file: median %d  (6 s pipeline gave a median of 9)"
          % int(np.median(n_chunks[ok])))
    # the files this change is actually aimed at: too short for a 6 s chunk,
    # so the old pipeline tiled them and averaged one view with itself
    short = ok & (n_chunks >= 2) & (n_chunks * CHUNK_S < 12)
    print("files with 2-3 chunks now (these were single tiled chunks at 6 s): %d"
          % int(short.sum()))

    hr("2. SEPARATION")
    byk = collections.defaultdict(list)
    for i in np.flatnonzero(ok):
        if labels[i] != "unknown":
            byk[labels[i]].append(i)
    rng = np.random.RandomState(0)
    for tag in ["ecapa", "resnet"]:
        A = emb[tag] / (np.linalg.norm(emb[tag], axis=1, keepdims=True) + 1e-9)
        same = [float(A[a] @ A[b]) for s, ii in byk.items() if len(ii) >= 2
                for a, b in [rng.choice(ii, 2, replace=False)]]
        diff = []
        idx = np.flatnonzero(ok)
        while len(diff) < 2000:
            a, b = rng.choice(idx, 2, replace=False)
            if labels[a] != labels[b]:
                diff.append(float(A[a] @ A[b]))
        print("  %-8s same %.4f  diff %.4f  separation %.4f"
              % (tag, np.mean(same), np.mean(diff), np.mean(same) - np.mean(diff)))
    print("\n  6 s pipeline: ecapa 0.6036   resnet 0.6495")
    print("  separation has misled me twice; the verdict comes from the holdout")
    print("  protocol and ultimately from the leaderboard.")

    hr("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

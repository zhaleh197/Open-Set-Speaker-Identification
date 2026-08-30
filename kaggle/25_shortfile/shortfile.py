"""
IAAA 3rd Contest - Speaker Identification
Notebook 25: how short files are padded, which is where all the error lives.

The diagnostic that led here. Per-speaker separability on the training data is
d' ~ 10.3. ECAPA on VoxCeleb1-O manages d' ~ 4.8. For well-formed files this task
is already solved -- the remaining 3% of error is not speakers being confusable,
it is concentrated in damaged recordings. Every finding lines up with that: 46% of
errors on files under 10 s, 1.0% error on well-covered files against 3.3% overall,
and no encoder swap, calibration, fine-tune or front-end change moving anything,
because none of them touch that subset.

So look at what the code actually does to those files:

    if len(y) < L:
        reps = ceil(L / len(y))
        return tile(y, reps)[:L]

A 2-second clip is repeated three times to fill a 6-second chunk. That fabricates
a signal with an artificial 2-second period and a discontinuity at every seam --
a broadband click smeared across the spectrum, fed to a model that was never
trained on anything like it. Written on day one, never questioned, and applied to
exactly the files that carry all the error.

Four policies, everything else held fixed:

    tile     what ships today
    pad      zero-pad to the chunk length
    reflect  mirror the clip at the boundary: no seam discontinuity
    raw      hand ECAPA the short clip as-is; it accepts variable length

Long files are processed identically under all four, so their embeddings must
come out bit-identical -- that is the built-in check that nothing else drifted.
"""
import os, sys, csv, json, time, collections, subprocess, traceback
from pathlib import Path

import numpy as np

SR, CHUNK_S, MAX_CHUNKS, MIN_SPEECH = 16000, 6.0, 10, 1.0
POLICIES = ["tile", "pad", "reflect", "raw"]
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
        raise RuntimeError("torch clobbered")
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


def to_chunks(y, policy):
    L = int(CHUNK_S * SR)
    if len(y) >= L:
        n = min(MAX_CHUNKS, len(y) // L)
        starts = np.linspace(0, len(y) - L, n).astype(int)
        return [y[s: s + L] for s in starts]
    if policy == "tile":
        reps = int(np.ceil(L / max(len(y), 1)))
        return [np.tile(y, reps)[:L]]
    if policy == "pad":
        return [np.pad(y, (0, L - len(y)))]
    if policy == "reflect":
        # mirror at the boundary: fills the chunk with no seam discontinuity
        out = y
        while len(out) < L:
            out = np.concatenate([out, out[::-1]])
        return [out[:L]]
    if policy == "raw":
        return [y]                      # ECAPA accepts variable length
    raise ValueError(policy)


def main():
    hr("0. SETUP")
    ensure_speechbrain()
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    device = pick_device()

    lab = list(Path("/kaggle/input").rglob("labels.csv"))[0]
    audio_dir = lab.parent
    rows = list(csv.DictReader(open(lab, encoding="utf-8")))
    files = [r["audio_file"] for r in rows]

    enc = {}
    for name, src in [("ecapa", "speechbrain/spkrec-ecapa-voxceleb"),
                      ("resnet", "speechbrain/spkrec-resnet-voxceleb")]:
        e = EncoderClassifier.from_hparams(source=src, savedir=str(OUT / name),
                                           run_opts={"device": device})
        e.eval()
        enc[name] = e
    dims = {"ecapa": 192, "resnet": 256}

    hr("1. EXTRACT UNDER EACH SHORT-FILE POLICY")
    emb = {p: {k: np.zeros((len(files), d), "float32") for k, d in dims.items()}
           for p in POLICIES}
    is_short = np.zeros(len(files), bool)
    speech_s = np.zeros(len(files))
    skipped, t0, n_err = {}, time.time(), 0

    for i, fn in enumerate(files):
        try:
            s = speech_only(read_mono(audio_dir / fn))
            speech_s[i] = len(s) / SR
            if len(s) < MIN_SPEECH * SR:
                skipped[fn] = "no_speech"
                continue
            short = len(s) < int(CHUNK_S * SR)
            is_short[i] = short
            for p in POLICIES:
                if not short and p != POLICIES[0]:
                    # long files are policy-independent: reuse, and the assertion
                    # below confirms that assumption rather than trusting it
                    for k in dims:
                        emb[p][k][i] = emb[POLICIES[0]][k][i]
                    continue
                ch = to_chunks(s, p)
                t = torch.from_numpy(np.stack(ch) if len({len(c) for c in ch}) == 1
                                     else ch[0][None, :]).to(device)
                for k, e in enc.items():
                    with torch.no_grad():
                        v = e.encode_batch(t).squeeze(1).cpu().numpy()
                    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
                    m = v.mean(0)
                    emb[p][k][i] = (m / (np.linalg.norm(m) + 1e-9)).astype("float32")
        except Exception as ex:
            skipped[fn] = "error: %s" % str(ex).splitlines()[0]
            n_err += 1
            if n_err <= 3:
                print("   error %s: %s" % (fn, str(ex).splitlines()[0]), flush=True)
            if n_err == 25:
                raise RuntimeError("25 failures, aborting")
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print("   %5d/%d  %.0fs  eta %.1f min  (short so far: %d)"
                  % (i + 1, len(files), el,
                     el / (i + 1) * (len(files) - i - 1) / 60, int(is_short[:i + 1].sum())),
                  flush=True)

    for p in POLICIES:
        for k in dims:
            np.save(OUT / ("emb_%s_%s.npy" % (p, k)), emb[p][k])
    json.dump({"is_short": is_short.tolist(), "speech_seconds": speech_s.tolist(),
               "skipped": skipped, "policies": POLICIES},
              open(OUT / "meta_short.json", "w"))

    hr("2. WHAT CHANGED")
    print("short files (under %.0f s of speech): %d of %d  (%.1f%%)"
          % (CHUNK_S, int(is_short.sum()), len(files), is_short.sum() / len(files) * 100))
    print("their speech duration: median %.2fs  min %.2fs"
          % (np.median(speech_s[is_short]), speech_s[is_short].min()))

    long_ok = (~is_short) & (speech_s > 0)
    same = np.allclose(emb["tile"]["ecapa"][long_ok], emb["pad"]["ecapa"][long_ok])
    print("\nlong files identical across policies: %s  (sanity check)" % same)

    print("\nhow far each policy moves a short file's embedding, vs tile:")
    for p in POLICIES[1:]:
        a = emb["tile"]["ecapa"][is_short]
        b = emb[p]["ecapa"][is_short]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        print("   %-8s cosine to tile: median %.4f   p10 %.4f   min %.4f"
              % (p, np.median(cos), np.percentile(cos, 10), cos.min()))
    print("\n   a cosine well below 1.0 means tiling was producing a materially")
    print("   different vector -- the question is which one is right, answered")
    print("   by the holdout protocol locally.")

    hr("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

"""
IAAA 3rd Contest - Speaker Identification
Notebook 17: fix the front-end instead of the decision rule.

Everything tried since the 0.9660 submission has been a different way of reading
the same scores: margin thresholds, AS-norm, QMF calibration, Sinkhorn, per-class
bias, starved-class rescue, query aggregation. The diagnostic that killed the
last one explains why none of them moved much -- when a speaker receives no
predictions, its own test files do not score for it. The scores are wrong, so no
rule applied to the scores can be right.

The one component never touched is the front-end. Our VAD is a single line:

    keep = rms > rms.max() * 10**(-35/20)

The threshold is RELATIVE TO THE FILE'S OWN LOUDEST FRAME. On a noisy recording
the loudest frame is noise, so everything passes and the encoder is handed noise.
That is exactly how the 34 files with collapsed embeddings were produced -- and I
papered over them with a guard rather than fixing the cause.

It also lines up with the error structure: 46% of errors are on files under 10
seconds. For a 4-second file, whether the VAD isolates 2 seconds of speech or
keeps 2 seconds of noise is the difference between a usable embedding and a
meaningless one.

This run re-extracts embeddings with a real neural VAD and compares, on identical
files, chunking and encoders. Only the speech selection changes.
"""
import os, sys, csv, json, time, collections, subprocess, traceback
from pathlib import Path

import numpy as np

SR, CHUNK_S, MAX_CHUNKS, MIN_SPEECH = 16000, 6.0, 10, 1.0
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


def energy_vad(y, frame=400, hop=160, top_db=35.0):
    """The current rule, kept so the comparison is like-for-like."""
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
    device = pick_device()
    from speechbrain.inference.VAD import VAD
    from speechbrain.inference.speaker import EncoderClassifier

    lab = list(Path("/kaggle/input").rglob("labels.csv"))[0]
    audio_dir = lab.parent
    rows = list(csv.DictReader(open(lab, encoding="utf-8")))
    files = [r["audio_file"] for r in rows]
    print("files:", len(files))

    hr("1. NEURAL VAD")
    vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty",
                           savedir=str(OUT / "vad"), run_opts={"device": device})
    print("VAD loaded", flush=True)

    MIN_VAD_S = 3.0

    def neural_speech(y):
        """Return only the frames the VAD calls speech. Falls back to the energy
        rule when the VAD finds nothing or cannot run, so a file is never lost.

        The CRDNN needs a few frames of context: handed a very short clip it
        produces a 1-frame feature map and its convolution padding throws. Short
        files are exactly the ones this whole experiment is for, so they are
        zero-padded up to MIN_VAD_S and the mask is cut back afterwards."""
        import torch
        n0 = len(y)
        yy = y
        if n0 < int(MIN_VAD_S * SR):
            yy = np.pad(y, (0, int(MIN_VAD_S * SR) - n0))
        t = torch.from_numpy(yy).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                prob = vad.get_speech_prob_chunk(t).cpu().squeeze().numpy()
        except Exception:
            return energy_vad(y), -1.0
        prob = np.atleast_1d(prob)
        # map the padded-length mask back onto the original samples
        hop_pad = max(1, int(round(len(yy) / len(prob))))
        n_keep = int(np.ceil(n0 / hop_pad))
        prob = prob[:max(n_keep, 1)]
        y = y[:n0]
        if prob.size == 0:
            return energy_vad(y), 0.0
        keep = prob > 0.5
        if not keep.any():
            return energy_vad(y), float(prob.mean())
        mask = np.zeros(len(y), dtype=bool)
        for s in np.flatnonzero(keep):
            mask[s * hop_pad: min((s + 1) * hop_pad, len(y))] = True
        out = y[mask]
        return (out if len(out) >= MIN_SPEECH * SR else energy_vad(y)), float(prob.mean())

    hr("2. RE-EXTRACT WITH NEURAL VAD")
    encoders = {}
    for name, src in [("ecapa", "speechbrain/spkrec-ecapa-voxceleb"),
                      ("resnet", "speechbrain/spkrec-resnet-voxceleb")]:
        e = EncoderClassifier.from_hparams(source=src, savedir=str(OUT / name),
                                           run_opts={"device": device})
        e.eval()
        encoders[name] = e
    dims = {"ecapa": 192, "resnet": 256}

    emb = {k: np.zeros((len(files), d), "float32") for k, d in dims.items()}
    speech_s, prob_m, skipped = np.zeros(len(files)), np.zeros(len(files)), {}
    t0, n_err = time.time(), 0
    for i, fn in enumerate(files):
        try:
            y = read_mono(audio_dir / fn)
            s, pm = neural_speech(y)
            speech_s[i] = len(s) / SR
            prob_m[i] = pm
            if len(s) < MIN_SPEECH * SR:
                skipped[fn] = "no_speech(%.2fs)" % (len(s) / SR)
                continue
            ch = torch.from_numpy(to_chunks(s)).to(device)
            for k, enc in encoders.items():
                with torch.no_grad():
                    v = enc.encode_batch(ch).squeeze(1).cpu().numpy()
                v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
                m = v.mean(0)
                emb[k][i] = (m / (np.linalg.norm(m) + 1e-9)).astype("float32")
        except Exception as ex:
            skipped[fn] = "error: %s" % str(ex).splitlines()[0]
            n_err += 1
            if n_err <= 3:
                print("   error %s: %s" % (fn, str(ex).splitlines()[0]), flush=True)
            if n_err == 200:
                raise RuntimeError("200 failures with a fallback in place -- something systematic")
        if (i + 1) % 400 == 0:
            el = time.time() - t0
            print("   %5d/%d  %.0fs  eta %.1f min"
                  % (i + 1, len(files), el, el / (i + 1) * (len(files) - i - 1) / 60), flush=True)

    for k in emb:
        np.save(OUT / ("emb_vad_%s.npy" % k), emb[k])
    json.dump({"speech_seconds": speech_s.tolist(), "vad_prob": prob_m.tolist(),
               "skipped": skipped},
              open(OUT / "meta_vad.json", "w"))
    print("\nsaved. skipped %d files" % len(skipped))

    hr("3. WHAT THE VAD CHANGED")
    old = json.load(open(list(Path("/kaggle/input").rglob("meta.json"))[0]))
    od = np.array(old["durations"])
    print("  raw audio        : median %.1fs" % np.median(od))
    print("  speech kept (VAD): median %.1fs" % np.median(speech_s[speech_s > 0]))
    print("  files where the VAD kept under half the audio: %d"
          % int((speech_s < 0.5 * od).sum()))
    print("  files the VAD calls almost pure non-speech (mean prob < 0.2): %d"
          % int((prob_m < 0.2).sum()))

    hr("4. SEPARATION, OLD FRONT-END vs NEW")
    labels = [r["speaker_id"] for r in rows]
    ok_new = np.flatnonzero(np.abs(emb["ecapa"]).sum(1) > 0)
    byk = collections.defaultdict(list)
    for i in ok_new:
        if labels[i] != "unknown":
            byk[labels[i]].append(i)
    rng = np.random.RandomState(0)
    for tag, arr in [("ecapa", emb["ecapa"]), ("resnet", emb["resnet"])]:
        A = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        same = [float(A[a] @ A[b]) for s, ii in byk.items() if len(ii) >= 2
                for a, b in [rng.choice(ii, 2, replace=False)]]
        diff = []
        while len(diff) < 2000:
            a, b = rng.choice(ok_new, 2, replace=False)
            if labels[a] != labels[b]:
                diff.append(float(A[a] @ A[b]))
        print("  %-8s same %.4f  diff %.4f  separation %.4f"
              % (tag, np.mean(same), np.mean(diff), np.mean(same) - np.mean(diff)))
    print("\n  for reference, the energy-VAD pipeline gave:")
    print("    ecapa separation 0.6036   resnet separation 0.6495")
    print("  separation alone has misled me twice, so the real verdict comes from")
    print("  the holdout protocol run locally on these vectors.")

    hr("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

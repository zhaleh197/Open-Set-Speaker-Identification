"""
IAAA 3rd Contest - Speaker Identification
Notebook 05: extract embeddings from two more speaker encoders.

Why: the error analysis says 22% of errors are genuine speaker confusion, which
only a better representation fixes. ECAPA-VoxCeleb is one 2020-era model trained
on English; fusing architecturally different encoders is the standard remedy.

  spkrec-resnet-voxceleb   different architecture, same training data
  wavlm-base-plus-sv       self-supervised pretraining, very different inductive
                           bias -- the useful kind of diversity for fusion

Preprocessing is byte-identical to notebook 02 so the three embedding sets line
up file-for-file and can be fused at score level.
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
    try:
        import speechbrain  # noqa: F401
    except ImportError:
        print("installing speechbrain (--no-deps, preserving torch %s)" % torch.__version__, flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hyperpyyaml"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "speechbrain"], check=True)


def pick_device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    cap = torch.cuda.get_device_capability(0)
    print("gpu: %s sm_%d%d" % (torch.cuda.get_device_name(0), cap[0], cap[1]))
    a = torch.randn(64, 64, device="cuda")
    torch.mm(a, a).sum().item()
    return "cuda:0"


def read_mono(path, max_s=90.0):
    import soundfile as sf
    try:
        with sf.SoundFile(str(path)) as f:
            n = min(len(f), int(max_s * f.samplerate))
            y = f.read(frames=n, dtype="float32", always_2d=True)
            sr = f.samplerate
        y = y[:, 0]
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


def extract(files, audio_dir, encode, dim, tag):
    """encode: (chunks ndarray) -> (n_chunks, dim) ndarray"""
    import torch
    out, skipped, t0, n_err = [], {}, time.time(), 0
    for i, fn in enumerate(files):
        vec = np.zeros(dim, dtype="float32")
        try:
            s = speech_only(read_mono(audio_dir / fn))
            if len(s) < MIN_SPEECH * SR:
                skipped[fn] = "no_speech"
            else:
                e = encode(to_chunks(s))
                e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
                m = e.mean(0)
                vec = (m / (np.linalg.norm(m) + 1e-9)).astype("float32")
        except Exception as ex:
            skipped[fn] = "error: %s" % str(ex).splitlines()[0]
            n_err += 1
            if n_err <= 3:
                print("   error on %s: %s" % (fn, str(ex).splitlines()[0]), flush=True)
            if n_err == 25:
                raise RuntimeError("25 failures, aborting: %s" % skipped[fn])
        out.append(vec)
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print("   [%s] %5d/%d  %.0fs  eta %.1f min"
                  % (tag, i + 1, len(files), el,
                     el / (i + 1) * (len(files) - i - 1) / 60), flush=True)
    return np.stack(out), skipped


def main():
    hr("0. SETUP")
    ensure_speechbrain()
    import torch
    device = pick_device()

    lab = list(Path("/kaggle/input").rglob("labels.csv"))[0]
    audio_dir = lab.parent
    rows = list(csv.DictReader(open(lab, encoding="utf-8")))
    files = [r["audio_file"] for r in rows]
    print("files:", len(files), " audio dir:", audio_dir)

    # ---------------- ResNet ----------------
    hr("1. spkrec-resnet-voxceleb")
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        src = "speechbrain/spkrec-resnet-voxceleb"
        try:
            enc = EncoderClassifier.from_hparams(source=src, savedir=str(OUT / "resnet"),
                                                 run_opts={"device": device})
        except Exception as ex:
            print("resnet unavailable (%s); falling back to xvector" % str(ex).splitlines()[0])
            src = "speechbrain/spkrec-xvect-voxceleb"
            enc = EncoderClassifier.from_hparams(source=src, savedir=str(OUT / "xvect"),
                                                 run_opts={"device": device})
        enc.eval()
        with torch.no_grad():
            dim = enc.encode_batch(torch.zeros(1, SR, device=device) + 1e-3).shape[-1]
        print("source:", src, " dim:", dim, flush=True)

        def enc_sb(ch):
            with torch.no_grad():
                return enc.encode_batch(torch.from_numpy(ch).to(device)).squeeze(1).cpu().numpy()

        emb, sk = extract(files, audio_dir, enc_sb, dim, "resnet")
        np.save(OUT / "emb_resnet.npy", emb)
        json.dump({"source": src, "dim": dim, "skipped": sk},
                  open(OUT / "meta_resnet.json", "w"))
        print("saved emb_resnet.npy", emb.shape, " skipped:", len(sk))
        del enc
        torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc()
        print("!! resnet stage failed, continuing")

    # ---------------- WavLM ----------------
    hr("2. microsoft/wavlm-base-plus-sv")
    try:
        from transformers import AutoFeatureExtractor, WavLMForXVector
        name = "microsoft/wavlm-base-plus-sv"
        fe = AutoFeatureExtractor.from_pretrained(name)
        mdl = WavLMForXVector.from_pretrained(name).to(device).eval()
        dim = mdl.config.xvector_output_dim
        print("dim:", dim, flush=True)

        def enc_wavlm(ch):
            inp = fe(list(ch), sampling_rate=SR, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                return mdl(**inp).embeddings.cpu().numpy()

        emb, sk = extract(files, audio_dir, enc_wavlm, dim, "wavlm")
        np.save(OUT / "emb_wavlm.npy", emb)
        json.dump({"source": name, "dim": dim, "skipped": sk},
                  open(OUT / "meta_wavlm.json", "w"))
        print("saved emb_wavlm.npy", emb.shape, " skipped:", len(sk))
    except Exception:
        traceback.print_exc()
        print("!! wavlm stage failed, continuing")

    hr("3. QUICK SANITY PER MODEL")
    labels = [r["speaker_id"] for r in rows]
    known = [i for i, l in enumerate(labels) if l != "unknown"]
    by = collections.defaultdict(list)
    for i in known:
        by[labels[i]].append(i)
    rng = np.random.RandomState(0)
    for f in ["emb_resnet.npy", "emb_wavlm.npy"]:
        p = OUT / f
        if not p.exists():
            continue
        E = np.load(p)
        ok = np.flatnonzero(np.abs(E).sum(1) > 0)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        okset = set(ok.tolist())
        same = [float(E[a] @ E[b]) for s, idx in by.items()
                if len([i for i in idx if i in okset]) >= 2
                for a, b in [rng.choice([i for i in idx if i in okset], 2, replace=False)]]
        diff = []
        for _ in range(2000):
            a, b = rng.choice(ok, 2, replace=False)
            if labels[a] != labels[b]:
                diff.append(float(E[a] @ E[b]))
        print("  %-18s usable %4d  same %.4f  diff %.4f  separation %.4f"
              % (f, len(ok), np.mean(same), np.mean(diff), np.mean(same) - np.mean(diff)))

    hr("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

"""
IAAA 3rd Contest - Speaker Identification
Notebook 09: embeddings from the 3D-Speaker encoders (CAM++, ERes2Net, ERes2NetV2).

Why these. Our stack runs ECAPA (0.86% EER on VoxCeleb1-O) and ResNet (1.05%).
The 3D-Speaker family reaches 0.65% (CAM++), 0.61% (ERes2NetV2) and 0.52%
(ERes2Net-large) on the same benchmark -- up to 40% relative error reduction.
ERes2NetV2 was specifically designed for SHORT utterances, and 46% of our errors
sit on files under 10 seconds, so the match to our failure mode is not incidental.

modelscope >= 1.38.1 is listed in the contest's own pyproject.toml, so these
weights are inside the rules.

Install discipline: same --no-deps pattern that kept torch 2.10.0+cu128 intact in
notebooks 02-07. A plain `pip install modelscope` drags in its own torch pin and
we lose the GPU.

Preprocessing is byte-identical to notebooks 02 and 05 so every embedding set
lines up file-for-file for score fusion.
"""
import os, sys, csv, json, time, collections, subprocess, traceback
from pathlib import Path

import numpy as np

SR, CHUNK_S, MAX_CHUNKS, MIN_SPEECH = 16000, 6.0, 10, 1.0
OUT = Path("/kaggle/working")

MODELS = [
    ("campplus",   "iic/speech_campplus_sv_zh-cn_16k-common"),
    ("eres2netv2", "iic/speech_eres2netv2_sv_zh-cn_16k-common"),
    ("eres2net",   "iic/speech_eres2net_sv_zh-cn_16k-common"),
]


def hr(t=""):
    print("\n" + "=" * 72); print(t); print("=" * 72, flush=True)


def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + list(args), check=False)


FORBIDDEN = ("torch", "torchaudio", "torchvision", "nvidia", "triton")


def check_torch(before):
    from importlib.metadata import version
    if not before.startswith(version("torch")):
        raise RuntimeError("torch was clobbered: %s -> %s (loaded module is stale)"
                           % (before, version("torch")))


def ensure_modelscope(max_rounds=8):
    """Install modelscope without letting it replace torch.

    A plain install pins its own torch and the T4 kernels stop working. --no-deps
    avoids that but leaves modelscope's light dependencies missing, one import at
    a time. Rather than burning one Kaggle run per missing module, read the name
    out of the ModuleNotFoundError and install it here -- refusing outright to
    touch anything torch-shaped, and re-checking torch after every install.
    """
    import torch
    before = torch.__version__
    pip("addict", "simplejson", "sortedcontainers", "einops")
    pip("--no-deps", "modelscope")
    check_torch(before)

    for attempt in range(max_rounds):
        try:
            import modelscope  # noqa: F401
            print("modelscope", modelscope.__version__, "| torch intact:", before)
            return
        except ModuleNotFoundError as ex:
            missing = (ex.name or "").split(".")[0]
            if not missing:
                raise
            if any(missing.startswith(f) for f in FORBIDDEN):
                raise RuntimeError(
                    "modelscope wants to pull %s, which would break the GPU. Stopping."
                    % missing)
            print("  [%d] missing %r -> installing" % (attempt + 1, missing), flush=True)
            pip(missing.replace("_", "-"))
            try:
                __import__(missing)
            except ImportError:
                pip(missing)
            check_torch(before)
            for m in [k for k in list(sys.modules) if k.startswith("modelscope")]:
                del sys.modules[m]
    raise RuntimeError("modelscope still not importable after %d rounds" % max_rounds)


def pick_device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    cap = torch.cuda.get_device_capability(0)
    print("gpu: %s sm_%d%d" % (torch.cuda.get_device_name(0), cap[0], cap[1]))
    a = torch.randn(64, 64, device="cuda"); torch.mm(a, a).sum().item()
    return "cuda"


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


def load_encoder(model_id, device):
    """Pull the checkpoint through modelscope, then drive the torch module
    directly. The pipeline API adds its own IO layer we do not want in a loop
    over 4529 files."""
    import torch
    from modelscope.hub.snapshot_download import snapshot_download
    d = Path(snapshot_download(model_id))
    print("   snapshot:", d, flush=True)
    cfgs = sorted(d.glob("*.yaml")) + sorted(d.glob("*.json"))
    print("   files:", [p.name for p in sorted(d.iterdir())][:10], flush=True)

    from modelscope.pipelines import pipeline
    pl = pipeline(task="speaker-verification", model=model_id, device=device)
    return pl


def embed_with_pipeline(pl, chunks):
    """modelscope's sv pipeline returns an embedding per input when asked."""
    out = []
    for c in chunks:
        r = pl([c.astype("float32")], output_emb=True)
        e = r["embs"] if isinstance(r, dict) and "embs" in r else r
        out.append(np.asarray(e).reshape(-1))
    return np.stack(out)


def extract(files, audio_dir, embed_fn, tag):
    out, skipped, t0, n_err, dim = [], {}, time.time(), 0, None
    for i, fn in enumerate(files):
        vec = None
        try:
            s = speech_only(read_mono(audio_dir / fn))
            if len(s) < MIN_SPEECH * SR:
                skipped[fn] = "no_speech"
            else:
                e = embed_fn(to_chunks(s))
                e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
                m = e.mean(0)
                vec = (m / (np.linalg.norm(m) + 1e-9)).astype("float32")
                dim = len(vec)
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
    if dim is None:
        raise RuntimeError("no file produced an embedding")
    E = np.stack([v if v is not None else np.zeros(dim, "float32") for v in out])
    return E, skipped


def main():
    hr("0. SETUP")
    ensure_modelscope()
    device = pick_device()

    lab = list(Path("/kaggle/input").rglob("labels.csv"))[0]
    audio_dir = lab.parent
    rows = list(csv.DictReader(open(lab, encoding="utf-8")))
    files = [r["audio_file"] for r in rows]
    labels = [r["speaker_id"] for r in rows]
    print("files:", len(files))

    done = []
    for tag, model_id in MODELS:
        hr("%s  (%s)" % (tag, model_id))
        try:
            pl = load_encoder(model_id, device)
            # prove it runs before committing to 4529 files
            probe = embed_with_pipeline(pl, np.zeros((1, SR), dtype="float32") + 1e-3)
            print("   smoke test OK, dim =", probe.shape[-1], flush=True)

            E, sk = extract(files, audio_dir, lambda c: embed_with_pipeline(pl, c), tag)
            np.save(OUT / ("emb_%s.npy" % tag), E)
            json.dump({"model": model_id, "dim": int(E.shape[1]), "skipped": sk},
                      open(OUT / ("meta_%s.json" % tag), "w"))
            print("saved emb_%s.npy %s  skipped %d" % (tag, E.shape, len(sk)))
            done.append(tag)
            del pl
            import torch
            torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()
            print("!! %s failed, continuing to the next model" % tag)

    hr("SEPARATION PER MODEL")
    print("(same-speaker minus different-speaker cosine; ECAPA scored 0.6036,")
    print(" ResNet 0.6495 -- a bigger gap means a sharper representation)\n")
    known_by = collections.defaultdict(list)
    for i, l in enumerate(labels):
        if l != "unknown":
            known_by[l].append(i)
    rng = np.random.RandomState(0)
    for tag in done:
        E = np.load(OUT / ("emb_%s.npy" % tag))
        ok = np.flatnonzero(np.abs(E).sum(1) > 0)
        okset = set(ok.tolist())
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        same = []
        for s, idx in known_by.items():
            g = [i for i in idx if i in okset]
            if len(g) >= 2:
                a, b = rng.choice(g, 2, replace=False)
                same.append(float(E[a] @ E[b]))
        diff = []
        while len(diff) < 2000:
            a, b = rng.choice(ok, 2, replace=False)
            if labels[a] != labels[b]:
                diff.append(float(E[a] @ E[b]))
        print("  %-12s dim %3d  same %.4f  diff %.4f  separation %.4f"
              % (tag, E.shape[1], np.mean(same), np.mean(diff),
                 np.mean(same) - np.mean(diff)))

    hr("DONE  (extracted: %s)" % (", ".join(done) if done else "NOTHING"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

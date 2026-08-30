"""
IAAA 3rd Contest - Speaker Identification
Notebook 03: build and self-test the deliverable submission bundle.

Consumes the embeddings from notebook 02 (attached as a kernel source), bundles
the ECAPA weights so the script needs no internet at judging time, writes
submission.py, then actually runs it through the exact command the organisers
will use and validates the CSV it produces.

Method (chosen by measurement in notebook 02 + the local sweep):
  1-NN over every usable training file, cosine on ECAPA embeddings.
  Local 5-fold macro-F1 = 0.9478 +/- 0.0144.
  Clustering the unknown bucket was tested and is unnecessary: 1-NN already
  gives every unknown file its own prototype, which is what mattered.
"""
import os, sys, json, shutil, subprocess, time, collections, traceback
from pathlib import Path

import numpy as np

OUT = Path("/kaggle/working")
BUNDLE = OUT / "submission"
SEED = 0


def hr(t=""):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72, flush=True)


def ensure_speechbrain():
    import torch
    before = torch.__version__
    try:
        import speechbrain  # noqa: F401
    except ImportError:
        print("installing speechbrain (--no-deps, preserving torch %s) ..." % before, flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hyperpyyaml"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "speechbrain"], check=True)


# --------------------------------------------------------------------------
# the deliverable itself
# --------------------------------------------------------------------------
SUBMISSION_SRC = r'''
# IAAA 3rd Contest - Speaker Identification
# Run exactly as the organisers specify:
#   python submission.py --data-dir <dir> --predictions-file-path <out.csv>
#
# Output format is the one the grader actually accepts (verified empirically):
#   audio_file,speaker_id
# where speaker_id is a training speaker UUID or the literal string "unknown".
#
# Method: 1-NN over precomputed ECAPA embeddings of every usable training file.
#
# Note on the audio: the contest ships WAV data (PCM_16, 16 kHz, stereo with
# identical channels) under a .mp3 extension. soundfile reads that directly with
# no decode; librosa is kept as a fallback in case the judging set is real mp3.

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
CHUNK_S = 6.0
MAX_CHUNKS = 10
MIN_SPEECH = 1.0
UNKNOWN = "unknown"
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}

HERE = Path(__file__).resolve().parent


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


def pick_device():
    import torch
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device -- this will be slow", file=sys.stderr)
        return "cpu"
    try:
        a = torch.randn(64, 64, device="cuda")
        torch.mm(a, a).sum().item()
        return "cuda:0"
    except Exception as ex:
        # Pascal (sm_60) cards cannot run torch cu128 wheels. Say so loudly
        # rather than silently dropping to CPU and blowing the time budget.
        print("WARNING: GPU present but unusable (%s); falling back to CPU"
              % str(ex).splitlines()[0], file=sys.stderr)
        return "cpu"


def load_data_files(data_dir):
    d = Path(data_dir)
    files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXT)
    if not files:
        files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--predictions-file-path", required=True)
    args = ap.parse_args()

    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    proto = np.load(HERE / "train_emb.npy")
    proto_lab = json.load(open(HERE / "train_labels.json"))
    proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-9)
    print("prototypes: %d over %d distinct labels"
          % (len(proto), len(set(proto_lab))), flush=True)

    device = pick_device()
    enc = EncoderClassifier.from_hparams(
        source=str(HERE / "models" / "ecapa"),
        savedir=str(HERE / "models" / "ecapa"),
        run_opts={"device": device},
    )
    enc.eval()

    files = load_data_files(args.data_dir)
    print("found %d audio files in %s" % (len(files), args.data_dir), flush=True)

    rows, t0, n_fail = [], time.time(), 0
    for i, p in enumerate(files):
        label = UNKNOWN                       # safe default: ~50% of the set is unknown
        try:
            s = speech_only(read_mono(p))
            if len(s) >= MIN_SPEECH * SR:
                t = torch.from_numpy(to_chunks(s)).to(device)
                with torch.no_grad():
                    e = enc.encode_batch(t).squeeze(1).cpu().numpy()
                e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
                m = e.mean(0)
                m = m / (np.linalg.norm(m) + 1e-9)
                label = proto_lab[int((proto @ m).argmax())]
        except Exception as ex:
            n_fail += 1
            if n_fail <= 5:
                print("  unreadable %s (%s) -> %s"
                      % (p.name, str(ex).splitlines()[0], UNKNOWN), file=sys.stderr)
        rows.append((p.name, label))
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print("  %5d/%d  %.0fs elapsed  eta %.1f min"
                  % (i + 1, len(files), el, el / (i + 1) * (len(files) - i - 1) / 60),
                  flush=True)

    out = Path(args.predictions_file_path)
    if out.parent and str(out.parent) not in ("", "."):
        os.makedirs(out.parent, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write("audio_file,speaker_id\n")
        for name, lab in rows:
            fh.write("%s,%s\n" % (name, lab))

    n_unk = sum(1 for _, l in rows if l == UNKNOWN)
    print("wrote %d predictions to %s  (%.1f%% unknown, %d unreadable, %.1f min)"
          % (len(rows), out, n_unk / max(len(rows), 1) * 100, n_fail,
             (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
'''


# --------------------------------------------------------------------------
def find_embeddings():
    """Notebook 02's output, attached as a kernel source."""
    root = Path("/kaggle/input")
    hits = list(root.rglob("emb_file.npy"))
    if not hits:
        raise SystemExit("emb_file.npy not found -- attach notebook 02 as a kernel source")
    p = hits[0]
    print("embeddings:", p)
    return np.load(p), json.load(open(p.parent / "meta.json"))


def materialise(d):
    """speechbrain may populate savedir with symlinks into the HF cache; the
    bundle has to carry real bytes."""
    n = 0
    for p in Path(d).rglob("*"):
        if p.is_symlink():
            target = p.resolve()
            p.unlink()
            shutil.copy2(target, p)
            n += 1
    return n


def main():
    hr("0. SETUP")
    ensure_speechbrain()
    import torch
    print("torch", torch.__version__, "| gpu:",
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

    emb, meta = find_embeddings()
    labels = meta["labels"]
    valid = np.array(meta["valid"])
    keep = np.flatnonzero(valid)
    proto = emb[keep]
    proto_lab = [labels[i] for i in keep]
    print("prototypes: %d (dropped %d unusable) over %d distinct labels"
          % (len(proto), len(valid) - len(keep), len(set(proto_lab))))

    hr("1. BUILD BUNDLE")
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "models").mkdir(parents=True)

    np.save(BUNDLE / "train_emb.npy", proto.astype("float32"))
    json.dump(proto_lab, open(BUNDLE / "train_labels.json", "w"))
    (BUNDLE / "submission.py").write_text(SUBMISSION_SRC.lstrip(), encoding="utf-8")

    from speechbrain.inference.speaker import EncoderClassifier
    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(BUNDLE / "models" / "ecapa"),
        run_opts={"device": "cpu"},
    )
    n = materialise(BUNDLE / "models" / "ecapa")
    print("ecapa weights bundled (%d symlinks replaced with real files)" % n)
    for p in sorted((BUNDLE / "models" / "ecapa").rglob("*")):
        if p.is_file():
            print("   %-34s %8.1f MB" % (p.name, p.stat().st_size / 1e6))
    total = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file())
    print("bundle total: %.1f MB" % (total / 1e6))

    hr("2. SELF-TEST: run the organisers' exact command")
    src_dir = Path(meta.get("audio_dir", "")) if meta.get("audio_dir") else None
    if src_dir is None or not src_dir.exists():
        hits = list(Path("/kaggle/input").rglob("labels.csv"))
        src_dir = hits[0].parent if hits else None
    if src_dir is None:
        print("!! source audio not attached -- skipping the end-to-end test")
        return

    sample_dir = OUT / "selftest_data"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True)
    rng = np.random.RandomState(SEED)
    picks = [meta["files"][i] for i in rng.choice(keep, 60, replace=False)]
    # include a known-corrupt file: the judging set may contain them too
    bad = [f for f, why in meta["skipped"].items() if "no_speech" in why][:3]
    for f in picks + bad:
        shutil.copy2(src_dir / f, sample_dir / f)
    print("sample dir: %d files (%d usable + %d deliberately corrupt)"
          % (len(picks) + len(bad), len(picks), len(bad)))

    csv_path = OUT / "selftest_submission.csv"
    cmd = [sys.executable, str(BUNDLE / "submission.py"),
           "--data-dir", str(sample_dir),
           "--predictions-file-path", str(csv_path)]
    print("\n$ " + " ".join(cmd) + "\n", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr.strip():
        print("--- stderr ---\n" + r.stderr)
    print("exit code: %d   wall time: %.1fs" % (r.returncode, time.time() - t0))
    if r.returncode != 0:
        raise SystemExit("submission.py failed -- bundle is not deliverable")

    hr("3. VALIDATE THE CSV")
    import csv as csvmod
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csvmod.reader(fh))
    header, body = rows[0], rows[1:]
    truth = dict(zip(meta["files"], labels))
    valid_labels = set(labels)

    checks = []
    checks.append(("header is audio_file,speaker_id", header == ["audio_file", "speaker_id"]))
    checks.append(("one row per input file", len(body) == len(picks) + len(bad)))
    checks.append(("no missing/extra files",
                   {r[0] for r in body} == {p.name for p in sample_dir.iterdir()}))
    checks.append(("every label is a real speaker id or 'unknown'",
                   all(r[1] in valid_labels for r in body)))
    checks.append(("corrupt files fell back to 'unknown'",
                   all(dict(body)[f] == "unknown" for f in bad)))
    for name, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))

    # consistency: the bundle embeds these files, so it must recognise them.
    # a miss here means submission.py preprocesses differently than notebook 02.
    hit = sum(1 for f, p in body if f in truth and p == truth[f])
    n_known = sum(1 for f, _ in body if f in truth)
    print("\n  pipeline consistency: %d/%d sample files matched their own label"
          % (hit, n_known))
    print("  (these are bundled prototypes, so anything below ~100%% means the")
    print("   submission-time preprocessing diverges from the extraction pipeline)")

    if not all(ok for _, ok in checks):
        raise SystemExit("validation failed -- do not ship this bundle")

    hr("4. PACKAGE")
    zip_path = shutil.make_archive(str(OUT / "submission_bundle"), "zip", str(BUNDLE))
    print("ready:", zip_path, "(%.1f MB)" % (os.path.getsize(zip_path) / 1e6))
    print("\nrun it with:")
    print("  python submission.py --data-dir <dir> --predictions-file-path <out.csv>")

    hr("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

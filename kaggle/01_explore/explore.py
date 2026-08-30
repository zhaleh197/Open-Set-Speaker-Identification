"""
IAAA 3rd Contest - Speaker Identification
Notebook 01: data discovery + UUIDv7 hypothesis test + ECAPA smoke test

Run on Kaggle with GPU + Internet enabled.
Cheap by design: nothing here should take more than a few minutes.
"""
import os, sys, csv, json, glob, random, datetime, collections, traceback
from pathlib import Path

random.seed(0)


def hr(title=""):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72, flush=True)


def safe(fn):
    """Run a section, never let one failure kill the whole run."""
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except Exception:
            print("!! section %s FAILED:" % fn.__name__)
            traceback.print_exc()
            return None
    return wrapper


# --------------------------------------------------------------------------
# 1. environment
# --------------------------------------------------------------------------
@safe
def sec_env():
    hr("1. ENVIRONMENT")
    print("python  :", sys.version.split()[0])
    for mod in ["numpy", "pandas", "torch", "torchaudio", "librosa",
                "soundfile", "speechbrain", "sklearn", "transformers"]:
        try:
            m = __import__(mod)
            print("  %-14s %s" % (mod, getattr(m, "__version__", "?")))
        except ImportError:
            print("  %-14s -- NOT INSTALLED" % mod)
    try:
        import torch
        print("cuda available :", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu            :", torch.cuda.get_device_name(0))
    except Exception:
        pass
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=10)
        print("internet       : OK")
    except Exception as e:
        print("internet       : BLOCKED ->", e)


# --------------------------------------------------------------------------
# 2. what is actually mounted under /kaggle/input
# --------------------------------------------------------------------------
@safe
def sec_tree():
    hr("2. INPUT TREE")
    root = Path("/kaggle/input")
    if not root.exists():
        print("no /kaggle/input -- running outside Kaggle?")
        return
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        depth = len(d.relative_to(root).parts)
        if depth > 3:
            dirnames[:] = []
            continue
        pad = "  " * depth
        if filenames:
            ext = collections.Counter(Path(f).suffix.lower() for f in filenames)
            print("%s%s/  [%d files] %s" % (pad, d.name, len(filenames), dict(ext)))
            for f in sorted(filenames)[:3]:
                print("%s  - %s" % (pad, f))
        else:
            print("%s%s/" % (pad, d.name))


# --------------------------------------------------------------------------
# 3. any csv/json metadata sitting in the input
# --------------------------------------------------------------------------
@safe
def sec_tables():
    hr("3. METADATA FILES")
    for p in sorted(Path("/kaggle/input").rglob("*.csv"))[:20]:
        print("\n--- %s" % p)
        try:
            with open(p, newline="", encoding="utf-8", errors="replace") as fh:
                rows = list(csv.reader(fh))
            print("    rows=%d  header=%s" % (len(rows), rows[0] if rows else None))
            for r in rows[1:4]:
                print("   ", r)
        except Exception as e:
            print("    unreadable:", e)
    for p in sorted(Path("/kaggle/input").rglob("*.json"))[:10]:
        print("\n--- %s  (%d bytes)" % (p, p.stat().st_size))


# --------------------------------------------------------------------------
# 4. THE BIG ONE: are test filenames UUIDv7 (i.e. timestamped)?
# --------------------------------------------------------------------------
def uuid_version(name):
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) != 5 or len(parts[2]) != 4:
        return None
    return parts[2][0]


def uuid7_time(name):
    """UUIDv7: first 48 bits = unix milliseconds."""
    hexs = Path(name).stem.replace("-", "")
    return int(hexs[:12], 16) / 1000.0


@safe
def sec_uuid():
    hr("4. UUID VERSION / TIMESTAMP ANALYSIS")
    audio = list(Path("/kaggle/input").rglob("*.mp3"))
    audio += list(Path("/kaggle/input").rglob("*.wav"))
    print("total audio files found:", len(audio))
    if not audio:
        print("no audio found -- adjust the glob once the tree above is known")
        return

    by_dir = collections.defaultdict(list)
    for p in audio:
        by_dir[p.parent].append(p.name)

    for d, names in by_dir.items():
        vers = collections.Counter(uuid_version(n) for n in names)
        print("\n--- %s  (%d files)" % (d, len(names)))
        print("    uuid versions:", dict(vers))

        v7 = [n for n in names if uuid_version(n) == "7"]
        if len(v7) < 10:
            continue

        ts = sorted(uuid7_time(n) for n in v7)
        span = ts[-1] - ts[0]
        print("    v7 count: %d" % len(v7))
        print("    earliest:", datetime.datetime.fromtimestamp(ts[0], datetime.UTC))
        print("    latest  :", datetime.datetime.fromtimestamp(ts[-1], datetime.UTC))
        print("    span    : %.2f hours" % (span / 3600))

        gaps = [b - a for a, b in zip(ts, ts[1:])]
        gaps_sorted = sorted(gaps)
        n = len(gaps_sorted)
        print("    consecutive-gap percentiles (seconds):")
        for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
            print("      p%02d: %10.3f" % (int(q * 100), gaps_sorted[int(q * (n - 1))]))
        for thr in (1, 2, 5, 10, 30, 60):
            share = sum(g < thr for g in gaps) / n
            print("      gaps < %3ds : %5.1f%%" % (thr, share * 100))
        print("    -> strongly bimodal gaps would suggest session bursts")
        print("       (verified against embeddings in section 6)")


# --------------------------------------------------------------------------
# 5. audio probe
# --------------------------------------------------------------------------
@safe
def sec_audio():
    hr("5. AUDIO PROBE")
    import librosa
    audio = list(Path("/kaggle/input").rglob("*.mp3"))[:400]
    if not audio:
        print("no mp3 found")
        return
    sample = random.sample(audio, min(15, len(audio)))
    durs, srs = [], []
    for p in sample:
        try:
            y, sr = librosa.load(p, sr=None, mono=True)
            durs.append(len(y) / sr)
            srs.append(sr)
            print("  %-42s %7.2fs  sr=%s" % (p.name[:40], len(y) / sr, sr))
        except Exception as e:
            print("  %-42s LOAD FAILED: %s" % (p.name[:40], e))
    if durs:
        print("\n  duration: min %.1fs  mean %.1fs  max %.1fs"
              % (min(durs), sum(durs) / len(durs), max(durs)))
        print("  sample rates seen:", collections.Counter(srs))


# --------------------------------------------------------------------------
# 6. ECAPA smoke test + adjacency-vs-similarity check
# --------------------------------------------------------------------------
@safe
def sec_ecapa():
    hr("6. ECAPA SMOKE TEST")
    import torch, numpy as np, librosa
    from speechbrain.inference.speaker import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/kaggle/working/ecapa",
        run_opts={"device": device},
    )
    print("ECAPA loaded on", device)

    def embed(path, sr=16000, max_s=30):
        y, _ = librosa.load(path, sr=sr, mono=True, duration=max_s)
        t = torch.tensor(y, device=device).unsqueeze(0)
        with torch.no_grad():
            e = enc.encode_batch(t).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    audio = list(Path("/kaggle/input").rglob("*.mp3"))
    if not audio:
        print("no audio")
        return
    by_dir = collections.defaultdict(list)
    for p in audio:
        by_dir[p.parent].append(p)
    target = max(by_dir, key=lambda d: len(by_dir[d]))
    files = by_dir[target]
    print("using %s  (%d files)" % (target, len(files)))

    v7 = [p for p in files if uuid_version(p.name) == "7"]
    if len(v7) >= 60:
        files = sorted(v7, key=lambda p: uuid7_time(p.name))[:60]
        ordered_by_time = True
        print("files are UUIDv7 -> sorted by embedded timestamp")
    else:
        files = files[:60]
        ordered_by_time = False
        print("files are not UUIDv7 -> arbitrary order")

    embs = []
    for p in files:
        try:
            embs.append(embed(p))
        except Exception as e:
            print("embed failed", p.name, e)
            embs.append(None)
    keep = [(p, e) for p, e in zip(files, embs) if e is not None]
    print("embedded %d/%d files" % (len(keep), len(files)))
    if len(keep) < 20:
        return
    E = np.stack([e for _, e in keep])

    adjacent = [float(E[i] @ E[i + 1]) for i in range(len(E) - 1)]
    rnd = []
    for _ in range(400):
        i, j = random.sample(range(len(E)), 2)
        rnd.append(float(E[i] @ E[j]))

    mean = lambda v: sum(v) / len(v)
    print("\n  cosine sim, temporally ADJACENT pairs : %.4f" % mean(adjacent))
    print("  cosine sim, RANDOM pairs              : %.4f" % mean(rnd))
    print("  share of adjacent pairs > 0.5         : %.1f%%"
          % (sum(a > .5 for a in adjacent) / len(adjacent) * 100))
    print("  share of random   pairs > 0.5         : %.1f%%"
          % (sum(a > .5 for a in rnd) / len(rnd) * 100))
    if ordered_by_time:
        print("\n  >> if adjacent >> random, the UUIDv7 ordering leaks speaker identity")
    else:
        print("\n  (no timestamp ordering available -- this is just a sanity baseline)")


if __name__ == "__main__":
    sec_env()
    sec_tree()
    sec_tables()
    sec_uuid()
    sec_audio()
    sec_ecapa()
    hr("DONE")

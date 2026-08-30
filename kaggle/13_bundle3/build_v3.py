"""
IAAA 3rd Contest - Speaker Identification
Notebook 13: build and self-test submission bundle v3.

v2 scored 0.9660. v3 keeps that bundle byte-for-byte -- same two encoders, same
prototypes -- and changes only the decision step, adding a mild Sinkhorn prior
correction worth about +0.0026 in cross-validation.

Deliberately NOT included: ERes2NetV2. On clean folds it looked worth +0.0016,
but on folds with realistic uneven speaker counts it added exactly nothing
(0.9493 vs 0.9492), so there was no case for taking on modelscope offline-loading
risk at judging time for it.

  kept     ECAPA + ResNet score fusion (1:3)
           AS-norm per model before fusing
           margin threshold tau = 0.20 on (best known - best unknown)
           guard for the 34 non-speech files whose embeddings collapse

  rejected transductive clustering (-0.004), WavLM fusion (-0.004),
           chunk-level scoring (too slow to judge), AAM fine-tuning (~0, overfits)

Local CV: holdout-1 0.9480, holdout-2 0.9544. v1 measured 0.9422 on holdout-2 and
scored 0.9600 for real, so the observed protocol offset is about +0.018.
"""
import os, sys, csv, json, shutil, subprocess, time, collections, traceback
from pathlib import Path

import numpy as np

OUT = Path("/kaggle/working")
BUNDLE = OUT / "submission"
TAU = 0.20
W_ECAPA, W_RESNET = 1.0, 3.0
COHORT = 300
SINK_EPS, SINK_ITERS, SINK_UNK = 1.0, 3, 0.50
SEED = 0


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


SUBMISSION_SRC = r'''
# IAAA 3rd Contest - Speaker Identification (v2)
#   python submission.py --data-dir <dir> --predictions-file-path <out.csv>
# Output: audio_file,speaker_id   where speaker_id is a training UUID or "unknown".
#
# Pipeline
#   1. read each file (WAV PCM_16 16k stereo behind a .mp3 name; librosa fallback)
#   2. energy VAD -> up to 10 six-second chunks -> mean of L2-normed embeddings
#   3. two encoders: ECAPA and ResNet, both VoxCeleb-pretrained
#   4. AS-norm each score matrix against the training cohort, then fuse 1:3
#   5. predict the best known speaker only if it beats the best unknown
#      prototype by TAU, otherwise "unknown"
#
# Everything the script needs is bundled next to it. No network access required.

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

SR, CHUNK_S, MAX_CHUNKS, MIN_SPEECH = 16000, 6.0, 10, 1.0
UNKNOWN = "unknown"
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}
HERE = Path(__file__).resolve().parent
CFG = json.load(open(HERE / "config.json"))


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


def pick_device():
    import torch
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device; this will be slow", file=sys.stderr)
        return "cpu"
    try:
        a = torch.randn(64, 64, device="cuda")
        torch.mm(a, a).sum().item()
        return "cuda:0"
    except Exception as ex:
        # Pascal cards cannot run torch cu128 wheels. Fail loudly, not silently.
        print("WARNING: GPU unusable (%s); using CPU" % str(ex).splitlines()[0], file=sys.stderr)
        return "cpu"


def l2(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--predictions-file-path", required=True)
    args = ap.parse_args()

    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    d = Path(args.data_dir)
    files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXT)
    if not files:
        files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)
    print("found %d audio files" % len(files), flush=True)

    proto_lab = np.array(json.load(open(HERE / "train_labels.json")))
    device = pick_device()
    t0 = time.time()

    # ---- pass 1: embed every file with every encoder -----------------------
    per_model, failed = {}, set()
    for name in ["ecapa", "resnet"]:
        enc = EncoderClassifier.from_hparams(
            source=str(HERE / "models" / name), savedir=str(HERE / "models" / name),
            run_opts={"device": device})
        enc.eval()
        dim = int(CFG["dims"][name])
        V = np.zeros((len(files), dim), dtype="float32")
        for i, p in enumerate(files):
            try:
                s = speech_only(read_mono(p))
                if len(s) < MIN_SPEECH * SR:
                    failed.add(i); continue
                t = torch.from_numpy(to_chunks(s)).to(device)
                with torch.no_grad():
                    e = enc.encode_batch(t).squeeze(1).cpu().numpy()
                m = l2(l2(e).mean(0))
                V[i] = m.astype("float32")
            except Exception as ex:
                failed.add(i)
                if len(failed) <= 5:
                    print("  unreadable %s: %s" % (p.name, str(ex).splitlines()[0]), file=sys.stderr)
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print("  [%s] %d/%d  %.0fs" % (name, i + 1, len(files), el), flush=True)
        per_model[name] = V
        del enc
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- pass 2: AS-norm, fuse, decide -------------------------------------
    S = 0.0
    wsum = CFG["w_ecapa"] + CFG["w_resnet"]
    for name, w in [("ecapa", CFG["w_ecapa"]), ("resnet", CFG["w_resnet"])]:
        Et = l2(np.load(HERE / ("train_emb_%s.npy" % name)).astype("float64"))
        Ev = l2(per_model[name].astype("float64"))
        M = Ev @ Et.T
        top = np.sort(M, axis=1)[:, -CFG["cohort"]:]
        mv, sv = top.mean(1, keepdims=True), top.std(1, keepdims=True) + 1e-9
        mt = np.array(CFG["cohort_mean"][name])[None, :]
        st = np.array(CFG["cohort_std"][name])[None, :]
        S = S + (w / wsum) * 0.5 * ((M - mv) / sv + (M - mt) / st)

    # collapse prototype scores to one column per class, unknown last
    known_ids = sorted(set(proto_lab) - {UNKNOWN})
    cls = known_ids + [UNKNOWN]
    KU = len(cls) - 1
    C = np.full((len(files), len(cls)), -9e9)
    for ci, c in enumerate(cls):
        cols = np.flatnonzero(proto_lab == c)
        if len(cols):
            C[:, ci] = S[:, cols].max(1)
    best = C[:, :KU].argmax(1)

    # Mild Sinkhorn prior correction.
    #
    # Independent argmax cannot know that roughly half the eval set is unknown
    # (the contest split every person's audio ~50/50, and training measures
    # 50.2% unknown). Left alone it over-claims known identities: the error
    # analysis found 208 false alarms against 101 misses. Sinkhorn nudges the
    # label totals toward the expected split.
    #
    # Three iterations, not more. Run to convergence it also forces every known
    # speaker to receive an equal share, which is false -- speakers contribute
    # different numbers of test files -- and that cost 0.023 macro-F1 in testing.
    # Three iterations is a nudge; ten is a constraint.
    L = C / CFG["sink_eps"]
    L[:, KU] += CFG["tau"] / CFG["sink_eps"]
    share = CFG["sink_unknown_share"]
    tgt = np.full(len(cls), (1.0 - share) / (len(cls) - 1))
    tgt[KU] = share
    logc = np.log(tgt * len(files))
    for _ in range(int(CFG["sink_iters"])):
        L -= L.max(1, keepdims=True)
        L -= np.log(np.exp(L).sum(1, keepdims=True))
        L += logc[None, :] - np.log(np.exp(L).sum(0, keepdims=True) + 1e-30)
    pick = L.argmax(1)

    degen = np.array(CFG["degenerate_vector"])
    dg = l2(per_model["ecapa"].astype("float64")) @ (degen / np.linalg.norm(degen))

    rows = []
    for i, p in enumerate(files):
        if i in failed or dg[i] > 0.999:
            lab = UNKNOWN                    # non-speech: ~50% of the set is unknown
        else:
            lab = cls[pick[i]]
        rows.append((p.name, lab))

    out = Path(args.predictions_file_path)
    if str(out.parent) not in ("", "."):
        os.makedirs(out.parent, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write("audio_file,speaker_id\n")
        for n, l in rows:
            fh.write("%s,%s\n" % (n, l))
    n_unk = sum(1 for _, l in rows if l == UNKNOWN)
    print("wrote %d predictions (%.1f%% unknown, %d unreadable, %.1f min)"
          % (len(rows), n_unk / max(len(rows), 1) * 100, len(failed),
             (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
'''


def materialise(d):
    n = 0
    for p in Path(d).rglob("*"):
        if p.is_symlink():
            t = p.resolve(); p.unlink(); shutil.copy2(t, p); n += 1
    return n


def main():
    hr("0. SETUP")
    ensure_speechbrain()
    import torch
    print("torch", torch.__version__, "| gpu:",
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

    e2 = list(Path("/kaggle/input").rglob("emb_file.npy"))
    e5 = list(Path("/kaggle/input").rglob("emb_resnet.npy"))
    if not e2 or not e5:
        raise SystemExit("attach notebooks 02 and 05 as kernel sources")
    emb_ec = np.load(e2[0]); emb_rn = np.load(e5[0])
    meta = json.load(open(e2[0].parent / "meta.json"))
    labels = meta["labels"]; valid = np.array(meta["valid"])
    keep = np.flatnonzero(valid)
    print("prototypes: %d (dropped %d unusable)" % (len(keep), len(valid) - len(keep)))

    l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    P = {"ecapa": l2(emb_ec[keep].astype("float64")),
         "resnet": l2(emb_rn[keep].astype("float64"))}
    proto_lab = [labels[i] for i in keep]

    hr("1. PRECOMPUTE COHORT STATISTICS")
    cmean, cstd = {}, {}
    for k, E in P.items():
        Ct = E @ E.T
        np.fill_diagonal(Ct, -np.inf)
        top = np.sort(Ct, axis=1)[:, -COHORT:]
        cmean[k] = top.mean(1).tolist()
        cstd[k] = (top.std(1) + 1e-9).tolist()
        print("  %-7s cohort mean %.4f  std %.4f" % (k, np.mean(cmean[k]), np.mean(cstd[k])))

    G = P["ecapa"] @ P["ecapa"].T
    np.fill_diagonal(G, 0)
    dmask = (G > 0.9999).sum(1) > 0
    degen_vec = P["ecapa"][dmask].mean(0) if dmask.any() else np.zeros(P["ecapa"].shape[1])
    print("  degenerate prototypes: %d" % dmask.sum())

    hr("2. BUILD BUNDLE")
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "models").mkdir(parents=True)
    np.save(BUNDLE / "train_emb_ecapa.npy", P["ecapa"].astype("float32"))
    np.save(BUNDLE / "train_emb_resnet.npy", P["resnet"].astype("float32"))
    json.dump(proto_lab, open(BUNDLE / "train_labels.json", "w"))
    json.dump({"tau": TAU, "w_ecapa": W_ECAPA, "w_resnet": W_RESNET, "cohort": COHORT,
               "dims": {"ecapa": P["ecapa"].shape[1], "resnet": P["resnet"].shape[1]},
               "cohort_mean": cmean, "cohort_std": cstd,
               "sink_eps": SINK_EPS, "sink_iters": SINK_ITERS,
               "sink_unknown_share": SINK_UNK,
               "degenerate_vector": degen_vec.tolist()},
              open(BUNDLE / "config.json", "w"))
    (BUNDLE / "submission.py").write_text(SUBMISSION_SRC.lstrip(), encoding="utf-8")

    from speechbrain.inference.speaker import EncoderClassifier
    for name, src in [("ecapa", "speechbrain/spkrec-ecapa-voxceleb"),
                      ("resnet", "speechbrain/spkrec-resnet-voxceleb")]:
        EncoderClassifier.from_hparams(source=src, savedir=str(BUNDLE / "models" / name),
                                       run_opts={"device": "cpu"})
        n = materialise(BUNDLE / "models" / name)
        print("  %-7s bundled (%d symlinks materialised)" % (name, n))
    total = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file())
    print("bundle total: %.1f MB" % (total / 1e6))

    hr("3. SELF-TEST WITH THE ORGANISERS' COMMAND")
    src_dir = list(Path("/kaggle/input").rglob("labels.csv"))[0].parent
    sample = OUT / "selftest_data"
    if sample.exists():
        shutil.rmtree(sample)
    sample.mkdir(parents=True)
    rng = np.random.RandomState(SEED)
    # Sinkhorn reads the composition of whatever it is given, so a sample that is
    # 95% known speakers would be pushed hard toward "unknown" and the check below
    # would fail for the wrong reason. Mirror the real ~50/50 split instead.
    kn = [i for i in keep if labels[i] != "unknown"]
    un = [i for i in keep if labels[i] == "unknown"]
    picks = ([meta["files"][i] for i in rng.choice(kn, 30, replace=False)] +
             [meta["files"][i] for i in rng.choice(un, 30, replace=False)])
    bad = [f for f, w in meta["skipped"].items() if "no_speech" in w][:3]
    for f in picks + bad:
        shutil.copy2(src_dir / f, sample / f)

    csv_path = OUT / "selftest_submission.csv"
    cmd = [sys.executable, str(BUNDLE / "submission.py"),
           "--data-dir", str(sample), "--predictions-file-path", str(csv_path)]
    print("$ " + " ".join(cmd) + "\n", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr.strip():
        print("--- stderr ---\n" + r.stderr[:2000])
    print("exit %d   %.1fs for %d files" % (r.returncode, time.time() - t0, len(picks) + len(bad)))
    if r.returncode != 0:
        raise SystemExit("submission.py failed -- not deliverable")

    hr("4. VALIDATE")
    import csv as cm
    rows = list(cm.reader(open(csv_path, newline="", encoding="utf-8")))
    header, body = rows[0], rows[1:]
    truth = dict(zip(meta["files"], labels))
    checks = [
        ("header is audio_file,speaker_id", header == ["audio_file", "speaker_id"]),
        ("one row per input file", len(body) == len(picks) + len(bad)),
        ("filenames match exactly",
         {r[0] for r in body} == {p.name for p in sample.iterdir()}),
        ("labels are valid ids or 'unknown'", all(r[1] in set(labels) for r in body)),
        ("corrupt files -> unknown", all(dict(body)[f] == "unknown" for f in bad)),
    ]
    for n, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", n))
    hit = sum(1 for f, p in body if f not in bad and truth.get(f) == p)
    print("\n  pipeline consistency: %d/%d bundled prototypes recovered their own label"
          % (hit, len(picks)))
    print("  (below ~100%% would mean submission.py preprocesses differently than")
    print("   the extraction notebooks, which would invalidate the whole bundle)")
    if not all(ok for _, ok in checks):
        raise SystemExit("validation failed")

    hr("5. PACKAGE")
    shutil.rmtree(sample)                      # keep it out of the kernel output
    z = shutil.make_archive(str(OUT / "submission_v3"), "zip", str(BUNDLE))
    print("ready:", z, "(%.1f MB)" % (os.path.getsize(z) / 1e6))
    hr("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

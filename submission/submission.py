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

    # import torch
    # from speechbrain.inference.speaker import EncoderClassifier
    import torch
    import torchaudio

    # ============================================================
    # SpeechBrain 1.1.0 <-> torchaudio 2.11 compatibility patch
    # torchaudio 2.11 removed list_audio_backends(),
    # but SpeechBrain 1.1.0 still expects it during import.
    # ============================================================
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: []

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

    isu = proto_lab == UNKNOWN
    Sk = np.where(isu[None, :], -9e9, S)
    Su = np.where(isu[None, :], S, -9e9)
    sk, su, best = Sk.max(1), Su.max(1), Sk.argmax(1)

    degen = np.array(CFG["degenerate_vector"])
    dg = l2(per_model["ecapa"].astype("float64")) @ (degen / np.linalg.norm(degen))

    rows = []
    for i, p in enumerate(files):
        if i in failed or dg[i] > 0.999:
            lab = UNKNOWN                    # non-speech: ~50% of the set is unknown
        else:
            lab = proto_lab[best[i]] if sk[i] - su[i] > CFG["tau"] else UNKNOWN
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

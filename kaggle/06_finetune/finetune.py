"""
IAAA 3rd Contest - Speaker Identification
Notebook 06: domain-adapt the ECAPA encoder with AAM-softmax.

Rationale. ECAPA-VoxCeleb was trained on English celebrity interviews; this
corpus is a different language and different recording conditions. The pretrained
model already separates speakers well (cosine gap 0.60), so this is adaptation,
not training from scratch: low LR, heavy augmentation, few epochs.

Class design. Training on "446 known + one unknown class" would hand 50% of the
data to a single heterogeneous class and wreck the AAM geometry -- the loss would
try to pull 553 different people into one tight cluster. Instead the unknown
bucket is split into K pseudo-identities by clustering the cached embeddings, so
every class is a real single speaker and the prior is balanced. That is the same
insight that took the baseline from 0.735 to 0.942.

Selection. Checkpoints are chosen by the contest metric (macro-F1 under the
holdout protocol), never by training loss or classification accuracy -- the head
is thrown away afterwards and only the embedding matters.
"""
import os, sys, csv, json, math, time, random, collections, subprocess, traceback
from pathlib import Path

import numpy as np

SR          = 16000
CROP_S      = 4.0
BATCH       = 48
EPOCHS      = 30
LR_ENC      = 5e-5
LR_HEAD     = 1e-3
MARGIN_MAX  = 0.20
SCALE       = 30.0
WARMUP_EP   = 3
K_UNKNOWN   = 553          # the contest states 553 OOD speakers
VAL_EVERY   = 2
SEED        = 0
OUT         = Path("/kaggle/working")

random.seed(SEED); np.random.seed(SEED)


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
        raise RuntimeError("no CUDA device -- fine-tuning needs a GPU (use T4)")
    cap = torch.cuda.get_device_capability(0)
    print("gpu: %s sm_%d%d" % (torch.cuda.get_device_name(0), cap[0], cap[1]))
    a = torch.randn(64, 64, device="cuda"); torch.mm(a, a).sum().item()
    return "cuda:0"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def read_crop(path, rng):
    """Read a random CROP_S window without decoding the whole file. These are
    PCM WAVs (with a .mp3 name), so seeking is free."""
    import soundfile as sf
    need = int(CROP_S * SR)
    with sf.SoundFile(str(path)) as f:
        total, sr = len(f), f.samplerate
        want = int(CROP_S * sr)
        if total <= want:
            y = f.read(dtype="float32", always_2d=True)[:, 0]
        else:
            f.seek(rng.randint(0, total - want))
            y = f.read(frames=want, dtype="float32", always_2d=True)[:, 0]
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < need:
        y = np.tile(y, int(np.ceil(need / max(len(y), 1))))[:need]
    return y[:need]


def augment(y, rng):
    if rng.rand() < 0.5:                                  # speed / tempo jitter
        rate = rng.uniform(0.9, 1.1)
        idx = np.clip((np.arange(int(len(y) / rate)) * rate).astype(int), 0, len(y) - 1)
        y = y[idx]
        need = int(CROP_S * SR)
        y = np.tile(y, int(np.ceil(need / max(len(y), 1))))[:need] if len(y) < need else y[:need]
    if rng.rand() < 0.5:                                  # additive noise at 5-20 dB SNR
        snr = rng.uniform(5, 20)
        p = (y ** 2).mean() + 1e-9
        y = y + rng.randn(len(y)).astype("float32") * np.sqrt(p / (10 ** (snr / 10)))
    if rng.rand() < 0.5:                                  # gain
        y = y * rng.uniform(0.5, 1.5)
    return np.clip(y, -1.0, 1.0).astype("float32")


class Clips:
    def __init__(self, files, targets, audio_dir, train):
        self.files, self.targets, self.dir, self.train = files, targets, audio_dir, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        rng = np.random.RandomState((os.getpid() * 7919 + i * 104729 + random.randint(0, 1 << 30)) % (1 << 31))
        try:
            y = read_crop(self.dir / self.files[i], rng)
            if self.train:
                y = augment(y, rng)
        except Exception:
            y = np.zeros(int(CROP_S * SR), dtype="float32")
        return y, self.targets[i]


def collate(batch):
    import torch
    ys, ts = zip(*batch)
    return torch.from_numpy(np.stack(ys)), torch.tensor(ts, dtype=torch.long)


# --------------------------------------------------------------------------
# AAM-softmax
# --------------------------------------------------------------------------
def build_head(dim, n_classes):
    import torch, torch.nn as nn
    class AAM(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(torch.empty(n_classes, dim))
            nn.init.xavier_normal_(self.W)

        def forward(self, x, y, m):
            cos = torch.nn.functional.linear(
                torch.nn.functional.normalize(x), torch.nn.functional.normalize(self.W)
            ).clamp(-1 + 1e-7, 1 - 1e-7)
            th = torch.acos(cos)
            oh = torch.zeros_like(cos).scatter_(1, y.view(-1, 1), 1.0)
            return torch.cos(th + m * oh) * SCALE
    return AAM()


# --------------------------------------------------------------------------
def main():
    hr("0. SETUP")
    ensure_speechbrain()
    import torch
    from torch.utils.data import DataLoader
    from speechbrain.inference.speaker import EncoderClassifier
    device = pick_device()
    torch.manual_seed(SEED)

    lab = list(Path("/kaggle/input").rglob("labels.csv"))[0]
    audio_dir = lab.parent
    rows = list(csv.DictReader(open(lab, encoding="utf-8")))
    files = [r["audio_file"] for r in rows]
    labels = [r["speaker_id"] for r in rows]

    cached = list(Path("/kaggle/input").rglob("emb_file.npy"))
    if not cached:
        raise SystemExit("attach notebook 02 as a kernel source (needs emb_file.npy)")
    emb = np.load(cached[0])
    meta = json.load(open(cached[0].parent / "meta.json"))
    valid = np.array(meta["valid"])
    assert meta["files"] == files, "cached embeddings are not aligned with labels.csv"
    print("files %d | usable %d" % (len(files), valid.sum()))

    hr("1. CLASS DESIGN")
    known = sorted({l for l in labels if l != "unknown"})
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    unk_idx = np.array([i for i, l in enumerate(labels) if l == "unknown" and valid[i]])
    from sklearn.cluster import AgglomerativeClustering
    k = min(K_UNKNOWN, len(unk_idx))
    cl = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                 linkage="average").fit_predict(E[unk_idx])
    print("unknown bucket split into %d pseudo-identities" % k)
    sizes = np.bincount(cl)
    print("  cluster sizes: min %d  med %d  max %d" % (sizes.min(), int(np.median(sizes)), sizes.max()))

    target = np.full(len(files), -1, dtype=int)
    for ci, s in enumerate(known):
        for i, l in enumerate(labels):
            if l == s and valid[i]:
                target[i] = ci
    for j, c in zip(unk_idx, cl):
        target[j] = len(known) + int(c)
    n_classes = len(known) + k
    print("total classes: %d (%d known + %d pseudo-unknown)" % (n_classes, len(known), k))

    # hold out one file per class for the metric; everything else trains
    rng = np.random.RandomState(SEED)
    by = collections.defaultdict(list)
    for i in np.flatnonzero(valid):
        by[int(target[i])].append(i)
    val_idx, trn_idx = [], []
    for c, idx in by.items():
        idx = list(idx); rng.shuffle(idx)
        if len(idx) >= 3:
            val_idx.append(idx[0]); trn_idx += idx[1:]
        else:
            trn_idx += idx
    val_idx = np.array(val_idx); trn_idx = np.array(trn_idx)
    print("train clips %d | held-out %d" % (len(trn_idx), len(val_idx)))

    hr("2. MODEL")
    wrap = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(OUT / "ecapa"), run_opts={"device": device})
    feats, norm, enc = wrap.mods.compute_features, wrap.mods.mean_var_norm, wrap.mods.embedding_model
    enc.train()
    dim = 192
    head = build_head(dim, n_classes).to(device)
    params = list(enc.parameters()) + list(head.parameters())
    # Two learning rates, not one. The AAM head starts from random weights and
    # needs to move fast; the encoder is already good and must only be nudged.
    # Run 1 used a single low LR for both, the head never left its initialisation,
    # and training accuracy sat at exactly 0.000 for 18 epochs.
    groups = [{"params": list(enc.parameters()), "lr": LR_ENC},
              {"params": list(head.parameters()), "lr": LR_HEAD}]
    opt = torch.optim.AdamW(groups, weight_decay=2e-5)
    print("trainable params: %.1fM  (encoder lr %.0e, head lr %.0e)"
          % (sum(p.numel() for p in params) / 1e6, LR_ENC, LR_HEAD))

    def embed_batch(wav, train):
        f = feats(wav)
        f = norm(f, torch.ones(len(wav), device=wav.device))
        if train:                                        # SpecAugment-style masking
            for _ in range(2):
                t0 = random.randint(0, max(f.shape[1] - 12, 1))
                f[:, t0:t0 + random.randint(2, 12), :] = 0
            f0 = random.randint(0, max(f.shape[2] - 10, 1))
            f[:, :, f0:f0 + random.randint(2, 10)] = 0
        return enc(f).squeeze(1)

    def full_embeddings(idx, bs=64):
        enc.eval()
        out = []
        loader = DataLoader(Clips([files[i] for i in idx], [0] * len(idx), audio_dir, False),
                            batch_size=bs, num_workers=2, collate_fn=collate)
        with torch.no_grad():
            for wav, _ in loader:
                out.append(embed_batch(wav.to(device), False).cpu().numpy())
        enc.train()
        v = np.concatenate(out)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def val_macro_f1():
        """The contest metric on the held-out clips: 1-NN against the training
        clips, with the unknown pseudo-classes folded back to 'unknown'."""
        from sklearn.metrics import f1_score
        # a fixed 2500-clip reference set keeps this cheap and comparable across epochs
        sub = trn_idx[np.random.RandomState(1).permutation(len(trn_idx))[:2500]]
        Ev, Et = full_embeddings(val_idx), full_embeddings(sub)
        yt = [labels[i] for i in val_idx]
        yp = [labels[sub[j]] for j in (Ev @ Et.T).argmax(1)]
        return f1_score(yt, yp, average="macro", labels=known + ["unknown"], zero_division=0)

    hr("3. BASELINE BEFORE ANY TRAINING")
    best = val_macro_f1()
    print("pretrained encoder, same protocol: macro-F1 = %.4f" % best, flush=True)
    torch.save(enc.state_dict(), OUT / "encoder_best.pt")
    json.dump({"epoch": 0, "val_macro_f1": float(best)}, open(OUT / "best.json", "w"))

    hr("4. FINE-TUNE")
    loader = DataLoader(Clips([files[i] for i in trn_idx], list(target[trn_idx]), audio_dir, True),
                        batch_size=BATCH, shuffle=True, num_workers=2,
                        collate_fn=collate, drop_last=True, persistent_workers=True)
    steps = EPOCHS * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_ENC, LR_HEAD], total_steps=steps, pct_start=0.15)
    ce = torch.nn.CrossEntropyLoss()
    step = 0
    for ep in range(1, EPOCHS + 1):
        # margin warmup: full AAM margin from step one tends to diverge
        m = MARGIN_MAX * min(1.0, ep / WARMUP_EP)
        t0, tot, hit, n = time.time(), 0.0, 0, 0
        for wav, y in loader:
            wav, y = wav.to(device), y.to(device)
            x = embed_batch(wav, True)
            logits = head(x, y, m)
            loss = ce(logits, y)
            with torch.no_grad():
                plain = head(x, y, 0.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step(); sched.step(); step += 1
            tot += loss.item() * len(y); hit += (plain.argmax(1) == y).sum().item(); n += len(y)
        line = "  ep %2d/%d  margin %.2f  loss %.3f  train-acc %.3f  %.1f min" % (
            ep, EPOCHS, m, tot / n, hit / n, (time.time() - t0) / 60)
        if ep % VAL_EVERY == 0 or ep == EPOCHS:
            f1 = val_macro_f1()
            line += "  | val macro-F1 %.4f" % f1
            if f1 > best:
                best = f1
                torch.save(enc.state_dict(), OUT / "encoder_best.pt")
                json.dump({"epoch": ep, "val_macro_f1": float(f1)}, open(OUT / "best.json", "w"))
                line += "  <- best, saved"
        print(line, flush=True)

    hr("5. RESULT")
    b = json.load(open(OUT / "best.json"))
    print("best checkpoint: epoch %d, val macro-F1 %.4f" % (b["epoch"], b["val_macro_f1"]))
    if b["epoch"] == 0:
        print("fine-tuning did not beat the pretrained encoder -- keeping the original.")
        print("that is a real answer: this corpus may be too small per speaker to adapt on.")
    print("saved: encoder_best.pt")
    hr("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

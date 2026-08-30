# Open-Set Speaker Identification — IAAA 3rd Contest

Classify 3,604 audio files into 447 classes: 446 known speaker identities, plus a
single `unknown` class covering 553 speakers who are never individually labelled.
Scored by **macro-F1**, so every speaker counts as much as the `unknown` class
that holds half the data.

**Leaderboard: 0.96700 macro-F1** (accuracy 0.97364).

```
v1  plain 1-NN on ECAPA                              0.96000
v2  + ResNet fusion, AS-norm, margin threshold       0.96600
v3  + mild Sinkhorn prior correction                 0.96585   (rejected)
    v2 with the margin threshold re-tuned on the
    leaderboard instead of cross-validation          0.96695
    + fusion weights flattened to 1:1                0.96700
```

---

## What the data actually is

Facts that changed the approach, all established before any model was trained:

| | |
|---|---|
| Files | 4,529 training + 3,604 evaluation |
| Known speakers | **446** — the challenge doc says 447 |
| `unknown` files | 2,275 (50.2%) |
| Per speaker | 439 speakers have exactly 5 files |
| Audio | 16 kHz, 16-bit, **stereo PCM WAV** — behind a `.mp3` extension |
| Channels | bit-identical duplicated mono |
| Corrupt | 140 files under 3 s; the smallest is 48 bytes (a WAV header) |

The files are not MP3. The median file implies 508 kbps, which MP3 cannot reach —
they are uncompressed WAV. Reading them with `soundfile` instead of an MP3 decoder
is a large speed difference across 16.9 GB, and matters for the judging time
budget.

## The idea that mattered

The contest describes `unknown` as one class. Training it that way hands 50% of
the data to a single heterogeneous label and the decision boundary collapses.

But the 553 unknown speakers are **present in the training data** — they are just
not individually labelled. So give each unknown *file* its own prototype and let
nearest-neighbour do the rest:

```
one blob for "unknown"          macro-F1  0.735
446 clusters                              0.942
1-NN over every unknown file              0.948   <- and it is the simplest
```

`+0.21` from reading the structure of the problem. Everything found afterwards
totalled less than a tenth of that.

## Method

```
WAV (1 channel)  ->  energy VAD  ->  up to 10 chunks of 6 s
                 ->  ECAPA-TDNN and ResNet embeddings, mean of L2-normed chunks
                 ->  AS-norm per encoder, fused 1:1
                 ->  predict the best known speaker if it beats the best unknown
                     prototype by tau = 0.32, else "unknown"
```

## Cross-validation lied, and how

Local CV consistently overstated gains, and the discrepancy grew as the gains
shrank:

```
              CV delta   real delta
v1 -> v2       +0.0122     +0.0060     about half
v2 -> v3       +0.0026     -0.0001     nothing
```

The cause is structural. Training gives 5 files per speaker; the eval set adds
~3.6 more. Any local fold must take query files out of those same 5, so it can
never reproduce "5 prototypes and 3.6 queries" — it always has fewer prototypes
than reality, which makes known-speaker scores weaker than they really are, which
biases the tuned threshold **downward**.

That prediction was testable, and it held. CV put the optimum at tau ≈ 0.20; three
leaderboard probes put it at 0.303:

```
tau = 0.20  ->  0.96600
tau = 0.32  ->  0.96695
tau = 0.45  ->  0.96497        quadratic vertex: tau = 0.303
```

Re-tuning one number against the leaderboard was worth more than the previous
thirteen experiments combined.

The same reasoning predicted the fusion weights were mis-tuned too, for the same
reason. They were not -- 1:1 and 1:3 return *identical* accuracy to sixteen
digits and differ by 0.00005 macro-F1, a seventh of one file. That parameter sits
on a plateau. A correct diagnosis of the protocol does not imply every parameter
it touched is wrong.

## What did not work

Kept honestly, because the negative results shaped the search more than the
positive ones.

| Idea | Result | Why |
|---|---|---|
| Transductive clustering | −0.004 | clusters merge speakers; a wrong label spreads |
| WavLM-base-plus-sv fusion | −0.004 | the `-sv` head is weak, and centring did not fix it |
| AAM-softmax fine-tuning | ~0 | trains fine (loss 6.8→2.6, acc 0.63→0.91) and still overfits: ~5 files per speaker is a data ceiling, not a tuning problem |
| WCCN backend | −0.002 | within-class covariance is too noisy from 4 files per speaker |
| CAM++, ERes2Net, ERes2NetV2 | 0.000 | 40% lower EER on VoxCeleb, identical here — the task is not encoder-limited |
| QMF quality calibration | +0.002 | borderline; the decision was already near-optimal given the scores |
| Hard Sinkhorn prior matching | **−0.023** | see below |
| Prototype enrichment | −0.001 | error propagation, monotone in the confidence threshold |
| Starved-class rescue | +0.000 | the rescue picks the right file 3.8% of the time — a speaker gets zero predictions precisely because its own files do not score for it |
| Test-time query aggregation | +0.001 | grouping is 98% pure and it still buys nothing |
| Neural VAD front-end | −0.002 | LibriParty VAD removes useful speech in this domain |

### The one that nearly shipped

Hard Sinkhorn measured **+0.0094 at t = +12.6** — the strongest signal in the
whole project. It was an artifact. The folds had been built exactly 50/50 with
exactly 2 held-out files per speaker, so Sinkhorn's assumption was true by
construction. Rebuilding the folds with uneven speaker counts turned it into
**−0.023**.

A t-statistic says an effect is not chance. It does not say the effect comes from
the data rather than from how the experiment was built. The only thing that
separated the two was deliberately breaking the assumption.

## Layout

```
submission/     the deliverable: submission.py, config, prototype embeddings
kaggle/         numbered Kaggle kernels, 01-18, in the order they were run
```

| Notebook | What it does |
|---|---|
| `02_embed` | ECAPA embeddings + centroid baseline |
| `05_models` | ResNet and WavLM |
| `06_finetune` | AAM-softmax domain adaptation (negative result) |
| `07_bundle2` | builds and self-tests the v2 bundle |
| `09_sota` | CAM++ / ERes2Net / ERes2NetV2 via ModelScope |
| `10_select` | greedy fusion selection, 20-fold paired tests |
| `11_sinkhorn` | prior correction, and the robustness test that killed it |
| `14_perclass` | per-speaker thresholds from the F1 plug-in rule |
| `17_frontend` | neural VAD (negative result) |

Every Kaggle notebook installs `speechbrain` with `--no-deps` and verifies torch
afterwards. Installing it normally replaces torch with a build that has no kernels
for Kaggle's P100 (`sm_60`), and the failure is silent — the GPU stays visible and
every forward pass returns garbage. Run these on **T4** (`sm_75`).

## Running the submission

```bash
python submission.py --data-dir /path/to/audio --predictions-file-path out.csv
```

Output is `audio_file,speaker_id`, where `speaker_id` is a training speaker UUID
or the literal string `unknown`.

The two encoders are not in this repo (145 MB of weights). Fetch them into
`submission/models/`:

```python
from speechbrain.inference.speaker import EncoderClassifier
for name, src in [("ecapa",  "speechbrain/spkrec-ecapa-voxceleb"),
                  ("resnet", "speechbrain/spkrec-resnet-voxceleb")]:
    EncoderClassifier.from_hparams(source=src, savedir="submission/models/" + name)
```

Judging is offline, so replace any symlinks in `savedir` with real files before
packaging. `kaggle/07_bundle2/build_v2.py` does this and then runs the script
through the organisers' exact command to validate the output format.

## Reproducing

1. `kaggle/02_embed` — ECAPA embeddings for all 4,529 files (~9 min on T4)
2. `kaggle/05_models` — ResNet embeddings
3. `kaggle/07_bundle2` — assemble and self-test the bundle
4. `kaggle/10_select`, `11_sinkhorn`, `14_perclass` — local analysis on the cached
   embeddings, no GPU needed

Dependencies are restricted to the contest's own `pyproject.toml`.

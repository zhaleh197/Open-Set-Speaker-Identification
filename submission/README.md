# Deliverable

```bash
python submission.py --data-dir <audio dir> --predictions-file-path out.csv
```

Writes `audio_file,speaker_id`, one row per input file. `speaker_id` is a training
speaker UUID or the literal string `unknown`.

## Missing piece: the encoder weights

`models/` is not in the repository (145 MB). Fetch it once:

```python
from speechbrain.inference.speaker import EncoderClassifier
for name, src in [("ecapa",  "speechbrain/spkrec-ecapa-voxceleb"),
                  ("resnet", "speechbrain/spkrec-resnet-voxceleb")]:
    EncoderClassifier.from_hparams(source=src, savedir="models/" + name)
```

speechbrain may fill `savedir` with symlinks into the HuggingFace cache. Judging
runs offline, so resolve them to real files before packaging.

## Files

| | |
|---|---|
| `submission.py` | inference: read, VAD, chunk, embed, AS-norm, fuse, decide |
| `config.json` | tau, fusion weights, per-prototype AS-norm cohort statistics |
| `train_emb_ecapa.npy` | 4,408 x 192 prototypes |
| `train_emb_resnet.npy` | 4,408 x 256 prototypes |
| `train_labels.json` | the speaker UUID for each prototype |

`tau = 0.32` is the margin a known speaker must beat the best unknown prototype
by. It was tuned against the leaderboard, not cross-validation — CV put it at
0.20 and was wrong for a structural reason explained in the top-level README.

## Runtime

Roughly 15-20 minutes for 3,604 files on a T4. Requires a GPU whose compute
capability the installed torch supports: Kaggle's P100 is `sm_60`, which the
`cu128` wheels dropped, and the failure is silent. The script warns and falls
back to CPU rather than returning garbage.

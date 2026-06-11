# Modernized environment & timm/HuggingFace integration

This documents the 2026 modernization: upgraded dependency stack (Python 3.11,
PyTorch 2.10/cu126), removal of unmaintained deps (pytorchvideo vendored,
detectron2 made demo-only), and new model wrappers for timm and HuggingFace
video models. The legacy environment remains documented by `slowfast_env.yml`
and is still what historic results were produced with.

## Environment

```bash
pip install -r requirements_modern.txt
pip install -e . --no-deps
```

Or build the container for the cluster:

```bash
apptainer build slowfast_modern.sif containers/slowfast_modern.def
```

Job scripts work as before (`singularity exec --nv ... python ./tools/run_net.py`).
The image needs no `module load cuda/...`: the cu126 torch wheels bundle the
CUDA runtime and NCCL, and run on any host driver supporting CUDA 12.x (>=525).
Do not switch to cu130 wheels without confirming the cluster driver is >=580.

## Running

Entry point and config system are unchanged from upstream SlowFast
(see `GETTING_STARTED.md`); any config key can be overridden on the
command line after `--cfg`:

```bash
# train (timm / HF examples; native configs work the same)
python tools/run_net.py --cfg configs/timm/CONVNEXT_T_MEANPOOL_SPECIES.yaml
python tools/run_net.py --cfg configs/hf/VIDEOMAE_B_SPECIES.yaml NUM_GPUS 1 TRAIN.BATCH_SIZE 4

# test / no-label inference with CSV export
python tools/run_net.py --cfg configs/species_classification/test/r50/SLOW_8x8_R50_DISJOINT_F=32_NO-LABELS_TEST.yaml \
    TRAIN.ENABLE False TEST.ENABLE True TEST.NO_LABELS True \
    TEST.CHECKPOINT_FILE_PATH /path/to/checkpoint.pyth \
    TEST.SAVE_PROBS_CSV_PATH probs.csv
```

On the cluster, job scripts keep the existing pattern (`jobs/*.sh`) with two
changes: point at the new image and drop `module load cuda/12.3`:

```bash
singularity exec --nv --bind /lfs1i3/home/b35u/obrookes.b35u:/mnt ~/slowfast_modern.sif \
    python -W ignore ./tools/run_net.py --cfg '/mnt/imslowfast/configs/....yaml'
```

Key pins and why (verified against live indexes, 2026-06):

| Package | Pin | Why |
|---|---|---|
| torch / torchvision | 2.10.* / 0.25.* (cu126) | mature release with patches; cu126 wheels safe on the hopper partition |
| transformers | >=4.53,<5 | v4 LTS branch; v5 has weekly breaking changes and a dtype-loading change. 4.53 adds V-JEPA 2 |
| timm | >=1.0.15 | stable 1.x line |
| av | 17.1.0 | wheels bundle FFmpeg (no conda ffmpeg needed); decode path verified |
| numpy | 1.26.4 | last 1.x; a numpy 2 trial is a follow-up, don't bump casually |
| fvcore | 0.1.5.post20221221 | same release the legacy env used; pure Python |
| detectron2 | removed | only the demo person detector needs it: `pip install -e .[demo]` |
| pytorchvideo | removed | the 4 modules slowfast used are vendored in `slowfast/vendor/pytorchvideo/` |

## What changed in the code

- `slowfast/vendor/pytorchvideo/`: vendored `distributed`, `batch_norm`
  (NaiveSyncBatchNorm), `swish`, `soft_target_cross_entropy` (+ helpers);
  import sites in `slowfast/utils/distributed.py`, `models/batchnorm_helper.py`,
  `models/operators.py`, `models/losses.py` updated.
- `slowfast/models/head_helper.py`: detectron2 `ROIAlign` -> `torchvision.ops.RoIAlign`
  (drop-in; `aligned` passed explicitly). detectron2 imports in
  `slowfast/visualization/` are now optional (clear error if demo paths used
  without it).
- `torch.load(..., weights_only=False)` in `slowfast/utils/checkpoint.py` and
  `tools/test_net.py`: required since torch 2.6 to load existing checkpoints
  (they contain cfg dumps/optimizer state).
- `torch.cuda.amp.*` -> `torch.amp.*("cuda", ...)` in `tools/train_net.py`;
  `torch.optim._multi_tensor.AdamW` -> `AdamW(..., foreach=True)`.
- **Decoding backend switched to pyav** (default in `slowfast/config/defaults.py`
  and all configs). torchvision's video decoding is deprecated since 0.22 and
  used a private API. The pyav path in `slowfast/datasets/decoder.py` was
  repaired in the process: it never worked after the multi-clip refactor
  (scalar/list mismatch, undefined `start_end_delta_time`, tensor-unsafe
  `None in` check) — this affects upstream SlowFast too.
- Optimizer layer-decay (`SOLVER.LAYER_DECAY < 1.0`) generalized: strips a
  `backbone.` prefix and uses the model's `get_num_layers()` when available;
  native MViT behavior unchanged.

## New models

Both follow the existing model contract (single-pathway `[B, C, T, H, W]` in,
logits out) and work with the unmodified train/test loops, FGBG configs aside.

**`TimmVideoModel`** (`slowfast/models/timm_model_builder.py`): any timm
backbone applied per-frame + temporal mean/attention pooling. See
`configs/timm/CONVNEXT_T_MEANPOOL_SPECIES.yaml` and
`VIT_B16_ATTNPOOL_SPECIES.yaml` (the latter exercises layer-decay). Set
`DATA.MEAN/STD` to the backbone's pretraining stats — the model logs a warning
on mismatch.

**`HFVideoModel`** (`slowfast/models/hf_model_builder.py`): any
`AutoModelForVideoClassification` Hub model (VideoMAE, TimeSformer, ViViT,
V-JEPA 2). See `configs/hf/*.yaml`; DATA values there match each model's
processor. Leave `TRAIN.CHECKPOINT_FILE_PATH` empty — pretrained weights come
from the Hub and the slowfast checkpoint loader will not touch them; saving /
resuming slowfast checkpoints works as usual. Frame-count and crop-size
mismatches raise at init.

## Verification status

Verified locally (CPU, torch 2.11/Python 3.13/numpy 2.4 — a *newer* stack than
the target, so version-sensitive code got a harsher test):

- imports of `slowfast.models` / `datasets` / `utils.distributed` (vendored modules)
- dummy forwards: native ResNet + MViT from species configs, TimmVideoModel
  (convnext, ViT+attention pool), HFVideoModel (VideoMAE, from config)
- optimizer param groups: MViT layer-decay regression, timm ViT layer-decay
  (correct decay range), HF/convnext standard grouping
- config parsing of all species/timm/hf configs against the new defaults
- pyav decode through `decoder.decode()` and a full `Kinetics.__getitem__`
  (train + test sampling) on synthetic video, under av 17.1

Remaining gates — run on the cluster (GPU) before long jobs:

1. `tools/test_net.py` with a pre-upgrade checkpoint (exercises the
   `weights_only` fix end-to-end) and a `TEST.NO_LABELS: True` run -> CSV
   output compared against the legacy environment's output on the same videos.
2. Decode equivalence on ~20 real videos: pyav (new env) vs torchvision
   (legacy env) sampled frames / predictions. Small differences are expected
   from seek behavior; characterize before re-running production inference.
3. Short real train of an R50/MViT species config; loss trajectory sanity vs
   legacy env.
4. FGBG smoke: a few iterations of an `MVIT_B_16x4_CONCAT*_BG_SUB` config.
5. Multi-GPU run with `BN.NORM_TYPE: sync_batchnorm` (vendored NaiveSyncBatchNorm).
6. timm + HF fine-tune smokes (1 epoch); for HF, initial loss should be well
   below random-init if pretrained weights loaded.

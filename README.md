# PGC DINOv3 Finetuning

This repo trains/evaluates PGC with Hugging Face datasets only. There is no
local folder conversion path and no filesystem image-folder training reader.

## Dataset

Use either:

- `--hf_dataset_path`: a local Hugging Face `Dataset` / `DatasetDict` saved with
  `save_to_disk()`.
- `--hf_dataset_repo`: a Hugging Face Hub dataset repo id.

Required columns:

- `image`: image value decoded by Hugging Face datasets, a path, or bytes.
- `label`: binary label, `0=real` and `1=fake`.

Optional columns:

- `source`: subset name used for per-subset evaluation metrics.

Expected splits:

- `train` for training.
- `val` for evaluation, unless overridden with `--hf_eval_split`.

## DINO Weights

Place local DINO checkpoints under `pretrained_dino/`, for example:

```text
pretrained_dino/
  dinov3-large/
    config.json
    model.safetensors
```

## Train

Single process:

```bash
python run_pipeline.py \
  --hf_dataset_path <DATA_ROOT>/hf_real_fake__version_6 \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --devices 0 \
  --name pgc_dinov3_large
```

DDP:

```bash
python run_pipeline.py \
  --hf_dataset_path <DATA_ROOT>/hf_real_fake__version_6 \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --devices 0,1 \
  --nproc_per_node 2 \
  --name pgc_dinov3_large_ddp
```

Direct `torchrun`:

```bash
torchrun --nproc_per_node=2 train.py \
  --name pgc_dinov3_large_ddp \
  --checkpoints_dir checkpoints \
  --devices 0,1 \
  --batch_size 16 \
  --dino_variant dinov3-large \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --hf_dataset_path <DATA_ROOT>/hf_real_fake__version_6 \
  --hf_train_split train \
  --hf_eval_split val \
  --cropSize 224 \
  --niter 100 \
  --lr 5e-5 \
  --optim adam \
  --accumulation_steps 4 \
  --weight_decay 0.05 \
  --label_smoothing 0.1 \
  --eval_every_steps 200
```

Hub dataset:

```bash
python run_pipeline.py \
  --hf_dataset_repo <HF_USER_OR_ORG>/<HF_DATASET_NAME> \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --devices 0,1 \
  --nproc_per_node 2
```

## Evaluate

```bash
python test.py \
  --name test_hf \
  --checkpoint checkpoints/pgc_dinov3_large_ddp/best.pth \
  --hf_dataset_path <DATA_ROOT>/hf_real_fake__version_6 \
  --hf_eval_split val \
  --dino_variant dinov3-large \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --devices 0 \
  --batch_size 32
```

## Notes

- `--batch_size` is per process/GPU under DDP.
- Only rank 0 logs, evaluates, and saves checkpoints during DDP training.
- Checkpoints are saved without `module.` prefixes so `test.py` can load them
  directly.

# PGC DINOv3 Finetuning

This repo trains/evaluates PGC with Hugging Face datasets only. There is no
local folder conversion path and no filesystem image-folder training reader.

## Dataset

Use one of these sources:

- `--dataset_root`: local folder containing `real/` and `fake/` subfolders.
  The pipeline converts it to a local HF dataset before training.
- `--hf_dataset_path`: a local Hugging Face `Dataset` / `DatasetDict` saved with
  `save_to_disk()`.
- `--hf_dataset_repo`: a Hugging Face Hub dataset repo id.
- `--dataset_manifest`: a local CSV/JSONL/JSON/Parquet manifest that will be
  converted to `--hf_dataset_path` before training.

Required columns:

- `image`: image value decoded by Hugging Face datasets, a path, or bytes.
- `label`: binary label, `0=real` and `1=fake`.

Optional columns:

- `source`: subset name used for per-subset evaluation metrics.

Expected splits:

- `train` for training.
- `val` for evaluation, unless overridden with `--hf_eval_split`.

Manifest example:

```csv
image,label,source,split
images/real_0001.jpg,0,real_camera,train
images/fake_0001.jpg,1,sdxl,train
images/real_1001.jpg,real,real_camera,val
images/fake_1001.jpg,fake,sdxl,val
```

If the manifest has no `split` column, `data/create_dataset.py` creates a
stratified train/val split with `--val_ratio`.

Folder example:

```text
<DATA_ROOT>/
  real/
    image_0001.jpg
    image_0002.jpg
  fake/
    image_0001.jpg
    image_0002.jpg
```

## DINO Weights

Place local DINO checkpoints under `pretrained_dino/`, for example:

```text
pretrained_dino/
  dinov3-large/
    config.json
    model.safetensors
```

## Train

Create a local HF dataset from `real/` and `fake/`, then train:

```bash
python run_pipeline.py \
  --dataset_root <DATA_ROOT> \
  --hf_dataset_path <DATA_ROOT>/hf_real_fake__version_6 \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --devices 0 \
  --name pgc_dinov3_large
```

Create a local HF dataset from a manifest, then train:

```bash
python run_pipeline.py \
  --dataset_manifest <DATA_ROOT>/manifest.csv \
  --image_base_dir <DATA_ROOT> \
  --hf_dataset_path <DATA_ROOT>/hf_real_fake__version_6 \
  --dino_pretrained_root <PROJECT_ROOT>/pretrained_dino \
  --devices 0 \
  --name pgc_dinov3_large
```

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

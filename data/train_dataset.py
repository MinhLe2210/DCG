import argparse
import logging
import os
from collections import Counter
from typing import Any, Dict

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

from .hf_dataset_utils import decode_rgb_image, load_hf_split
from .transforms import MEAN, STD, create_train_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class RealFakeDataset(Dataset):
    """
    PGC training dataset backed by a Hugging Face Dataset split.

    The class name intentionally remains ``RealFakeDataset`` so the existing
    ``data/dataloader.py`` does not need to change.

    Runtime output remains compatible with PGCTrainer:
      {
          "image":  Tensor[6, H, W],  # RGB || residual
          "label":  int,
          "source": str,
      }

    No forced 384x384 resize is introduced. The existing PGC transform uses
    PadRandomCrop(opt.cropSize), normally 224.
    """

    def __init__(self, opt: argparse.Namespace):
        self.opt = opt

        split = (
            getattr(opt, "hf_train_split", None)
            or os.environ.get("HF_TRAIN_SPLIT")
            or "train"
        )
        dataset_path = getattr(opt, "hf_dataset_path", None)
        dataset_repo = getattr(opt, "hf_dataset_repo", None)

        self.dataset = load_hf_split(
            split=split,
            dataset_path=dataset_path,
            dataset_repo=dataset_repo,
        )

        required_columns = {"image", "label"}
        missing = required_columns.difference(self.dataset.column_names)
        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}. "
                f"Available columns: {self.dataset.column_names}"
            )

        labels = [int(label) for label in self.dataset["label"]]
        counts = Counter(labels)

        logger.info("=" * 60)
        logger.info("HF TRAIN DATASET CLASS DISTRIBUTION:")
        logger.info(" Split: %s", split)
        logger.info(" Total samples: %d", len(labels))
        logger.info(" Real (label=0): %d", counts.get(0, 0))
        logger.info(" Fake (label=1): %d", counts.get(1, 0))
        logger.info("=" * 60)

        self.transform = create_train_transforms(
            image_size=opt.cropSize,
            mean=MEAN["dino"],
            std=STD["dino"],
            is_crop=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            sample = self.dataset[int(idx)]
            image = decode_rgb_image(sample["image"])
            label = int(sample["label"])
            source = str(sample.get("source", "hf_train"))
        except Exception as exc:  # intentional corrupt-image fallback
            logger.warning("Failed to load HF train sample index=%s: %s", idx, exc)
            image = Image.new("RGB", (self.opt.cropSize, self.opt.cropSize))
            label = 0
            source = "hf_train_corrupt_fallback"

        image_tensor: torch.Tensor = self.transform(image)

        return {
            "image": image_tensor,
            "label": label,
            "source": source,
        }
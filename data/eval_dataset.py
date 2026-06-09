import logging
from typing import Dict, List, Optional

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .hf_dataset_utils import decode_rgb_image, load_hf_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class HFRealFakeEvalDataset(Dataset):
    """
    Evaluation reader for a Hugging Face Dataset/DatasetDict split.

    The expected columns are:
      - image: PIL image / path / bytes-like HF image value
      - label: binary label where 0=real and 1=fake

    If a ``source`` column exists, it is used as subset name for per-subset
    metrics. Otherwise the whole split is exposed as one subset.
    """

    def __init__(
        self,
        transform=None,
        split: str = "val",
        subset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        dataset_repo: Optional[str] = None,
    ):
        self.transform = transform
        self.split = split
        self.default_subset_name = subset_name or split

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

        self.subset_indices: Dict[str, List[int]] = {}
        has_source = "source" in self.dataset.column_names
        if has_source:
            sources = self.dataset["source"]
            for idx, source in enumerate(sources):
                subset = str(source or self.default_subset_name)
                self.subset_indices.setdefault(subset, []).append(idx)
        else:
            self.subset_indices[self.default_subset_name] = list(
                range(len(self.dataset))
            )

        logger.info(
            "Loaded %d images from HF split=%s across %d subset(s)",
            len(self.dataset),
            self.split,
            len(self.subset_indices),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        try:
            sample = self.dataset[int(idx)]
            image = decode_rgb_image(sample["image"])
            label = int(sample["label"])
            subset = str(sample.get("source", self.default_subset_name))
        except Exception as exc:
            logger.warning("Failed to load HF eval sample index=%s: %s", idx, exc)
            image = Image.new("RGB", (224, 224))
            label = 0
            subset = self.default_subset_name

        if self.transform is not None:
            image = self.transform(image)

        return image, label, subset

    def get_subset_names(self) -> List[str]:
        return list(self.subset_indices.keys())

    def get_subset_indices(self, subset_name: str) -> List[int]:
        return self.subset_indices.get(subset_name, [])


__all__ = ["HFRealFakeEvalDataset"]

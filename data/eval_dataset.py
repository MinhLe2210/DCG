import logging
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .hf_dataset_utils import decode_rgb_image, load_hf_split
from .transforms import get_list

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class _PerSubsetEvalDataset(Dataset):
    """
    Original filesystem benchmark reader.

    Keep this path for UniversalFakeDetect-style test sets:
        root/
        ├── subset_a/
        │   ├── 0_real/
        │   └── 1_fake/
        └── subset_b/
            ├── 0_real/
            └── 1_fake/
    """

    REAL_DIR_NAME: str = ""
    FAKE_DIR_NAME: str = ""
    BENCHMARK_NAME: str = "Eval"

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Each entry: (path, label, subset_name)
        self.samples: List[Tuple[str, int, str]] = []
        self.subset_indices: Dict[str, List[int]] = {}

        subsets = [
            dirname
            for dirname in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, dirname))
        ]

        for subset in subsets:
            subset_path = os.path.join(root_dir, subset)
            real_list = get_list(subset_path, self.REAL_DIR_NAME)
            fake_list = get_list(subset_path, self.FAKE_DIR_NAME)

            logger.info(
                "Subset '%s': %d real images (%s), %d fake images (%s)",
                subset,
                len(real_list),
                self.REAL_DIR_NAME,
                len(fake_list),
                self.FAKE_DIR_NAME,
            )

            for path in real_list:
                self.samples.append((path, 0, subset))

            for path in fake_list:
                self.samples.append((path, 1, subset))

        self.samples.sort(key=lambda item: item[0])

        for idx, (_, _, subset) in enumerate(self.samples):
            self.subset_indices.setdefault(subset, []).append(idx)

        logger.info(
            "Loaded %d images from %d %s subsets",
            len(self.samples),
            len(subsets),
            self.BENCHMARK_NAME,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, subset = self.samples[idx]

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # intentional corrupt-image fallback
            logger.warning("Failed to load image %s: %s", path, exc)
            image = Image.new("RGB", (224, 224))

        if self.transform is not None:
            image = self.transform(image)

        return image, label, subset

    def get_subset_names(self) -> List[str]:
        return list(self.subset_indices.keys())

    def get_subset_indices(self, subset_name: str) -> List[int]:
        return self.subset_indices.get(subset_name, [])


class UniversalFakeDetectDataset(_PerSubsetEvalDataset):
    """
    Original UniversalFakeDetect benchmark reader used by train.py and test.py.
    """

    REAL_DIR_NAME = "0_real"
    FAKE_DIR_NAME = "1_fake"
    BENCHMARK_NAME = "UniversalFakeDetect"


class HFRealFakeEvalDataset(Dataset):
    """
    Optional validation reader for the ``val`` split stored in a Hugging Face
    DatasetDict.

    It implements get_subset_names() and get_subset_indices(), matching the
    contract expected by engine/evaluator.py. The complete validation split is
    exposed as one subset by default.
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
        self.subset_name = subset_name or split

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

        self.subset_indices: Dict[str, List[int]] = {
            self.subset_name: list(range(len(self.dataset)))
        }

        logger.info(
            "Loaded %d images from HF split=%s as evaluation subset=%s",
            len(self.dataset),
            self.split,
            self.subset_name,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        try:
            sample = self.dataset[int(idx)]
            image = decode_rgb_image(sample["image"])
            label = int(sample["label"])
        except Exception as exc:  # intentional corrupt-image fallback
            logger.warning("Failed to load HF eval sample index=%s: %s", idx, exc)
            image = Image.new("RGB", (224, 224))
            label = 0

        if self.transform is not None:
            image = self.transform(image)

        return image, label, self.subset_name

    def get_subset_names(self) -> List[str]:
        return list(self.subset_indices.keys())

    def get_subset_indices(self, subset_name: str) -> List[int]:
        return self.subset_indices.get(subset_name, [])


__all__ = [
    "UniversalFakeDetectDataset",
    "HFRealFakeEvalDataset",
]

DatasetType = Optional[_PerSubsetEvalDataset]
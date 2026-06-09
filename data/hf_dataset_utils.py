import os
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset, load_from_disk
from PIL import Image

DEFAULT_HF_DATASET_PATH = "../data/hf_real_fake__version_6"


def load_hf_split(
    split: str,
    dataset_path: Optional[str] = None,
    dataset_repo: Optional[str] = None,
) -> HFDataset:
    """
    Read one split from either:
      - a local Hugging Face Dataset/DatasetDict created with save_to_disk(), or
      - an optional Hugging Face Hub dataset repository.

    Environment fallbacks:
      HF_DATASET_PATH
      HF_DATASET_REPO
    """
    repo = dataset_repo or os.environ.get("HF_DATASET_REPO")
    path = (
        dataset_path
        or os.environ.get("HF_DATASET_PATH")
        or DEFAULT_HF_DATASET_PATH
    )

    if repo:
        loaded: Union[HFDataset, DatasetDict] = load_dataset(
            repo,
            keep_in_memory=False,
        )
    else:
        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Hugging Face dataset path does not exist: {resolved}\n"
                "Set HF_DATASET_PATH=/absolute/path/to/dataset or pass "
                "--hf_dataset_repo to read from the Hugging Face Hub."
            )

        loaded = load_from_disk(
            str(resolved),
            keep_in_memory=False,
        )

    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise KeyError(
                f"Split {split!r} was not found. "
                f"Available splits: {list(loaded.keys())}"
            )
        return loaded[split]

    if isinstance(loaded, HFDataset):
        return loaded

    raise TypeError(
        "Expected datasets.Dataset or datasets.DatasetDict, "
        f"got {type(loaded).__name__}"
    )


def decode_rgb_image(image_value: Any) -> Image.Image:
    """
    HF datasets normally decode an Image feature to PIL.Image.Image.
    Path and {'path', 'bytes'} variants are accepted defensively.
    """
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, (str, os.PathLike)):
        return Image.open(image_value).convert("RGB")

    if isinstance(image_value, dict):
        path = image_value.get("path")
        if path:
            return Image.open(path).convert("RGB")

        raw_bytes = image_value.get("bytes")
        if raw_bytes:
            return Image.open(BytesIO(raw_bytes)).convert("RGB")

    raise TypeError(
        "Unsupported HF image value type: "
        f"{type(image_value).__name__}"
    )

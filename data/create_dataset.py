import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import ClassLabel, Dataset, DatasetDict, Image, load_dataset


LABEL_MAPPING = {
    "0": 0,
    "1": 1,
    "real": 0,
    "fake": 1,
    "human": 0,
    "ai": 1,
    "generated": 1,
    "synthetic": 1,
}

TRAIN_SPLITS = {"train", "training"}
VAL_SPLITS = {"val", "valid", "validation", "eval", "test"}
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def _iter_image_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _load_real_fake_folders(dataset_root: str) -> Dataset:
    root = Path(dataset_root).expanduser().resolve()
    real_dir = root / "real"
    fake_dir = root / "fake"

    missing = [str(path) for path in [real_dir, fake_dir] if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Expected dataset root to contain 'real' and 'fake' folders. "
            f"Missing: {missing}"
        )

    rows = []
    for image_path in _iter_image_files(real_dir):
        rows.append(
            {
                "image": str(image_path),
                "label": 0,
                "source": "real",
            }
        )
    for image_path in _iter_image_files(fake_dir):
        rows.append(
            {
                "image": str(image_path),
                "label": 1,
                "source": "fake",
            }
        )

    if not rows:
        raise RuntimeError(f"No images found under {real_dir} or {fake_dir}")

    return Dataset.from_list(rows)


def _load_manifest(path: str) -> Dataset:
    manifest = Path(path).expanduser()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest}")

    suffix = manifest.suffix.lower()
    if suffix == ".csv":
        dataset = load_dataset("csv", data_files=str(manifest), split="train")
    elif suffix in {".json", ".jsonl"}:
        dataset = load_dataset("json", data_files=str(manifest), split="train")
    elif suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=str(manifest), split="train")
    else:
        raise ValueError(
            "Unsupported manifest extension. Use .csv, .json, .jsonl, or .parquet."
        )

    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected datasets.Dataset, got {type(dataset).__name__}")
    return dataset


def _normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        label = int(value)
        if label in {0, 1}:
            return label

    text = str(value).strip().lower()
    if text in LABEL_MAPPING:
        return LABEL_MAPPING[text]

    raise ValueError(
        f"Unsupported label {value!r}. Expected 0/1 or one of "
        f"{sorted(LABEL_MAPPING.keys())}."
    )


def _resolve_image_value(value: Any, image_base_dir: Optional[str]) -> Any:
    if not isinstance(value, str) or not image_base_dir:
        return value

    value_path = Path(value).expanduser()
    if value_path.is_absolute():
        return str(value_path)

    return str((Path(image_base_dir).expanduser() / value_path).resolve())


def _prepare_columns(
    dataset: Dataset,
    image_column: str,
    label_column: str,
    source_column: str,
    image_base_dir: Optional[str],
) -> Dataset:
    missing = {
        column
        for column in [image_column, label_column]
        if column not in dataset.column_names
    }
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Available columns: {dataset.column_names}"
        )

    if image_column != "image":
        dataset = dataset.rename_column(image_column, "image")
    if label_column != "label":
        dataset = dataset.rename_column(label_column, "label")
    if source_column in dataset.column_names and source_column != "source":
        dataset = dataset.rename_column(source_column, "source")

    def normalize_example(example: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "image": _resolve_image_value(example["image"], image_base_dir),
            "label": _normalize_label(example["label"]),
        }
        if "source" in example:
            out["source"] = str(example["source"])
        return out

    dataset = dataset.map(normalize_example, desc="Normalize image/label columns")
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", ClassLabel(names=["real", "fake"]))

    keep_columns = {"image", "label", "source"}
    drop_columns = [
        column for column in dataset.column_names if column not in keep_columns
    ]
    if drop_columns:
        dataset = dataset.remove_columns(drop_columns)

    return dataset


def _split_by_manifest_column(dataset: Dataset, split_column: str) -> DatasetDict:
    split_values = [str(value).strip().lower() for value in dataset[split_column]]
    train_indices = [
        idx for idx, value in enumerate(split_values) if value in TRAIN_SPLITS
    ]
    val_indices = [
        idx for idx, value in enumerate(split_values) if value in VAL_SPLITS
    ]

    if not train_indices or not val_indices:
        raise ValueError(
            f"Split column {split_column!r} must contain train and val/test rows."
        )

    split_dataset = DatasetDict(
        {
            "train": dataset.select(train_indices),
            "val": dataset.select(val_indices),
        }
    )
    return split_dataset.remove_columns([split_column])


def _print_distribution(dataset_dict: DatasetDict) -> None:
    for split_name, split_dataset in dataset_dict.items():
        labels = [int(label) for label in split_dataset["label"]]
        counts = Counter(labels)
        print(f"\n{split_name.upper()}")
        print(f"total: {len(labels)}")
        print(f"real label=0: {counts.get(0, 0)}")
        print(f"fake label=1: {counts.get(1, 0)}")


def create_hf_dataset(
    manifest: Optional[str],
    output_dir: str,
    dataset_root: Optional[str] = None,
    image_column: str = "image",
    label_column: str = "label",
    source_column: str = "source",
    split_column: str = "split",
    image_base_dir: Optional[str] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    if dataset_root:
        dataset = _load_real_fake_folders(dataset_root)
    elif manifest:
        dataset = _load_manifest(manifest)
    else:
        raise ValueError("Pass either --dataset_root or --manifest.")

    if split_column in dataset.column_names:
        split_dataset = _split_by_manifest_column(dataset, split_column)
        prepared = DatasetDict(
            {
                split: _prepare_columns(
                    split_data,
                    image_column=image_column,
                    label_column=label_column,
                    source_column=source_column,
                    image_base_dir=image_base_dir,
                )
                for split, split_data in split_dataset.items()
            }
        )
    else:
        prepared_full = _prepare_columns(
            dataset,
            image_column=image_column,
            label_column=label_column,
            source_column=source_column,
            image_base_dir=image_base_dir,
        )
        split = prepared_full.train_test_split(
            test_size=val_ratio,
            seed=seed,
            stratify_by_column="label",
        )
        prepared = DatasetDict({"train": split["train"], "val": split["test"]})

    output_path = Path(output_dir).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.save_to_disk(str(output_path))

    _print_distribution(prepared)
    print(f"\nSaved Hugging Face dataset to: {output_path}")

    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Create a local Hugging Face DatasetDict from data/real and "
            "data/fake folders, or from a manifest file."
        ),
    )
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Folder containing real/ and fake/ subfolders.",
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_column", default="image")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--source_column", default="source")
    parser.add_argument("--split_column", default="split")
    parser.add_argument("--image_base_dir", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_hf_dataset(
        manifest=args.manifest,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        image_column=args.image_column,
        label_column=args.label_column,
        source_column=args.source_column,
        split_column=args.split_column,
        image_base_dir=args.image_base_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

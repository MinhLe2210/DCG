import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

from datasets import ClassLabel, DatasetDict, load_dataset, load_from_disk


def print_distribution(dataset: DatasetDict) -> None:
    for split in dataset.keys():
        labels = [int(label) for label in dataset[split]["label"]]
        counts = Counter(labels)

        label_feature = dataset[split].features.get("label")
        label_names: Optional[list[str]] = getattr(label_feature, "names", None)

        print(f"\n{split.upper()}")
        print(f"Total images: {len(labels)}")

        if label_names:
            for index, name in enumerate(label_names):
                print(f"label {index} ({name}): {counts.get(index, 0)}")
        else:
            for index in sorted(counts):
                print(f"label {index}: {counts[index]}")

def normalize_binary_labels(full_dataset):
    """
    Force the PGC convention:
        real -> 0
        fake -> 1
    """
    label_feature = full_dataset.features.get("label")
    label_names = getattr(label_feature, "names", None)

    if not label_names:
        raise ValueError(
            "Could not read label names from ImageFolder dataset."
        )

    print("Original label names:", label_names)

    canonical_mapping = {
        "real": 0,
        "fake": 1,
        "0_real": 0,
        "1_fake": 1,
    }

    old_id_to_new_id = {}

    for old_id, raw_name in enumerate(label_names):
        class_name = raw_name.strip().lower()

        if class_name not in canonical_mapping:
            raise ValueError(
                f"Unsupported class folder: {raw_name!r}. "
                "Expected fake/ and real/."
            )

        old_id_to_new_id[old_id] = canonical_mapping[class_name]

    if set(old_id_to_new_id.values()) != {0, 1}:
        raise ValueError(
            f"Invalid binary class mapping: {old_id_to_new_id}"
        )

    print("Remap label IDs:", old_id_to_new_id)

    full_dataset = full_dataset.map(
        lambda example: {
            "label": old_id_to_new_id[int(example["label"])]
        },
        desc="Normalize labels: real=0, fake=1",
    )

    full_dataset = full_dataset.cast_column(
        "label",
        ClassLabel(names=["real", "fake"]),
    )

    print(
        "Normalized label names:",
        full_dataset.features["label"].names,
    )

    return full_dataset

def create_dataset(
    imagefolder_dir: str,
    output_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Convert an ImageFolder-style training directory into a Hugging Face
    DatasetDict and persist it with save_to_disk().

    Expected example:
        train/
        ├── 0_real/
        └── 1_fake/

    lazily by PGC during training using the existing PadRandomCrop pipeline.
    """
    dataset = load_dataset(
        "imagefolder",
        data_dir=imagefolder_dir,
        drop_labels=False,
        keep_in_memory=False,
    )

    full_dataset = dataset["train"]

    print("Features before normalization:", full_dataset.features)
    print("Example before normalization:", full_dataset[0])

    full_dataset = normalize_binary_labels(full_dataset)

    print("Features after normalization:", full_dataset.features)
    print("Example after normalization:", full_dataset[0])

    train_val = full_dataset.train_test_split(
        test_size=val_ratio,
        seed=seed,
        stratify_by_column="label",
    )

    output = DatasetDict(
        {
            "train": train_val["train"],
            "val": train_val["test"],
        }
    )

    output_path = Path(output_dir).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save_to_disk(str(output_path))

    print_distribution(output)
    print(f"\nSaved to {output_path}")

    return output


def load_dataset_local(data_path: str) -> DatasetDict:
    dataset = load_from_disk(data_path, keep_in_memory=False)

    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict from {data_path}, "
            f"got {type(dataset).__name__}"
        )

    print_distribution(dataset)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Hugging Face DatasetDict for PGC training without "
            "resizing source images."
        )
    )
    parser.add_argument(
        "--imagefolder_dir",
        default="./train",
        help="ImageFolder root, e.g. ./train with 0_real/ and 1_fake/.",
    )
    parser.add_argument(
        "--output_dir",
        default="../data/hf_real_fake__version_6",
        help="Output directory used by DatasetDict.save_to_disk().",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the stratified split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_dataset(
        imagefolder_dir=args.imagefolder_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
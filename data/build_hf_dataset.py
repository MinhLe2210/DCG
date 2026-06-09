import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

from datasets import DatasetDict, load_dataset, load_from_disk


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


def create_dataset(
    imagefolder_dir: str,
    output_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    hub_repo_id: Optional[str] = None,
) -> DatasetDict:
    """
    Build a Hugging Face DatasetDict with ``train`` and ``val`` splits.

    No resize is performed here. PGC applies PadRandomCrop/PadCenterCrop lazily
    while training or evaluating.
    """
    dataset = load_dataset(
        "imagefolder",
        data_dir=imagefolder_dir,
        drop_labels=False,
        keep_in_memory=False,
    )

    full_dataset = dataset["train"]

    print("Features:", full_dataset.features)
    print("Example:", full_dataset[0])

    split_dataset = full_dataset.train_test_split(
        test_size=val_ratio,
        seed=seed,
        stratify_by_column="label",
    )

    output = DatasetDict(
        {
            "train": split_dataset["train"],
            "val": split_dataset["test"],
        }
    )

    output_path = Path(output_dir).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save_to_disk(str(output_path))

    print_distribution(output)
    print(f"\nSaved to {output_path}")

    if hub_repo_id:
        output.push_to_hub(hub_repo_id)
        print(f"Pushed to Hugging Face Hub: {hub_repo_id}")

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
        description="Build the local Hugging Face dataset for PGC."
    )
    parser.add_argument("--imagefolder_dir", default="./train")
    parser.add_argument(
        "--output_dir",
        default="../data/hf_real_fake__version_6",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub_repo_id", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    create_dataset(
        imagefolder_dir=args.imagefolder_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        hub_repo_id=args.hub_repo_id,
    )
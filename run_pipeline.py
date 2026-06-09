import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print(" ".join(cmd))
    print("=" * 80 + "\n")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train PGC from a Hugging Face dataset.",
    )
    parser.add_argument(
        "--hf_dataset_path",
        default="data/hf_real_fake__version_6",
        help="Local Hugging Face Dataset/DatasetDict path from save_to_disk().",
    )
    parser.add_argument(
        "--hf_dataset_repo",
        default=None,
        help="Optional Hugging Face Hub dataset repo used directly by train.py.",
    )
    parser.add_argument(
        "--dataset_manifest",
        default=None,
        help=(
            "Optional local CSV/JSONL/JSON/Parquet manifest. When set, "
            "the pipeline creates --hf_dataset_path before training."
        ),
    )
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Optional folder containing real/ and fake/ subfolders.",
    )
    parser.add_argument("--image_column", default="image")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--source_column", default="source")
    parser.add_argument("--split_column", default="split")
    parser.add_argument(
        "--image_base_dir",
        default=None,
        help="Base directory used to resolve relative image paths in the manifest.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--name", default="pgc_dinov3_large")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--devices", default="0")
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="Use >1 to launch DDP through torch.distributed.run.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--dino_variant", default="dinov3-large")
    parser.add_argument(
        "--dino_pretrained_root",
        required=True,
        help="Root containing local DINO checkpoints, e.g. pretrained_dino/.",
    )
    parser.add_argument("--cropSize", type=int, default=224)
    parser.add_argument("--niter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--optim", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--tau_rgb", type=float, default=0.5)
    parser.add_argument("--tau_res", type=float, default=0.5)
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_num_threads", type=int, default=8)
    parser.add_argument(
        "--train_extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments appended to train.py after '--'.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    hf_dataset_path = Path(args.hf_dataset_path).expanduser()

    if (args.dataset_manifest or args.dataset_root) and args.hf_dataset_repo:
        raise ValueError(
            "Use either --dataset_root/--dataset_manifest to create a local "
            "HF dataset, or --hf_dataset_repo to train from Hub, not both."
        )

    if args.dataset_root or args.dataset_manifest:
        create_cmd = [
            sys.executable,
            str(repo_root / "data" / "create_dataset.py"),
            "--output_dir",
            str(hf_dataset_path),
            "--val_ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
        ]
        if args.dataset_root:
            create_cmd.extend(["--dataset_root", args.dataset_root])
        if args.dataset_manifest:
            create_cmd.extend(
                [
                    "--manifest",
                    args.dataset_manifest,
                    "--image_column",
                    args.image_column,
                    "--label_column",
                    args.label_column,
                    "--source_column",
                    args.source_column,
                    "--split_column",
                    args.split_column,
                ]
            )
        if args.image_base_dir and args.dataset_manifest:
            create_cmd.extend(["--image_base_dir", args.image_base_dir])
        _run(create_cmd)

    train_script = str(repo_root / "train.py")
    if args.nproc_per_node > 1:
        train_cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(args.nproc_per_node),
            train_script,
        ]
    else:
        train_cmd = [sys.executable, train_script]

    train_cmd.extend(
        [
            "--name",
            args.name,
            "--checkpoints_dir",
            args.checkpoints_dir,
            "--devices",
            args.devices,
            "--batch_size",
            str(args.batch_size),
            "--num_threads",
            str(args.num_threads),
            "--dino_variant",
            args.dino_variant,
            "--dino_pretrained_root",
            args.dino_pretrained_root,
            "--hf_dataset_path",
            str(hf_dataset_path),
            "--hf_train_split",
            "train",
            "--hf_eval_split",
            "val",
            "--cropSize",
            str(args.cropSize),
            "--niter",
            str(args.niter),
            "--lr",
            str(args.lr),
            "--optim",
            args.optim,
            "--accumulation_steps",
            str(args.accumulation_steps),
            "--weight_decay",
            str(args.weight_decay),
            "--label_smoothing",
            str(args.label_smoothing),
            "--tau_rgb",
            str(args.tau_rgb),
            "--tau_res",
            str(args.tau_res),
            "--eval_every_steps",
            str(args.eval_every_steps),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--eval_num_threads",
            str(args.eval_num_threads),
            "--seed",
            str(args.seed),
        ]
    )

    if args.hf_dataset_repo:
        train_cmd.extend(["--hf_dataset_repo", args.hf_dataset_repo])

    extra_args = args.train_extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    train_cmd.extend(extra_args)

    _run(train_cmd)


if __name__ == "__main__":
    main()

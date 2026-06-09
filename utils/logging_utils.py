
import logging
import os
from typing import Tuple


def setup_logging(
    checkpoints_dir: str,
    exp_name: str,
    level: int = logging.INFO,
    log_type: str = "train",
    rank: int = 0,
    is_main_process: bool = True,
) -> Tuple[logging.Logger, str]:

    os.makedirs(checkpoints_dir, exist_ok=True)

    if log_type == "test":
        log_file = os.path.join(checkpoints_dir, f"{exp_name}_test.log")
    else:
        log_file = os.path.join(checkpoints_dir, f"{exp_name}.log")

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers.clear()

    if not is_main_process:
        root_logger.setLevel(logging.ERROR)
        root_logger.addHandler(logging.NullHandler())
        return root_logger, ""

    root_logger.setLevel(level)

    formatter = logging.Formatter(
        f"%(asctime)s - rank{rank} - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    return root_logger, log_file


def log_training_config(logger: logging.Logger, opt) -> None:
    """Log the training configuration in a structured, human-readable form.

    All field accesses are direct: every attribute below is guaranteed to be
    populated by :func:`utils.cli.build_train_parser`.
    """
    logger.info("==== Training configuration ====")
    logger.info("Name: %s", opt.name)
    logger.info("Checkpoints dir: %s", opt.checkpoints_dir)
    logger.info("Devices: %s", opt.devices)
    logger.info(
        "Distributed: %s | rank=%d | local_rank=%d | world_size=%d",
        opt.distributed,
        opt.rank,
        opt.local_rank,
        opt.world_size,
    )
    logger.info("Batch size: %d", opt.batch_size)
    logger.info("Learning rate: %.6f", opt.lr)
    logger.info("Optimizer: %s", opt.optim)
    logger.info("Weight decay: %.4f", opt.weight_decay)
    logger.info("Label smoothing: %.2f", opt.label_smoothing)
    logger.info("Number of epochs: %d", opt.niter)
    logger.info("Crop size: %d", opt.cropSize)
    logger.info("DINO variant: %s", opt.dino_variant)
    logger.info("LoRA rank: %d", opt.lora_rank)
    logger.info("LoRA alpha: %.2f", opt.lora_alpha)
    logger.info("LoRA dropout: %.2f", opt.lora_dropout)
    logger.info("HF dataset path: %s", opt.hf_dataset_path)
    logger.info("HF dataset repo: %s", opt.hf_dataset_repo)
    logger.info("HF train split: %s", opt.hf_train_split)
    logger.info("HF eval split: %s", opt.hf_eval_split)
    real_image_dirs = opt.real_image_dir or []
    fake_image_dirs = opt.fake_image_dir or []
    logger.info("Real image dirs (%d):", len(real_image_dirs))
    for idx, real_dir in enumerate(real_image_dirs):
        logger.info("  [%d] %s", idx, real_dir)
    logger.info("Fake image dirs (%d):", len(fake_image_dirs))
    for idx, fake_dir in enumerate(fake_image_dirs):
        logger.info("  [%d] %s", idx, fake_dir)
    logger.info("Test root: %s", opt.test_root)
    logger.info("DINO pretrained root: %s", opt.dino_pretrained_root)
    logger.info("PGCM tau (RGB): %.4f", opt.tau_rgb)
    logger.info("PGCM tau (Residual): %.4f", opt.tau_res)
    logger.info("Accumulation steps: %d", opt.accumulation_steps)
    logger.info("Max grad norm: %.4f", opt.max_grad_norm)
    logger.info("Seed: %d", opt.seed)


def log_test_config(logger: logging.Logger, args) -> None:
    """Log the test configuration in a structured, human-readable form.

    All attribute accesses are direct (guaranteed by
    :func:`utils.cli.build_test_parser`).
    """
    logger.info("==== Test configuration ====")
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Test root: %s", args.test_root)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Num workers: %d", args.num_workers)
    logger.info("Devices: %s", args.devices)
    logger.info("Crop size: %d", args.cropSize)
    logger.info("DINO variant: %s", args.dino_variant)
    logger.info("LoRA rank: %d", args.lora_rank)
    logger.info("LoRA alpha: %.2f", args.lora_alpha)
    logger.info("LoRA dropout: %.2f", args.lora_dropout)
    logger.info("DINO pretrained root: %s", args.dino_pretrained_root)
    logger.info("PGCM tau (RGB): %.4f", args.tau_rgb)
    logger.info("PGCM tau (Residual): %.4f", args.tau_res)

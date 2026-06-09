import torch

from data.eval_dataset import HFRealFakeEvalDataset
from data.transforms import MEAN, STD, create_eval_transforms
from engine.evaluator import evaluate_model
from models.pgc import PGCNetwork
from utils.cli import build_test_parser, parse_devices
from utils.logging_utils import log_test_config, setup_logging
from utils.metrics import compute_mean_metrics, log_metrics


def main():
    # ---------------- CLI parsing & global setup ----------------
    parser = build_test_parser()
    args = parser.parse_args()

    devices, device = parse_devices(args.devices)
    args.devices = devices
    if device.type == "cuda":
        torch.cuda.set_device(device)

    logger_inst, log_path = setup_logging(args.checkpoints_dir, args.name, log_type='test')
    logger_inst.info('Logging to file: %s', log_path)
    log_test_config(logger_inst, args)
    logger_inst.info('Using device: %s', device)

    # ---------------- Build dataset + transform ----------------
    transform = create_eval_transforms(
        image_size=args.cropSize, mean=MEAN['dino'], std=STD['dino'],
    )
    logger_inst.info('Loading HF eval split: %s', args.hf_eval_split)
    dataset = HFRealFakeEvalDataset(
        transform=transform,
        split=args.hf_eval_split,
        subset_name=args.hf_eval_subset_name,
        dataset_path=args.hf_dataset_path,
        dataset_repo=args.hf_dataset_repo,
    )

    # ---------------- Build & load model ----------------
    model = PGCNetwork(
        dino_variant=args.dino_variant,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=None,
        pretrained_root=args.dino_pretrained_root,
        tau_rgb=args.tau_rgb,
        tau_res=args.tau_res,
    )
    logger_inst.info('Loading checkpoint: %s', args.checkpoint)
    state = torch.load(args.checkpoint, map_location=device)
    state_dict = state.get('model', state)
    if any(key.startswith('module.') for key in state_dict):
        state_dict = {
            key.removeprefix('module.'): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # ---------------- Evaluation ----------------
    logger_inst.info('Starting evaluation...')
    subset_metrics, _overall_micro = evaluate_model(
        model=model,
        device=device,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    mean_metrics = compute_mean_metrics(subset_metrics)
    log_metrics(mean_metrics, 'Overall test set mean')
    logger_inst.info('Evaluation completed. Subsets evaluated: %d', len(subset_metrics))


if __name__ == '__main__':
    main()

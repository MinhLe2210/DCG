import math
import logging
import time

import numpy as np

from data.dataloader import create_dataloader
from data.eval_dataset import HFRealFakeEvalDataset
from data.transforms import MEAN, STD, create_eval_transforms
from engine.evaluator import evaluate_model
from engine.trainer import PGCTrainer
from utils.cli import build_train_parser, finalize_opt
from utils.distributed import barrier, cleanup_distributed, setup_distributed
from utils.logging_utils import log_training_config, setup_logging
from utils.metrics import compute_mean_metrics, log_metrics
from utils.seed import set_seed


logger = logging.getLogger(__name__)


def _build_test_datasets(opt, eval_transform):
    test_datasets = {}
    if opt.hf_eval_split is not None:
        test_datasets['HF'] = HFRealFakeEvalDataset(
            transform=eval_transform,
            split=opt.hf_eval_split,
            subset_name=opt.hf_eval_subset_name,
            dataset_path=opt.hf_dataset_path,
            dataset_repo=opt.hf_dataset_repo,
        )
    return test_datasets


def _evaluate_all_datasets(trainer, opt, test_datasets, logger_inst):
    all_mean_accs = []
    all_mean_aps = []
    for ds_name, ds in test_datasets.items():
        logger_inst.info('>>> Evaluating on %s dataset', ds_name)
        subset_metrics, overall = evaluate_model(
            model=trainer.eval_model,
            device=trainer.device,
            dataset=ds,
            batch_size=opt.eval_batch_size,
            num_workers=opt.eval_num_threads,
        )
        mean_metrics = compute_mean_metrics(subset_metrics)
        # Macro-mean ACC / AP across subsets; fall back to overall when NaN.
        ds_mean_acc = mean_metrics.get('acc', float('nan'))
        if np.isnan(ds_mean_acc):
            ds_mean_acc = overall.get('acc', float('-inf'))
        ds_mean_ap = mean_metrics.get('ap', float('nan'))
        if np.isnan(ds_mean_ap):
            ds_mean_ap = overall.get('ap', float('-inf'))
        all_mean_accs.append(ds_mean_acc)
        all_mean_aps.append(ds_mean_ap)
        log_metrics(mean_metrics, f'  {ds_name} mean')

    def _aggregate(values):
        valid = [v for v in values if not np.isnan(v) and v != float('-inf')]
        if len(valid) == 0:
            return float('-inf')
        return float(np.mean(valid))

    return _aggregate(all_mean_accs), _aggregate(all_mean_aps)


def _format_checkpoint_name(step: int, acc: float, ap: float) -> str:
    def _percent(value: float) -> int:
        if np.isnan(value) or value == float('-inf') or value == float('inf'):
            return 0
        return int(round(value * 100))

    return f'step{step}_acc{_percent(acc)}_ap{_percent(ap)}.pth'


def main():
    # ---------------- CLI parsing & global setup ----------------
    parser = build_train_parser()
    opt = parser.parse_args()
    try:
        opt_message = finalize_opt(opt)

        logger_inst, log_path = setup_logging(
            opt.checkpoints_dir,
            opt.name,
            log_type='train',
            rank=opt.rank,
            is_main_process=opt.is_main_process,
        )
        setup_distributed(opt)
        set_seed(opt.seed + opt.rank)

        if opt.is_main_process:
            logger_inst.info('Logging to file: %s', log_path)
            logger_inst.info('\n%s', opt_message)
            log_training_config(logger_inst, opt)

        # ---------------- Build trainer + dataloaders ----------------
        trainer = PGCTrainer(opt)
        data_loader = create_dataloader(opt)
        if len(data_loader) == 0:
            raise RuntimeError(
                "Training dataloader is empty. Check dataset size, "
                "batch_size, world_size, and drop_last settings."
            )

        eval_transform = create_eval_transforms(
            image_size=opt.cropSize, mean=MEAN['dino'], std=STD['dino'],
        )
        test_datasets = _build_test_datasets(opt, eval_transform)
        if len(test_datasets) == 0 and opt.is_main_process:
            logger_inst.info('No test datasets provided; skipping per-step evaluation.')

        total_micro_steps = opt.niter * len(data_loader)
        updates_per_epoch = math.ceil(len(data_loader) / opt.accumulation_steps)
        total_update_steps = opt.niter * updates_per_epoch
        trainer.setup_scheduler(total_update_steps)
        logger_inst.info(
            'Total micro steps: %d | optimizer updates: %d '
            '(epochs=%d, batches/epoch/rank=%d, updates/epoch=%d)',
            total_micro_steps,
            total_update_steps,
            opt.niter,
            len(data_loader),
            updates_per_epoch,
        )

        # ---------------- Training loop ----------------
        best_mean_ap = float('-inf')
        start_time = time.time()

        for epoch in range(opt.niter):
            sampler = getattr(data_loader, 'sampler', None)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

            for batch in data_loader:
                trainer.set_input(batch)
                trainer.optimize_parameters()

                if opt.is_main_process and trainer.total_steps % 100 == 0:
                    elapsed = time.time() - start_time
                    logger_inst.info(
                        'Step: %d | Loss: %.4f | Avg Time/Step: %.4fs',
                        trainer.total_steps,
                        trainer.loss.detach().item(),
                        elapsed / trainer.total_steps,
                    )

                should_eval = (
                    len(test_datasets) > 0
                    and opt.eval_every_steps > 0
                    and trainer.total_steps % opt.eval_every_steps == 0
                )
                if should_eval:
                    trainer.eval()
                    barrier()

                    if opt.is_main_process:
                        logger_inst.info('=' * 80)
                        logger_inst.info('Evaluating model at step %d', trainer.total_steps)
                        logger_inst.info('=' * 80)

                        overall_acc, overall_map = _evaluate_all_datasets(
                            trainer, opt, test_datasets, logger_inst,
                        )
                        logger_inst.info(
                            'Overall mean ACC: %.4f | Overall mean AP: %.4f',
                            overall_acc, overall_map,
                        )

                        if overall_map > best_mean_ap:
                            best_mean_ap = overall_map
                            logger_inst.info(
                                'New best model (mean AP=%.4f); saving best.pth',
                                overall_map,
                            )
                            trainer.save_networks('best.pth')

                        ckpt_name = _format_checkpoint_name(
                            trainer.total_steps, overall_acc, overall_map,
                        )
                        logger_inst.info(
                            'Saving evaluation checkpoint: %s '
                            '(step=%d, mean ACC=%.4f, mean AP=%.4f)',
                            ckpt_name, trainer.total_steps, overall_acc, overall_map,
                        )
                        trainer.save_networks(ckpt_name)

                    barrier()
                    trainer.train()

            trainer.finalize_epoch()

        if opt.is_main_process:
            logger_inst.info('Training finished. Best mean AP: %.4f', best_mean_ap)
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()

import argparse
import logging

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .train_dataset import RealFakeDataset

logger = logging.getLogger(__name__)


def create_dataloader(opt: argparse.Namespace) -> DataLoader:
    """
    Build the PGC training DataLoader.

    ``RealFakeDataset`` now reads the Hugging Face ``train`` split, but its
    class name and output contract stay unchanged.
    """
    dataset = RealFakeDataset(opt)

    logger.info("Dataset size: %d", len(dataset))
    sampler = None
    if getattr(opt, "distributed", False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=opt.world_size,
            rank=opt.rank,
            shuffle=True,
            drop_last=True,
        )

    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=opt.num_threads,
        pin_memory=True,
        drop_last=True,
    )

import contextlib
import datetime
import logging
import os
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn


logger = logging.getLogger(__name__)


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(opt) -> None:
    if not getattr(opt, "distributed", False):
        return

    if is_dist_avail_and_initialized():
        return

    backend = getattr(opt, "dist_backend", "auto")
    if backend == "auto":
        backend = (
            "nccl"
            if torch.cuda.is_available()
            and getattr(opt, "device", torch.device("cpu")).type == "cuda"
            and os.name != "nt"
            else "gloo"
        )

    dist.init_process_group(
        backend=backend,
        init_method=getattr(opt, "dist_url", "env://"),
        world_size=int(opt.world_size),
        rank=int(opt.rank),
        timeout=datetime.timedelta(hours=6),
    )
    barrier()

    if is_main_process():
        logger.info(
            "Distributed training initialized: backend=%s world_size=%d",
            backend,
            int(opt.world_size),
        )


def barrier() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


@contextlib.contextmanager
def main_process_first() -> Iterator[None]:
    if get_world_size() <= 1:
        yield
        return

    rank = get_rank()
    if rank != 0:
        barrier()

    yield

    if rank == 0:
        barrier()

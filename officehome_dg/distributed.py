from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import distributed as dist


@dataclass
class Runtime:
    device: torch.device
    distributed: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def init_runtime(requested_device: str) -> Runtime:
    requested = resolve_device(requested_device)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed:
        return Runtime(device=requested, distributed=False)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    if requested.type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = requested
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return Runtime(device=device, distributed=True, rank=rank, local_rank=local_rank, world_size=world_size)


def reduce_sums(values: torch.Tensor, runtime: Runtime) -> torch.Tensor:
    if runtime.distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def shared_text(value: str, runtime: Runtime) -> str:
    if not runtime.distributed:
        return value
    payload = [value if runtime.is_main else ""]
    dist.broadcast_object_list(payload, src=0)
    return str(payload[0])


def barrier(runtime: Runtime) -> None:
    if runtime.distributed:
        dist.barrier()


def cleanup(runtime: Runtime) -> None:
    if runtime.distributed and dist.is_initialized():
        dist.destroy_process_group()

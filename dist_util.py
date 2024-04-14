import os
import torch
import torch.distributed as dist


def setup_dist():
    if dist.is_initialized():
        return
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    _device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    torch.cuda.set_device(_device)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_params(params):
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)

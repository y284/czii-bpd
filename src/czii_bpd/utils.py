import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


def devices():
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [torch.device("cpu")]


def amp_dtype(device):
    """bfloat16 where the GPU supports it natively (compute capability >= 8), float16 otherwise (e.g. T4)."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def autocast(device):
    return torch.autocast(device.type, dtype=amp_dtype(device), enabled=device.type == "cuda")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_on_devices(jobs, fn):
    """Returns [fn(device, job) for job in jobs]. With 2+ GPUs, each GPU takes every n-th job in its own thread."""
    devs = devices()
    if len(devs) == 1:
        return [fn(devs[0], job) for job in jobs]
    out = [None] * len(jobs)

    def work(i):
        for k in range(i, len(jobs), len(devs)):
            out[k] = fn(devs[i], jobs[k])

    with ThreadPoolExecutor(len(devs)) as pool:
        list(pool.map(work, range(len(devs))))
    return out

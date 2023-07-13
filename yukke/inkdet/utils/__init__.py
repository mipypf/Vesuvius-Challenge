import os
import random
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from .config import Config, DictAction  # noqa
from .eval import calc_fbeta  # noqa


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id: int, seed: int):
    """Function to initialize each worker.

    The seed of each worker equals to
    ``worker_id + user_seed``.

    Args:
        worker_id (int): Id for each worker.
        seed (int): Random seed.
    """
    worker_seed = worker_id + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _minimal_ext_cmd(cmd: List[str]):
    # construct minimal environment
    env = {}
    for k in ["SYSTEMROOT", "PATH", "HOME"]:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env["LANGUAGE"] = "C"
    env["LANG"] = "C"
    env["LC_ALL"] = "C"
    out, err = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env).communicate()
    return out


def get_git_hash(fallback: str = "unknown", digits: Optional[int] = None):
    if digits is not None and not isinstance(digits, int):
        raise TypeError("digits must be None or an integer")

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        sha = out.strip().decode("ascii")
        if digits is not None:
            sha = sha[:digits]
    except OSError:
        sha = fallback

    return sha


def load_checkpoint(model: nn.Module, checkpoint_path: Union[str, Path], strict: bool = True):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["state_dict"].items()}
    status = model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Load checkpoint: {checkpoint_path} - {status}")

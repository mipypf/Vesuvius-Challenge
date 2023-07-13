"""Model average script."""

import argparse
import copy
from pathlib import Path
from typing import Dict, List

import torch
from loguru import logger


def average_weights(checkpoints: List[Path]):
    def _load_state_dict(_path: Path) -> Dict[str, torch.Tensor]:
        return torch.load(_path, map_location=torch.device("cpu"))["state_dict"]

    state_dict = _load_state_dict(checkpoints[0])
    not_found_keys: list = list(state_dict.keys())
    for i in range(1, len(checkpoints)):
        _state_dict = _load_state_dict(checkpoints[i])
        _not_found_keys = copy.deepcopy(not_found_keys)
        for k in state_dict.keys():
            state_dict[k] += _state_dict[k]
            _not_found_keys.remove(k)

        if len(_not_found_keys) > 0:
            logger.info(f"Not found keys in checkpoint: {_not_found_keys}")

    # average
    num_ckpts = float(len(checkpoints))
    for k in state_dict.keys():
        dtype = state_dict[k].dtype
        state_dict[k] = (state_dict[k] / num_ckpts).to(dtype=dtype)

    return {"state_dict": state_dict}


@logger.catch()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-ckpts",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--out-ckpt",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    logger.info(f"load checkpoints: {args.in_ckpts}")
    checkpoint = average_weights(args.in_ckpts)
    torch.save(checkpoint, args.out_ckpt)
    logger.info(f"saved to {args.out_ckpt}")


if __name__ == "__main__":
    main()

import argparse
import os
from functools import partial
from pathlib import Path
from typing import Callable, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as vF
from loguru import logger
from tqdm import tqdm

from inkdet.datasets.utils import build_dataloader, read_image
from inkdet.models import build_model
from inkdet.models.utils import interpolate
from inkdet.utils import Config, DictAction, calc_fbeta, set_seed


def valid_fn(
    cfg: Config,
    dataloader,
    model,
    device,
    valid_xyxys,
    valid_masks,
    use_tta: bool,
    precision=torch.half,
):
    mask_pred = np.zeros(valid_masks.shape)
    mask_count = np.zeros(valid_masks.shape)

    model.eval()
    if precision == torch.half:
        model = model.half()

    patch_size: int = cfg.patch_size
    target_size: int = cfg.target_size if cfg.target_size > 0 else cfg.patch_size
    tsi: int = patch_size // 2 - target_size // 2
    tei: int = patch_size // 2 + target_size // 2

    logger.info(f"Use TTA: {use_tta}")
    logger.info(f"Transforms: {cfg.transforms}")

    # disable to load labels
    dataloader.dataset.labels = None
    valid_xyxys = dataloader.dataset.xyxys

    for step, images in tqdm(enumerate(dataloader), total=len(dataloader)):
        images = images.to(device).to(precision)

        def to_transform(_trans: str) -> Tuple[Callable, Callable]:
            if _trans is None:
                forward_func = lambda x: x
                inverse_func = lambda x: x
            elif _trans.startswith("rot"):
                angle = float(_trans[3:])
                forward_func = partial(vF.rotate, angle=angle, interpolation=vF.InterpolationMode.BILINEAR)
                inverse_func = partial(vF.rotate, angle=-angle, interpolation=vF.InterpolationMode.BILINEAR)
            elif _trans.startswith("flip"):
                direction = _trans[4:]
                assert direction in ["h", "v"]
                dim = -2 if direction == "v" else -1
                forward_func = partial(torch.flip, dims=(dim,))
                inverse_func = partial(torch.flip, dims=(dim,))
            elif _trans.startswith("resize"):
                size = int(_trans[6:])
                forward_func = partial(F.interpolate, size=(size, size), mode="bilinear")
                inverse_func = partial(F.interpolate, size=(cfg.patch_size, cfg.patch_size), mode="bilinear")
            else:
                raise ValueError(f"Unknown transform: {_trans}")

            return forward_func, inverse_func

        with torch.no_grad():
            x = images
            if use_tta:
                assert len(cfg.transforms) > 0
                transforms = [to_transform(_trans) for _trans in cfg.transforms]
                xx = [trans(x) for trans, _ in transforms]

                _y = [torch.sigmoid(model(x)) for x in xx]

                yy = [trans(_y[ti]) for ti, (_, trans) in enumerate(transforms)]

                y = torch.mean(torch.stack(yy, dim=0), dim=0)
            else:
                assert len(cfg.transforms) <= 1

                if len(cfg.transforms) == 1:
                    forward_func, inverse_func = to_transform(cfg.transforms[0])
                    x = forward_func(x)
                    y = torch.sigmoid(model(x))
                    y = inverse_func(y)
                else:
                    y = torch.sigmoid(model(x))

        # if hasattr(model, "attention_outputs"):
        #     for i, attention in enumerate(model.attention_outputs):
        #         attention_outputs[f"attention{i}"].append(attention)

        if hasattr(model.decoder, "downsample_factor"):
            factor = model.decoder.downsample_factor
            if factor > 1:
                y = interpolate(y, mode="bilinear", scale_factor=factor, align_corners=True)

        # make whole mask
        y_preds = y.squeeze(1).to("cpu").to(torch.float32).numpy()
        start_idx = step * cfg.batch_size
        end_idx = start_idx + cfg.batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            _pred = np.zeros((patch_size, patch_size), dtype=np.float32)
            _pred[tsi:tei, tsi:tei] = y_preds[i, tsi:tei, tsi:tei]
            _cnt = np.zeros((patch_size, patch_size), dtype=np.float32)
            _cnt[tsi:tei, tsi:tei] = 1

            mask_pred[y1:y2, x1:x2] += _pred
            mask_count[y1:y2, x1:x2] += _cnt

    # import pickle
    # with open("work_dirs/20230518_225615_attentions.pkl", "wb") as wf:
    #     pickle.dump(attention_outputs, wf)

    # with open("work_dirs/20230518_225615_valid_xyxys.pkl", "wb") as wf:
    #     pickle.dump(valid_xyxys, wf)

    non_zero = mask_count > 0
    mask_pred[non_zero] /= mask_count[non_zero]

    return mask_pred


@logger.catch()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--valid-id", type=str, required=True, choices=["1", "2", "3", "2a", "2b"])
    parser.add_argument("--checkpoint-suffix", type=str, required=True)
    parser.add_argument("--eval-suffix", type=str, default="")
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    args = parser.parse_args()

    assert torch.cuda.is_available()

    work_dir: Path = args.work_dir
    valid_id: str = args.valid_id
    checkpoint_suffix: str = args.checkpoint_suffix
    eval_suffix: str = f"_{args.eval_suffix}" if args.eval_suffix else ""
    if args.use_tta:
        eval_suffix += "_tta"

    assert work_dir.exists()

    config_paths = list(work_dir.glob("*.yaml"))
    assert len(config_paths) == 1, config_paths
    config_path = config_paths[0]

    cfg_options = dict() if args.cfg_options is None else args.cfg_options
    cfg_options["use_mask"] = True
    cfg = Config.fromfile(config_path, cfg_options)
    set_seed(cfg.seed)

    checkpoint_paths = list(work_dir.glob(f"{cfg.model_name}_fold*_{checkpoint_suffix}.pth"))
    assert len(checkpoint_paths) == 1
    checkpoint_path = checkpoint_paths[0]
    assert checkpoint_path.exists()

    device = torch.device("cuda")
    model = build_model(cfg)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    org_state_dict = checkpoint["state_dict"]
    state_dict = dict()
    for k, v in org_state_dict.items():
        state_dict[k.replace("_orig_mod.", "")] = v
    status = model.load_state_dict(state_dict, strict=True)
    logger.info(f"Load checkpoint: {checkpoint_path} - {status}")
    # model = torch.compile(model)

    target_size = cfg.target_size if cfg.target_size > 0 else cfg.patch_size
    valid_masks, padding = read_image(
        cfg.data_root,
        f"train/{valid_id}/mask.png",
        cfg.patch_size,
        target_size,
    )
    valid_masks = valid_masks > 0

    valid_gt_labels, _ = read_image(
        cfg.data_root,
        f"train/{valid_id}/inklabels.png",
        cfg.patch_size,
        target_size,
        padding,
    )
    valid_gt_labels = valid_gt_labels > 0

    # valid_gt_labels = np.rot90(valid_gt_labels)
    # valid_masks = np.rot90(valid_masks)

    # eval
    valid_dataloader, valid_xyxys = build_dataloader("val", [valid_id], cfg)
    labels_pred = valid_fn(
        cfg,
        valid_dataloader,
        model,
        device,
        valid_xyxys,
        valid_masks,
        args.use_tta,
        precision=torch.half if args.precision == "fp16" else torch.float32,
    )
    labels_pred = padding.slice(labels_pred)

    stride = cfg.stride if cfg.val_stride == 0 else cfg.val_stride
    pred_path = work_dir / f"prediction_{checkpoint_suffix}_p{cfg.patch_size}_t{target_size}_s{stride}{eval_suffix}.npy"
    np.save(pred_path, labels_pred)
    logger.info(f"Save prediction: {pred_path}")

    valid_masks = padding.slice(valid_masks)
    valid_gt_labels = padding.slice(valid_gt_labels)
    best_metric_dict, metrics = calc_fbeta(valid_gt_labels, labels_pred, valid_masks)

    metrics_df = pd.DataFrame(metrics)
    metrics_path = work_dir / f"metrics_{checkpoint_suffix}_p{cfg.patch_size}_t{target_size}_s{stride}{eval_suffix}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Save metrics: {metrics_path}")

    best_threshold = best_metric_dict["threshold"]
    best_fscore = best_metric_dict["fscore"]
    best_precision = best_metric_dict["precision"]
    best_recall = best_metric_dict["recall"]

    img = np.zeros((*labels_pred.shape, 3), dtype=np.uint8)
    img[np.logical_and(valid_gt_labels == 1, (labels_pred >= best_threshold) == 1), 0] = 255  # True Positive
    img[np.logical_and(valid_gt_labels == 0, (labels_pred >= best_threshold) == 1), 1] = 255  # False Positive
    img[np.logical_and(valid_gt_labels == 1, (labels_pred >= best_threshold) == 0), 2] = 255  # False Negative
    labels_pred[0, 0] = 0
    labels_pred[0, 1] = 1

    _, axes = plt.subplots(1, 3, figsize=(18, 9))
    axes[0].imshow(valid_gt_labels)
    axes[1].imshow(img)
    axes[1].set_title(
        f"pred labels: TP=red FP=greed FN=blue\n"
        + f"Fscore={best_fscore:.3f} Precision={best_precision:.3f} Recall={best_recall:.3f}\n"
        + f"threshold={best_threshold:.2f}"
    )
    axes[2].imshow(labels_pred)

    fig_path = (
        work_dir / f"visualization_{checkpoint_suffix}_p{cfg.patch_size}_t{target_size}_s{stride}{eval_suffix}.png"
    )
    plt.savefig(fig_path)
    logger.info(f"Save fig: {fig_path}")


if __name__ == "__main__":
    main()

import argparse
import dataclasses
import datetime as dt
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from inkdet.datasets.utils import build_dataloader, get_train_valid_ids
from inkdet.models import Losses, build_criterion, build_model, build_optimizer_scheduler
from inkdet.models.utils import interpolate
from inkdet.utils import AverageMeter, Config, DictAction, calc_fbeta, get_git_hash, load_checkpoint, set_seed


class Mixup(object):
    def __init__(self, p: float = 0.5, alpha: float = 0.5):
        self.p = p
        self.alpha = alpha
        self.lam = 1.0
        self.do_mixup = False

    def init_lambda(self):
        if np.random.rand() < self.p:
            self.do_mixup = True
        else:
            self.do_mixup = False
        if self.do_mixup and self.alpha > 0.0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1.0

    def reset_lambda(self):
        self.lam = 1.0


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def is_module(_model) -> bool:
    return hasattr(_model, "module")


def train_fn(
    cfg: Config,
    dataloader,
    model,
    criterion: Losses,
    optional_criterion: Optional[Losses],
    optimizer,
    device,
    writer: Optional[SummaryWriter],
    total_step: int,
    mix_augmentation_p: float,
    mix_augmentation_alpha: float,
    use_manifold_mixup: bool,
    debug: bool,
):
    model.train()

    scaler = GradScaler(enabled=cfg.use_amp)
    loss_meter = AverageMeter()
    optional_loss_meter = AverageMeter()
    grad_norms = AverageMeter()

    if mix_augmentation_p > 0:
        logger.info(f"Mixup and CutMix images and labels: beta({mix_augmentation_alpha}, {mix_augmentation_alpha})")
        mixer = Mixup(p=mix_augmentation_p, alpha=mix_augmentation_alpha)

    # https://github.com/mipypf/Vesuvius-Challenge/blob/a2d23a0341f0972e9d2e3fc9f8ba40a7a6450113/tattaka/src/exp055/train.py#L744-L778
    def inference_with_mixup(
        _model: nn.Module,
        _images: torch.Tensor,
        _labels: torch.Tensor,
        _rot_labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if mix_augmentation_p <= 0:
            return _images, _labels, _rot_labels, None

        assert 0 < mix_augmentation_p <= 1
        assert mix_augmentation_alpha > 0

        _image_feature: Optional[torch.Tensor] = None
        mixer.init_lambda()
        if mixer.do_mixup:
            if use_manifold_mixup:
                if np.random.rand() < 0.5:
                    _images = mixer.lam * _images + (1 - mixer.lam) * _images.flip(0)
                else:
                    _image_feature = _model(_images, feature_only=True)
                    _image_feature = mixer.lam * _image_feature + (1 - mixer.lam) * _image_feature.flip(0)
            else:
                _images = mixer.lam * _images + (1 - mixer.lam) * _images.flip(0)
        else:
            if np.random.rand() < 0.5:
                lam = (
                    np.random.beta(mix_augmentation_alpha, mix_augmentation_alpha) if mix_augmentation_alpha > 0 else 0
                )
                x1, y1, x2, y2 = rand_bbox(_images.size(), lam)
                _images[:, :, y1:y2, x1:x2] = _images.flip(0)[:, :, y1:y2, x1:x2]
                _labels[:, :, y1:y2, x1:x2] = _labels.flip(0)[:, :, y1:y2, x1:x2]
                if _rot_labels is not None:
                    lam = 1 - (y2 - y1) * (x2 - x1) / math.prod(_images.size()[2:])
                    _rot_labels = lam * _rot_labels + (1 - lam) * _rot_labels.flip(0)

        return _images, _labels, _rot_labels, _image_feature

    def calc_loss(_criterion: Losses, _y_pred: torch.Tensor, _y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        _losses: dict[str, torch.Tensor] = _criterion(_y_pred, _y_true)

        if mixer.do_mixup:
            _y_true_flip = _y_true.flip(0)
            _losses_flip = _criterion(_y_pred, _y_true_flip)
            for key in _losses:
                _losses[key] = mixer.lam * _losses[key] + (1 - mixer.lam) * _losses_flip[key]

        return _losses

    for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        if len(data) == 2:
            images, labels = data
            rot_labels = None
        elif len(data) == 3:
            images, labels, rot_labels = data
            rot_labels: torch.Tensor = rot_labels.to(device)
        else:
            raise ValueError(f"Unexpected data size: {len(data)}")

        images: torch.Tensor = images.to(device)
        labels: torch.Tensor = labels.to(device)
        batch_size = labels.size(0)

        with autocast(enabled=cfg.use_amp):
            images, labels, rot_labels, _image_feature = inference_with_mixup(model, images, labels, rot_labels)

            if _image_feature is None:
                y_preds = model(images)
            else:
                if is_module(model):
                    y_preds = model.module.decoder.head(_image_feature)
                else:
                    y_preds = model.decoder.head(_image_feature)

            if isinstance(y_preds, tuple):
                y_preds, rot_preds = y_preds
            else:
                rot_preds = None

            if cfg.logit_clip_epsilon is not None:
                assert 0 < cfg.logit_clip_epsilon < 0.1
                e = cfg.logit_clip_epsilon

                def inverse_sigmoid(_y):
                    return math.log(_y / (1 - _y))

                y_preds = torch.clamp(y_preds, inverse_sigmoid(e), inverse_sigmoid(1 - e))

            if labels.shape[1] == 1:
                labels = labels
                pseudo_labels = None
            elif labels.shape[1] == 2:
                labels, pseudo_labels = torch.split(labels, 1, dim=1)
            else:
                raise ValueError(f"Unexpected labels shape: {labels.shape}")

            if is_module(model):
                if hasattr(model.module.decoder, "downsample_factor"):
                    factor = model.module.decoder.downsample_factor
                    labels = interpolate(labels, mode="bilinear", scale_factor=1 / factor, align_corners=True)
            else:
                if hasattr(model.decoder, "downsample_factor"):
                    factor = model.decoder.downsample_factor
                    labels = interpolate(labels, mode="bilinear", scale_factor=1 / factor, align_corners=True)

            losses: dict[str, torch.Tensor] = calc_loss(criterion, y_preds, labels)

            loss = sum(losses.values())
            if optional_criterion is not None:
                if pseudo_labels is not None:
                    assert mix_augmentation_p == 0, "pseudo_label loss isn't support mix_augmentation"
                    optional_loss = sum(optional_criterion(y_preds, pseudo_labels).values())
                elif rot_preds is not None:
                    optional_losses: dict[str, torch.Tensor] = calc_loss(optional_criterion, rot_preds, rot_labels)
                    optional_loss = sum(optional_losses.values())
                    # optional_loss = None
                else:
                    raise ValueError("Expected to use either pseudo-labeling or rot classification.")
            else:
                optional_loss = None

        if optional_loss is not None:
            loss += optional_loss
        scaler.scale(loss).backward()

        loss_meter.update(loss.item(), batch_size)
        if optional_loss is not None:
            optional_loss_meter.update(optional_loss.item(), batch_size)

        if cfg.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            grad_norms.update(float(grad_norm), batch_size)

        if writer is not None and total_step % cfg.log_interval == 0:
            writer.add_scalar("train/grad_norm", grad_norms.avg, total_step)
            writer.add_scalar("train/loss", loss_meter.avg, total_step)
            if optional_loss is not None:
                writer.add_scalar("train/optional_loss", optional_loss_meter.avg, total_step)

        if debug and step > 3:
            break

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_step += 1

    return loss_meter.avg, total_step


def valid_fn(
    cfg: Config,
    dataloader,
    model: nn.Module,
    criterion: Losses,
    optional_criterion: Optional[Losses],
    device: torch.device,
    valid_xyxys,
    valid_gt_labels,
    debug: bool,
):
    mask_pred = np.zeros(valid_gt_labels.shape)
    mask_count = np.zeros(valid_gt_labels.shape)

    model.eval()
    loss_meter = AverageMeter()
    patch_size: int = cfg.val_patch_size if cfg.val_patch_size > 0 else cfg.patch_size
    valid_xyxys = dataloader.dataset.xyxys
    rots_gt_ret = []
    rots_pd_ret = []

    for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        if len(data) == 2:
            images, labels = data
            rot_labels = None
        elif len(data) == 3:
            images, labels, rot_labels = data
            rot_labels: torch.Tensor = rot_labels.to(device)
        else:
            raise ValueError(f"Unexpected data size: {len(data)}")

        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(images)

            if isinstance(y_preds, tuple):
                y_preds, rot_preds = y_preds
            else:
                rot_preds = None

            if cfg.logit_clip_epsilon is not None:
                assert 0 < cfg.logit_clip_epsilon < 0.1
                e = cfg.logit_clip_epsilon

                def inverse_sigmoid(_y):
                    return math.log(_y / (1 - _y))

                y_preds = torch.clamp(y_preds, inverse_sigmoid(e), inverse_sigmoid(1 - e))

            if labels.shape[1] == 1:
                labels = labels
                pseudo_labels = None
            elif labels.shape[1] == 2:
                labels, pseudo_labels = torch.split(labels, 1, dim=1)
            else:
                raise ValueError(f"Unexpected labels shape: {labels.shape}")

            if is_module(model):
                if hasattr(model.module.decoder, "downsample_factor"):
                    factor = model.module.decoder.downsample_factor
                    labels = interpolate(labels, mode="bilinear", scale_factor=1 / factor, align_corners=True)
            else:
                if hasattr(model.decoder, "downsample_factor"):
                    factor = model.decoder.downsample_factor
                    labels = interpolate(labels, mode="bilinear", scale_factor=1 / factor, align_corners=True)

            loss = sum(criterion(y_preds, labels).values())
            if optional_criterion is not None:
                if pseudo_labels is not None:
                    loss += sum(optional_criterion(y_preds, pseudo_labels).values())
                elif rot_preds is not None:
                    loss += sum(optional_criterion(rot_preds, rot_labels).values())
                else:
                    raise ValueError("Expected to use either pseudo-labeling or rot classification.")

            if rot_labels is not None:
                rots_gt_ret.append(rot_labels.cpu().numpy())
                rots_pd_ret.append(torch.sigmoid(rot_preds).cpu().numpy())

        loss_meter.update(loss.item(), batch_size)

        if is_module(model):
            if hasattr(model.module.decoder, "downsample_factor"):
                factor = model.module.decoder.downsample_factor
                y_preds = interpolate(y_preds, mode="bilinear", scale_factor=factor, align_corners=True)
        else:
            if hasattr(model.decoder, "downsample_factor"):
                factor = model.decoder.downsample_factor
                y_preds = interpolate(y_preds, mode="bilinear", scale_factor=factor, align_corners=True)

        # make whole mask
        y_preds = torch.sigmoid(y_preds).to("cpu").numpy()
        start_idx = step * cfg.batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += np.ones((patch_size, patch_size))

        if debug and step > 3:
            break

    non_zero = mask_count > 0
    mask_pred[non_zero] /= mask_count[non_zero]

    del non_zero
    del mask_count

    if len(rots_gt_ret) > 0 and len(rots_pd_ret) > 0:
        rots_gt_ret = np.concatenate(rots_gt_ret).reshape(-1)
        rots_pd_ret = np.concatenate(rots_pd_ret).reshape(-1)
    else:
        rots_gt_ret = rots_pd_ret = np.array([])

    return loss_meter.avg, mask_pred, rots_gt_ret, rots_pd_ret


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--valid-id", type=str, default="1", choices=["1", "2", "3", "2a", "2b"])
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device-ids", type=int, nargs="+")
    return parser.parse_args()


@logger.catch()
def main():
    args = _parse_args()
    config_path: Path = args.config
    valid_id: str = args.valid_id

    assert config_path.exists()
    assert config_path.suffix == ".yaml"
    assert torch.cuda.is_available()

    cfg = Config.fromfile(config_path, args.cfg_options)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    exp_name: str = (
        f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{config_path.stem}_{cfg.model_name}_{cfg.encoder_name}"
    )
    work_dir = Path(f"/opt/kaggle-ink-detection/work_dirs/{cfg.competition_name}/{exp_name}_{valid_id}/")

    if not args.debug:
        logger.add(os.path.join(work_dir, "log.txt"))
        cfg.dump(work_dir / config_path.name)
    logger.info(f"WORK_DIR: {work_dir}")
    logger.info(f"git commit hash: {get_git_hash(digits=7)}")
    logger.info(f"CFG: \n {yaml.dump(dataclasses.asdict(cfg), indent=2, sort_keys=False)}")

    model = build_model(cfg)
    if cfg.pretrained_path:
        load_checkpoint(model, cfg.pretrained_path)
    model.to(device)
    if args.device_ids is not None:
        logger.info(f"DataParallel mode: {args.device_ids}")
        model = nn.DataParallel(model, device_ids=args.device_ids)
    if not args.debug:
        logger.info("Use torch.compile.")
        model = torch.compile(model)
    logger.info(f"model: {cfg.model_name}")
    logger.info(f"encoder: {cfg.encoder_name}")
    logger.info(f"{model}")

    criterion = build_criterion(cfg.losses)
    logger.info(f"criterion: {criterion}")
    if cfg.use_pseudo_label:
        assert len(cfg.optional_losses) > 0, "pseudo_label_loss is enabled, but optional_losses isn't set."
        optional_criterion = build_criterion(cfg.optional_losses)
        logger.info(f"optional_criterion: {optional_criterion}")
    elif cfg.task == "segmentation-classification":
        assert (
            len(cfg.optional_losses) > 0
        ), "segmentation-classification multi-task learning is enabled, but optional_losses isn't set."
        optional_criterion = build_criterion(cfg.optional_losses)
        logger.info(f"optional_criterion: {optional_criterion}")
    else:
        assert (
            len(cfg.optional_losses) == 0
        ), "use_pseudo_label is False, but optional_losses is set. Make sure to use pseudo_label for training."
        optional_criterion = None

    if cfg.encoder_lr_scale > 0:
        model_params = [
            {"params": model.encoder.parameters(), "lr": cfg.max_lr / cfg.warmup_factor * cfg.encoder_lr_scale},
            {"params": model.decoder.parameters()},
        ]
        optimizer, scheduler = build_optimizer_scheduler(model_params, cfg)
    else:
        optimizer, scheduler = build_optimizer_scheduler(model.parameters(), cfg)
    logger.info(f"optimizer: {optimizer}")
    logger.info(f"scheduler: {scheduler}")

    train_ids, valid_ids = get_train_valid_ids(valid_id)
    pseudo_label_fold_id = valid_id if cfg.use_pseudo_label else None
    train_dataloader, _ = build_dataloader("train", train_ids, cfg, pseudo_label_fold_id)
    valid_dataloader, valid_xyxys = build_dataloader("val", valid_ids, cfg)
    logger.info(f"train ids: {train_ids}")
    logger.info(f"train dataset size: {len(train_dataloader.dataset)}")
    logger.info(f"valid ids: {valid_ids}")
    logger.info(f"valid dataset size: {len(valid_dataloader.dataset)}")

    patch_size: int = cfg.val_patch_size if cfg.val_patch_size > 0 else cfg.patch_size
    valid_gt_labels = cv2.imread(os.path.join(cfg.data_root, f"train/{valid_id}/inklabels.png"), 0)
    valid_gt_labels = valid_gt_labels / 255
    pad0 = patch_size - valid_gt_labels.shape[0] % patch_size
    pad1 = patch_size - valid_gt_labels.shape[1] % patch_size
    valid_gt_labels = np.pad(valid_gt_labels, [(0, pad0), (0, pad1)], constant_values=0)

    if cfg.use_valid_mask:
        valid_masks = cv2.imread(os.path.join(cfg.data_root, f"train/{valid_id}/mask.png"), 0)
        valid_masks = valid_masks > 0
        pad0 = patch_size - valid_masks.shape[0] % patch_size
        pad1 = patch_size - valid_masks.shape[1] % patch_size
        valid_masks = np.pad(valid_masks, [(0, pad0), (0, pad1)], constant_values=0)
    else:
        valid_masks = np.ones_like(valid_gt_labels, dtype=bool)

    best_score: float = -1.0
    best_loss = np.inf

    if not args.debug:
        writer = SummaryWriter(log_dir=os.path.join(work_dir, "tf_logs"))
    else:
        writer = None

    use_mix_augmentation = cfg.mix_augmentation_p > 0
    total_step: int = 0
    best_pred = None
    LAST_SAVE_EPOCHS: int = 10
    for epoch in range(cfg.epochs):
        start_time = time.time()

        if (
            use_mix_augmentation
            and cfg.disable_mixup_last_epoch is not None
            and (epoch >= cfg.epochs - cfg.disable_mixup_last_epoch)
        ):
            use_mix_augmentation = False
            logger.info(f"Disable Mixup Augmentation! (epoch={epoch})")

        # train
        avg_loss, total_step = train_fn(
            cfg=cfg,
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            optional_criterion=optional_criterion,
            optimizer=optimizer,
            device=device,
            writer=writer,
            total_step=total_step,
            mix_augmentation_p=cfg.mix_augmentation_p if use_mix_augmentation else 0,
            mix_augmentation_alpha=cfg.mix_augmentation_alpha,
            use_manifold_mixup=cfg.use_manifold_mixup,
            debug=args.debug,
        )

        # eval
        avg_val_loss, labels_pred, rots_gt_ret, rots_pd_ret = valid_fn(
            cfg=cfg,
            dataloader=valid_dataloader,
            model=model,
            criterion=criterion,
            optional_criterion=optional_criterion,
            device=device,
            valid_xyxys=valid_xyxys,
            valid_gt_labels=valid_gt_labels,
            debug=args.debug,
        )

        scheduler.step(epoch)

        if writer is not None:
            lr = scheduler.get_lr()
            writer.add_scalar("learning_rate", lr[0], total_step)
            if len(lr) > 1:
                for i in range(1, len(lr)):
                    writer.add_scalar(f"learning_rate_{i}", lr[i], total_step)

        logger.info("Evaluation - Semantic Segmentation")
        best_metric_dict, _ = calc_fbeta(valid_gt_labels, labels_pred, valid_masks)

        if len(rots_gt_ret) > 0 and len(rots_pd_ret) > 0:
            logger.info("Evaluation - Rotation Classification")
            rot_best_metric_dict, _ = calc_fbeta(rots_gt_ret, rots_pd_ret, beta=1.0, show_confusion_matrix=True)
        else:
            rot_best_metric_dict = dict()

        elapsed = time.time() - start_time

        score = best_metric_dict["fscore"]
        if writer is not None:
            writer.add_scalar("val/loss_val", avg_val_loss, total_step)
            writer.add_scalar("val/Fscore", best_metric_dict["fscore"], total_step)
            writer.add_scalar("val/Precision", best_metric_dict["precision"], total_step)
            writer.add_scalar("val/Recall", best_metric_dict["recall"], total_step)
            writer.add_scalar("val/Threshold", best_metric_dict["threshold"], total_step)

            if rot_best_metric_dict:
                writer.add_scalar("val/Fscore_rot", rot_best_metric_dict["fscore"], total_step)
                writer.add_scalar("val/Precision_rot", rot_best_metric_dict["precision"], total_step)
                writer.add_scalar("val/Recall_rot", rot_best_metric_dict["recall"], total_step)
                writer.add_scalar("val/Threshold_rot", rot_best_metric_dict["threshold"], total_step)

        epoch_plus_1: int = epoch + 1
        logger.info(
            f"Epoch {epoch_plus_1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        logger.info(f"Epoch {epoch_plus_1} - avgScore: {score:.4f}")

        if score > best_score:
            best_loss = avg_val_loss
            best_score = score
            best_pred = labels_pred

            logger.info(f"Epoch {epoch_plus_1} - Save Best Score: {best_score:.4f} Model")
            logger.info(f"Epoch {epoch_plus_1} - Save Best Loss: {best_loss:.4f} Model")

            if not args.debug:
                _model = model.module if is_module(model) else model
                torch.save(
                    {"state_dict": _model.state_dict()},
                    os.path.join(work_dir, f"{cfg.model_name}_fold{valid_id}_best.pth"),
                )

        if not args.debug and epoch_plus_1 > cfg.epochs - LAST_SAVE_EPOCHS:
            _model = model.module if is_module(model) else model
            torch.save(
                {"state_dict": _model.state_dict()},
                os.path.join(work_dir, f"{cfg.model_name}_fold{valid_id}_epoch{epoch_plus_1}.pth"),
            )

    best_metric_dict, _ = calc_fbeta(valid_gt_labels, best_pred, valid_masks)
    best_threshold = best_metric_dict["threshold"]
    best_fscore = best_metric_dict["fscore"]
    best_precision = best_metric_dict["precision"]
    best_recall = best_metric_dict["recall"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    img = np.zeros((*best_pred.shape, 3), dtype=np.uint8)
    img[np.logical_and(valid_gt_labels == 1, (best_pred >= best_threshold) == 1), 0] = 255  # True Positive
    img[np.logical_and(valid_gt_labels == 0, (best_pred >= best_threshold) == 1), 1] = 255  # False Positive
    img[np.logical_and(valid_gt_labels == 1, (best_pred >= best_threshold) == 0), 2] = 255  # False Negative
    best_pred[0, 0] = 0
    best_pred[0, 1] = 1

    axes[0].imshow(valid_gt_labels)
    axes[1].imshow(img)
    axes[1].set_title(
        f"pred labels: TP=red FP=greed FN=blue\n"
        + f"Fscore={best_fscore:.3f} Precision={best_precision:.3f} "
        + f"Recall={best_recall:.3f}\nthreshold={best_threshold:.2f}"
    )
    axes[2].imshow(best_pred)

    plt.savefig(os.path.join(work_dir, "visualization.png"))


if __name__ == "__main__":
    main()

import argparse
import dataclasses
import datetime as dt
import math
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from inkdet.datasets.utils import build_dataloader, get_train_valid_ids
from inkdet.models import build_criterion, build_model, build_optimizer_scheduler
from inkdet.utils import AverageMeter, Config, DictAction, calc_fbeta, get_git_hash, load_checkpoint, set_seed


def train_fn(
    cfg: Config,
    train_loader,
    model,
    criterion,
    optimizer,
    device,
    writer: SummaryWriter,
    total_step: int,
):
    model.train()

    scaler = GradScaler(enabled=cfg.use_amp)
    losses = AverageMeter()
    grad_norms = AverageMeter()

    for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images: torch.Tensor = images.to(device)
        labels: torch.Tensor = labels.to(device)
        batch_size = labels.size(0)

        if images.ndim == 4:
            images = images.unsqueeze(1)

        with autocast(enabled=cfg.use_amp):
            y_preds = model(images)
            loss: torch.Tensor = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        if cfg.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            grad_norms.update(float(grad_norm), batch_size)

        if total_step % cfg.log_interval == 0:
            writer.add_scalar("train/grad_norm", grad_norms.avg, total_step)
            writer.add_scalar("train/loss", losses.avg, total_step)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_step += 1

    return losses.avg, total_step


def valid_fn(dataloader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    gt_labels = []
    pred_labels = []

    dataloader.dataset.training = False

    for images, _ in tqdm(dataloader, total=len(dataloader)):
        images = images.to(device)
        batch_size = images.size(0)

        if images.ndim == 4:
            images = images.unsqueeze(1)

        for rot in range(2):
            images = torch.rot90(images, k=rot, dims=(-2, -1))
            labels = rot * torch.ones((batch_size, 1), dtype=torch.float32).to(device)
            with torch.no_grad():
                y_preds = model(images)
                loss = criterion(y_preds, labels)

            gt_labels.append(labels.cpu().numpy())
            pred_labels.append(y_preds.cpu().numpy())

        losses.update(loss.item(), batch_size)

    gt_labels = np.concatenate(gt_labels)
    pred_labels = np.concatenate(pred_labels)

    print(gt_labels.shape, pred_labels.shape)

    return losses.avg, gt_labels, pred_labels


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--valid-id", type=str, default="1", choices=["1", "2", "3", "2a", "2b"])
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
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
    # torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    exp_name: str = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_classification_{config_path.stem}_{cfg.model_name}"
    work_dir = Path(f"/opt/kaggle-ink-detection/work_dirs/{cfg.competition_name}/{exp_name}_{valid_id}/")

    logger.add(work_dir / "log.txt")
    logger.info(f"WORK_DIR: {work_dir}")
    logger.info(f"git commit hash: {get_git_hash(digits=7)}")
    logger.info(f"CFG: \n {yaml.dump(dataclasses.asdict(cfg), indent=2, sort_keys=False)}")
    cfg.dump(work_dir / config_path.name)

    model = build_model(cfg)
    model.to(device)
    # model = torch.compile(model)
    logger.info(f"model: {cfg.model_name}")
    logger.info(f"encoder: {cfg.encoder_name}")
    logger.info(f"{model}")

    criterion = build_criterion(cfg.losses)
    optimizer, scheduler = build_optimizer_scheduler(model.parameters(), cfg)
    logger.info(f"optimizer: {optimizer}")
    logger.info(f"scheduler: {scheduler}")

    train_ids, valid_ids = get_train_valid_ids(valid_id)
    train_dataloader, _ = build_dataloader("train", train_ids, cfg)
    valid_dataloader, _ = build_dataloader("val", valid_ids, cfg)
    logger.info(f"train ids: {train_ids}")
    logger.info(f"train dataset size: {len(train_dataloader.dataset)}")
    logger.info(f"valid ids: {valid_ids}")
    logger.info(f"valid dataset size: {len(valid_dataloader.dataset)}")

    best_score: float = -1.0
    best_loss = np.inf

    writer = SummaryWriter(log_dir=os.path.join(work_dir, "tf_logs"))
    total_step: int = 0
    best_pred = None
    LAST_SAVE_EPOCHS: int = 10
    for epoch in range(cfg.epochs):
        start_time = time.time()

        # train
        avg_loss, total_step = train_fn(
            cfg,
            train_dataloader,
            model,
            criterion,
            optimizer,
            device,
            writer,
            total_step,
        )

        # eval
        avg_val_loss, gt_labels, pred_labels = valid_fn(
            valid_dataloader,
            model,
            criterion,
            device,
        )

        scheduler.step(epoch)

        lr = scheduler.get_lr()
        writer.add_scalar("learning_rate", lr[0], total_step)
        if len(lr) > 1:
            for i in range(1, len(lr)):
                writer.add_scalar(f"learning_rate_{i}", lr[i], total_step)

        elapsed = time.time() - start_time

        best_metric_dict, _ = calc_fbeta(gt_labels, pred_labels, beta=1.0)

        score = best_metric_dict["fscore"]
        writer.add_scalar("val/loss_val", avg_val_loss, total_step)
        writer.add_scalar("val/Fscore", best_metric_dict["fscore"], total_step)
        writer.add_scalar("val/Precision", best_metric_dict["precision"], total_step)
        writer.add_scalar("val/Recall", best_metric_dict["recall"], total_step)
        writer.add_scalar("val/Threshold", best_metric_dict["threshold"], total_step)

        epoch_plus_1: int = epoch + 1
        logger.info(
            f"Epoch {epoch_plus_1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        logger.info(f"Epoch {epoch_plus_1} - avgScore: {score:.4f}")

        if score > best_score:
            best_loss = avg_val_loss
            best_score = score
            best_pred = pred_labels

            logger.info(f"Epoch {epoch_plus_1} - Save Best Score: {best_score:.4f} Model")
            logger.info(f"Epoch {epoch_plus_1} - Save Best Loss: {best_loss:.4f} Model")

            torch.save(
                {"state_dict": model.state_dict()},
                os.path.join(work_dir, f"{cfg.model_name}_fold{valid_id}_best.pth"),
            )

        if epoch_plus_1 > cfg.epochs - LAST_SAVE_EPOCHS:
            torch.save(
                {"state_dict": model.state_dict()},
                os.path.join(work_dir, f"{cfg.model_name}_fold{valid_id}_epoch{epoch_plus_1}.pth"),
            )

    best_metric_dict, _ = calc_fbeta(valid_gt_labels, best_pred, valid_masks)


if __name__ == "__main__":
    main()

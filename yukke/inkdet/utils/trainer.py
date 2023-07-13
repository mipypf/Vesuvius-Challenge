from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from inkdet.datasets.utils import build_dataloader, get_train_valid_ids
from inkdet.models import build_criterion, build_model, build_optimizer_scheduler
from inkdet.utils import AverageMeter, Config, DictAction, calc_fbeta, get_git_hash, load_checkpoint, set_seed


class Trainer:
    def __init__(self, cfg: Config, valid_id: str, mode: str):
        pass

        self.cfg = cfg
        self.valid_id = valid_id
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # train
        self.total_steps: int = 0
        self.train_dataloader = None
        self.criterion: Optional[Callable] = None
        self.optimizer = None
        self.scheduler = None

        self._init()
        if mode == "train":
            self._init_train()
        elif mode == "val":
            pass
        else:
            pass

    def _init(self):
        cfg = self.cfg

        model = build_model(cfg)
        model.to(self.device)
        # model = torch.compile(model)
        logger.info(f"model: {cfg.model_name}")
        logger.info(f"encoder: {cfg.encoder_name}")
        logger.info(f"{model}")

    def _init_train(self):
        cfg = self.cfg

        if cfg.pretrained_path:
            load_checkpoint(self.model, cfg.pretrained_path)

        self.optimizer, self.scheduler = build_optimizer_scheduler(self.model.parameters(), cfg)
        logger.info(f"optimizer: {self.optimizer}")
        logger.info(f"scheduler: {self.scheduler}")

        self.criterion = build_criterion(cfg.losses)

        train_ids, valid_ids = get_train_valid_ids(self.valid_id)
        self.train_dataloader, _ = build_dataloader("train", train_ids, cfg)
        self.valid_dataloader, self.valid_xyxys = build_dataloader("val", valid_ids, cfg)
        logger.info(f"train ids: {train_ids}")
        logger.info(f"train dataset size: {len(self.train_dataloader.dataset)}")
        logger.info(f"valid ids: {valid_ids}")
        logger.info(f"valid dataset size: {len(self.valid_dataloader.dataset)}")

        self.writer = SummaryWriter(log_dir=os.path.join(work_dir, "tf_logs"))

    def train_one_epoch(self, epoch: int):
        self.model.train()

        model = self.model
        writer = self.writer
        cfg = self.cfg

        assert self.train_dataloader is not None
        dataloader = self.train_dataloader

        scaler = GradScaler(enabled=cfg.use_amp)
        losses = AverageMeter()
        grad_norms = AverageMeter()

        for _, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            images: torch.Tensor = images.to(self.device)
            labels: torch.Tensor = labels.to(self.device)
            batch_size = labels.size(0)

            with autocast(enabled=cfg.use_amp):
                y_preds = self.model(images)
                loss: torch.Tensor = self.criterion(y_preds, labels)

            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()

            if self.cfg.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                grad_norms.update(float(grad_norm), batch_size)

            if self.total_steps % cfg.log_interval == 0:
                writer.add_scalar("train/grad_norm", grad_norms.avg, self.total_steps)
                writer.add_scalar("train/loss", losses.avg, self.total_steps)

            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

            self.total_steps += 1

        return losses.avg, self.total_steps

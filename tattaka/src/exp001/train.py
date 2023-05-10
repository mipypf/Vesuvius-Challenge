import argparse
import datetime
import math
import os
import warnings
from functools import lru_cache
from glob import glob
from typing import List, Tuple

import albumentations as albu
import numpy as np
import PIL.Image as Image
import pytorch_lightning as pl
import timm
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule, callbacks
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_info
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

EXP_ID = "001"
COMMENT = "2.5D baseline + fragment_wise norm"
BASE_DIR = "../../input/vesuvius_patches_32"

NORM_PARAM = []
for i in tqdm([1, 2, 3]):
    means = []
    stds = []
    filenames = sorted(
        glob(
            f"../../input/vesuvius-challenge-ink-detection/train/{i}/surface_volume/*.tif"
        )
    )
    for f in filenames:
        img = np.array(Image.open(f), dtype=np.float32) / 65535.0
        means.append(img.mean())
        stds.append(img.std() + 1e-6)
    NORM_PARAM.append(np.stack([np.asarray(means), np.asarray(stds)]))
NORM_PARAM = np.stack(NORM_PARAM).astype(np.float32)  # (3, 2, 65)


def get_transforms(train: bool = False):
    if train:
        return albu.Compose(
            [
                albu.Flip(p=0.5),
                albu.RandomRotate90(p=0.5),
                albu.Transpose(p=0.5),
                albu.ShiftScaleRotate(p=0.5),
                # A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                albu.Normalize(mean=[0.0], std=[1.0]),
                # albu.ChannelShuffle(p=0.5),
                ToTensorV2(),
            ]
        )

    else:
        return albu.Compose(
            [
                albu.Normalize(mean=[0.0], std=[1.0]),
                ToTensorV2(),
            ]
        )


class PatchDataset(Dataset):
    def __init__(
        self,
        volume_paths: List[str],
        image_size: Tuple[int, int] = (256, 256),
        train: bool = True,
    ):
        self.volume_paths = volume_paths
        self.image_size = image_size
        assert (image_size[0] % 32 == 0) and (image_size[1] % 32 == 0)
        self.train = train
        self.transforms = get_transforms(train)
        self.PATCH_SIZE = 32

    def __len__(self):
        return len(self.volume_paths)

    @lru_cache(maxsize=1024)
    def np_load(self, path: str):
        return np.load(path)

    def __getitem__(self, idx: int):
        if self.train:
            np_load = np.load
        else:
            np_load = self.np_load
        volume = np.ones((65, *self.image_size), dtype=np.float32)
        label = np.zeros(self.image_size)
        if self.train:
            idx = np.random.choice(np.arange(len(self.volume_paths)))
        volume_lt_path = self.volume_paths[idx]
        data_prefix = "/".join(volume_lt_path.split("/")[:-3])
        data_source = int(volume_lt_path.split("/")[-3])
        volume *= NORM_PARAM[data_source - 1, 0, :, None, None]
        y, x = volume_lt_path.split("/")[-1].split(".")[-2].split("_")[-2:]
        x = int(x)
        y = int(y)
        for i in range(self.image_size[0] // self.PATCH_SIZE):
            for j in range(self.image_size[1] // self.PATCH_SIZE):
                volume_path = os.path.join(
                    data_prefix,
                    str(data_source),
                    f"surface_volume/volume_{y + i}_{x + j}.npy",
                )
                label_path = os.path.join(
                    data_prefix, str(data_source), f"label/label_{y + i}_{x + j}.npy"
                )
                if os.path.exists(volume_path):
                    volume[
                        :,
                        i * self.PATCH_SIZE : (i + 1) * self.PATCH_SIZE,
                        j * self.PATCH_SIZE : (j + 1) * self.PATCH_SIZE,
                    ] = np_load(volume_path)
                    if os.path.exists(label_path):
                        label[
                            i * self.PATCH_SIZE : (i + 1) * self.PATCH_SIZE,
                            j * self.PATCH_SIZE : (j + 1) * self.PATCH_SIZE,
                        ] = np_load(label_path)
        volume = volume.transpose(1, 2, 0)
        aug = self.transforms(image=volume, mask=label)
        volume = aug["image"]
        volume = (volume - NORM_PARAM[data_source - 1, 0, :, None, None]) / NORM_PARAM[
            data_source - 1, 1, :, None, None
        ]
        label = aug["mask"][None, :]
        if np.random.rand() > 0.5:
            volume = volume.flip(0)
        return (
            volume,
            label,
            x,
            y,
            data_source,
        )


class InkDetDataModule(LightningDataModule):
    def __init__(
        self,
        train_volume_paths: List[str],
        valid_volume_paths: List[str],
        image_size: int = 256,
        num_workers: int = 4,
        batch_size: int = 16,
    ):
        super().__init__()

        self._num_workers = num_workers
        self._batch_size = batch_size
        self.train_volume_paths = train_volume_paths
        self.valid_volume_paths = valid_volume_paths
        self.image_size = (image_size, image_size)
        self.save_hyperparameters("num_workers", "batch_size", "image_size")

    def create_dataset(self, train: bool = True):
        return (
            PatchDataset(
                volume_paths=self.train_volume_paths,
                image_size=self.image_size,
                train=True,
            )
            if train
            else PatchDataset(
                volume_paths=self.valid_volume_paths,
                image_size=self.image_size,
                train=False,
            )
        )

    def __dataloader(self, train: bool = True):
        """Train/validation loaders."""
        dataset = self.create_dataset(train)
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=train,
            drop_last=train,
            worker_init_fn=lambda x: np.random.seed(np.random.get_state()[1][0] + x),
            pin_memory=True,
        )

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("InkDetDataModule")
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            metavar="W",
            help="number of CPU workers",
            dest="num_workers",
        )
        parser.add_argument(
            "--batch_size",
            default=16,
            type=int,
            metavar="BS",
            help="number of sample in a batch",
            dest="batch_size",
        )
        parser.add_argument(
            "--image_size",
            default=256,
            type=int,
            metavar="IS",
            help="image size",
            dest="image_size",
        )

        return parent_parser


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class InkDetModel(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet34",
        pretrained: bool = False,
        drop_rate: float = 0.5,
        num_class: int = 1,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=65,
            features_only=True,
            drop_rate=drop_rate,
        )
        channels = self.encoder.feature_info.channels()
        self.head = nn.Sequential(
            nn.Conv2d(
                channels[-1],
                num_class,
                1,
            ),
        )

    def forward(
        self,
        img: torch.Tensor,
    ):
        """
        img: (bs, ch, h, w)
        """
        return self.head(self.encoder(img)[-1])


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


def dice(mask1, mask2, use_mask):
    mask1 = mask1 * use_mask
    mask2 = mask2 * use_mask
    intersection = 2.0 * (mask1 * mask2).sum()
    union = mask1.sum() + mask2.sum()
    if mask1.sum() == 0 and mask2.sum() == 0:
        return 1.0

    return intersection / union


def find_threshold(p_valid, y_valid, use_valid):
    best = 0
    best_score = -2
    totry = np.arange(0, 1, 0.01)
    for t in totry:
        score = dice(y_valid, p_valid > t, use_valid)
        if score > best_score:
            best_score = score
            best = t
    return best


class InkDetLightningModel(pl.LightningModule):
    def __init__(
        self,
        valid_fragment_id: int,
        model_name: str = "resnet34",
        pretrained: bool = False,
        mixup_p: float = 0.0,
        mixup_alpha: float = 0.5,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.__build_model(
            model_name,
            pretrained,
        )
        self.mixupper = Mixup(p=mixup_p, alpha=mixup_alpha)
        self.y_valid = np.array(
            Image.open(
                f"../../input/vesuvius-challenge-ink-detection/train/{valid_fragment_id}/inklabels.png"
            ).convert("1")
        )
        self.p_valid = np.zeros_like(self.y_valid, dtype=np.float32)
        self.count_pix = np.zeros_like(self.y_valid, dtype=np.float32)
        self.save_hyperparameters()

    def __build_model(
        self, model_name: str = "resnet34", pretrained: bool = False, num_class: int = 1
    ):
        self.model = InkDetModel(
            model_name,
            pretrained,
            num_class,
        )
        self.criterions = {"bce": QFocalLoss(nn.BCEWithLogitsLoss())}

    def calc_loss(self, outputs, labels):
        losses = {}
        losses["bce"] = self.criterions["bce"](outputs["logits"], labels["targets"])
        losses["loss"] = losses["bce"]
        return losses

    def training_step(self, batch, batch_idx):
        step_output = {}
        outputs = {}
        loss_target = {}
        self.mixupper.init_lambda()
        volume, label, _, _, _ = batch
        loss_target["targets"] = F.interpolate(
            label, scale_factor=1 / 32, mode="bilinear", align_corners=True
        )
        if self.mixupper.do_mixup:
            volume = self.mixupper.lam * volume + (1 - self.mixupper.lam) * volume.flip(
                0
            )
        outputs["logits"] = self.model(volume)
        losses = self.calc_loss(outputs, loss_target)

        if self.mixupper.do_mixup:
            loss_target["targets"] = loss_target["targets"].flip(0)
            losses_b = self.calc_loss(outputs, loss_target)
            for key in losses:
                losses[key] = (
                    self.mixupper.lam * losses[key]
                    + (1 - self.mixupper.lam) * losses_b[key]
                )
        losses = self.calc_loss(outputs, loss_target)
        step_output.update(losses)
        self.log_dict(
            dict(
                train_loss=losses["loss"],
                train_bce_loss=losses["bce"],
            )
        )
        return step_output

    def validation_step(self, batch, batch_idx):
        step_output = {}
        outputs = {}
        loss_target = {}

        volume, label, x, y, _ = batch
        loss_target["targets"] = F.interpolate(
            label, scale_factor=1 / 32, mode="bilinear", align_corners=True
        )
        outputs["logits"] = self.model(volume)
        losses = self.calc_loss(outputs, loss_target)
        step_output.update(losses)
        pred_batch = F.interpolate(
            torch.sigmoid(outputs["logits"]).detach().to(torch.float32).cpu(),
            scale_factor=32,
            mode="bilinear",
            align_corners=True,
        ).numpy()
        for xi, yi, pi in zip(
            x,
            y,
            pred_batch,
        ):
            y_lim, x_lim = self.y_valid[
                yi * 32 : yi * 32 + volume.shape[-2],
                xi * 32 : xi * 32 + volume.shape[-1],
            ].shape
            self.p_valid[
                yi * 32 : yi * 32 + volume.shape[-2],
                xi * 32 : xi * 32 + volume.shape[-1],
            ] += pi[0, :y_lim, :x_lim]
            self.count_pix[
                yi * 32 : yi * 32 + volume.shape[-2],
                xi * 32 : xi * 32 + volume.shape[-1],
            ] += np.ones_like(pi[0, :y_lim, :x_lim])
        self.log_dict(
            dict(
                val_loss=losses["loss"],
                val_bce_loss=losses["bce"],
            )
        )
        return step_output

    def on_validation_epoch_end(self):
        self.p_valid /= self.count_pix
        self.p_valid = np.nan_to_num(self.p_valid)
        self.count_pix = self.count_pix > 0
        threshold = find_threshold(self.p_valid, self.y_valid, self.count_pix)
        self.p_valid = self.p_valid > threshold
        score = dice(self.p_valid, self.y_valid, self.count_pix)
        self.p_valid = np.zeros_like(self.p_valid, dtype=np.float32)
        self.count_pix = np.zeros_like(self.count_pix, dtype=np.float32)
        self.log_dict(dict(val_dice_score=score, val_threshold=threshold))

    def get_optimizer_parameters(self):
        no_decay = ["bias", "gamma", "beta"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
                "lr": self.lr,
            },
        ]
        return optimizer_parameters

    def configure_optimizers(self):
        self.warmup = True
        optimizer = AdamW(self.get_optimizer_parameters())
        max_train_steps = self.trainer.estimated_stepping_batches
        warmup_steps = math.ceil((max_train_steps * 2) / 100) if self.warmup else 0
        rank_zero_info(
            f"max_train_steps: {max_train_steps}, warmup_steps: {warmup_steps}"
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("G2NetLightningModel")
        parser.add_argument(
            "--model_name",
            default="resnet34",
            type=str,
            metavar="MN",
            help="Name (as in ``timm``) of the feature extractor",
            dest="model_name",
        )
        parser.add_argument(
            "--mixup_p", default=0.0, type=float, metavar="MP", dest="mixup_p"
        )
        parser.add_argument(
            "--mixup_alpha", default=1.0, type=float, metavar="MA", dest="mixup_alpha"
        )
        parser.add_argument(
            "--lr",
            default=1e-3,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )

        return parent_parser


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--seed",
        default=2022,
        type=int,
        metavar="SE",
        help="seed number",
        dest="seed",
    )
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="1 batch run for debug",
        dest="debug",
    )
    dt_now = datetime.datetime.now()
    parent_parser.add_argument(
        "--logdir",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parent_parser.add_argument(
        "--gpus", type=int, default=4, help="number of gpus to use"
    )
    parent_parser.add_argument(
        "--epochs", default=10, type=int, metavar="N", help="total number of epochs"
    )
    parser = InkDetLightningModel.add_model_specific_args(parent_parser)
    parser = InkDetDataModule.add_model_specific_args(parser)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed)
    if not args.debug:
        warnings.simplefilter("ignore")
    fragment_ids = [1, 2, 3]
    for i, valid_idx in enumerate(fragment_ids):
        if args.fold != i:
            continue
        train_volume_paths = np.concatenate(
            [
                np.asarray(
                    sorted(
                        glob(
                            f"../../input/vesuvius_patches_32/train/{fragment_id}/surface_volume/**/*.npy",
                            recursive=True,
                        )
                    )
                )
                for fragment_id in fragment_ids
                if fragment_id != valid_idx
            ]
        )

        valid_volume_paths = np.concatenate(
            [
                np.asarray(
                    sorted(
                        glob(
                            f"../../input/vesuvius_patches_32/train/{fragment_id}/surface_volume/**/*.npy",
                            recursive=True,
                        )
                    )
                )
                for fragment_id in fragment_ids
                if fragment_id == valid_idx
            ]
        )

        datamodule = InkDetDataModule(
            train_volume_paths=train_volume_paths,
            valid_volume_paths=valid_volume_paths,
            image_size=args.image_size,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        model = InkDetLightningModel(
            valid_fragment_id=valid_idx,
            model_name=args.model_name,
            pretrained=True,
            mixup_p=args.mixup_p,
            mixup_alpha=args.mixup_alpha,
            lr=args.lr,
        )

        logdir = f"../../logs/exp{EXP_ID}/{args.logdir}/fold{i}"
        print(f"logdir = {logdir}")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
            mode="min",
        )
        mcc_checkpoint = callbacks.ModelCheckpoint(
            filename="best_dice",
            monitor="val_dice_score",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
            mode="max",
        )
        os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)
        if not args.debug:
            wandb_logger = WandbLogger(
                name=f"exp{EXP_ID}/{args.logdir}_fold{i}",
                save_dir=logdir,
                project="vesuvius-challenge-ink-detection",
            )

        trainer = pl.Trainer(
            default_root_dir=logdir,
            sync_batchnorm=True,
            gradient_clip_val=0.5,
            precision=16,
            devices=args.gpus,
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_false",
            max_epochs=args.epochs,
            logger=wandb_logger if not args.debug else True,
            callbacks=[
                loss_checkpoint,
                mcc_checkpoint,
                lr_monitor,
            ],
            fast_dev_run=args.debug,
            num_sanity_val_steps=0,
        )

        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main(get_args())
import ssl
from typing import Callable, Dict, List, Optional, Sequence

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from inkdet.models import resnet3d
from inkdet.models.encoder_decoder import Enc3dDec2d, EncDec
from inkdet.utils.config import Config

ssl._create_default_https_context = ssl._create_unverified_context


def initialize_head(module: nn.Module, init_weights: Optional[str], bias: Optional[float]):
    if init_weights is None and bias is None:
        return

    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if init_weights is not None:
                getattr(nn.init, init_weights)(m.weight)
            if m.bias is not None and bias is not None:
                nn.init.constant_(m.bias, bias)


def build_segmentation_model(cfg: Config) -> nn.Module:
    if cfg.model_name == "Enc3dDec2d":
        model = Enc3dDec2d(
            encoder_name=cfg.encoder_name,
            classes=cfg.num_classes,
            checkpoint_path=cfg.checkpoint_path,
            **cfg.model_params,
        )
        initialize_head(model.decoder.logit, cfg.head_init_weights, cfg.head_init_bias)
        logger.info(
            f"Init model.decoder.logit {model.decoder.logit}: weight={cfg.head_init_weights} bias={cfg.head_init_bias}"
        )

    elif cfg.model_name == "EncDec":
        model = EncDec(
            encoder_name=cfg.encoder_name,
            classes=cfg.num_classes,
            checkpoint_path=cfg.checkpoint_path,
            **cfg.model_params,
        )
        initialize_head(model.decoder.logit, cfg.head_init_weights, cfg.head_init_bias)
        logger.info(
            f"Init model.decoder.logit {model.decoder.logit}: weight={cfg.head_init_weights} bias={cfg.head_init_bias}"
        )
        if hasattr(model.encoder, "head"):
            logger.info(f"Init model.encoder.head {model.encoder.head}: weight=trunc_normal_(std=0.01) bias=0.0")
            for m in model.encoder.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
        if hasattr(model.encoder, "fc"):
            logger.info(f"Init model.encoder.fc {model.encoder.fc}: weight=trunc_normal_(std=0.01) bias=0.0")
            for m in model.encoder.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
    else:
        model: nn.Module = getattr(smp, cfg.model_name)(
            encoder_name=cfg.encoder_name,
            in_channels=cfg.in_channels,
            classes=cfg.num_classes,
            **cfg.model_params,
        )
        initialize_head(model.segmentation_head, cfg.head_init_weights, cfg.head_init_bias)
        logger.info(
            f"Init model.segmentation_head {model.segmentation_head}: weight={cfg.head_init_weights} bias={cfg.head_init_bias}"
        )

    return model


def build_classification_model(cfg: Config) -> nn.Module:
    if cfg.model_name == "resnet3d":
        model = resnet3d.generate_model(
            model_name=cfg.encoder_name,
            model_depth=cfg.model_params["encoder_depth"],
            no_max_pool=cfg.model_params["encoder_no_max_pool"],
            n_input_channels=1,
            n_classes=1,
        )

        def load_checkpoint(checkpoint_path: str):
            if cfg.encoder_name == "resnet":
                the_1st_layer_key = "conv1.weight"
            elif cfg.encoder_name == "resnet2p1d":
                the_1st_layer_key = "conv1_s.weight"
            else:
                raise ValueError(f"Pre-trained Model is NOT Available for {cfg.encoder_name}")

            # Convert 3 channel weights to single channel
            # ref: https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1
            state_dict = torch.load(checkpoint_path)["state_dict"]
            conv1_weight = state_dict[the_1st_layer_key]
            state_dict[the_1st_layer_key] = conv1_weight.sum(dim=1, keepdim=True)
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")
            logger.info(f"Load checkpoint {model.load_state_dict(state_dict, strict=False)}")

        if cfg.checkpoint_path:
            load_checkpoint(cfg.checkpoint_path)

        initialize_head(model.fc, cfg.head_init_weights, cfg.head_init_bias)
    else:
        raise ValueError(f"No Model Name: {cfg.model_name}")

    return model


def build_model(cfg: Config) -> nn.Module:
    if cfg.task in ["segmentation", "segmentation-classification"]:
        return build_segmentation_model(cfg)
    elif cfg.task == "classification":
        return build_classification_model(cfg)
    else:
        raise ValueError(f"Unknown task: {cfg.task}")


class Losses(nn.Module):
    def __init__(self, criterions: List[Callable], weights: List[float]):
        super().__init__()
        assert len(criterions) == len(weights)

        self.criterions = nn.ModuleList(criterions)
        self.weights = weights

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        for weight, criterion in zip(self.weights, self.criterions):
            name = criterion.__class__.__name__.lower()
            loss_dict[name] = weight * criterion(y_pred, y_true)

        return loss_dict

    def extra_repr(self) -> str:
        _repr = f"weights: {self.weights}"
        return _repr


def build_criterion(loss_dicts: Sequence[dict]) -> nn.Module:
    from inkdet.models import losses

    assert isinstance(loss_dicts, Sequence) and len(loss_dicts) > 0

    criterions: List[Callable] = []
    weights: List[float] = []
    for i, loss in enumerate(loss_dicts):
        assert "type" in loss, f"type isn't available: {loss}"

        loss_type: str = loss.pop("type")
        params = loss.copy()
        weight = params.pop("weight", 1.0)
        if "pos_weight" in params:
            params["pos_weight"] = torch.tensor(params["pos_weight"])

        if hasattr(nn, loss_type):
            c = getattr(nn, loss_type)(**params)
        elif hasattr(smp.losses, loss_type):
            c = getattr(smp.losses, loss_type)(**params)
        elif hasattr(losses, loss_type):
            c = getattr(losses, loss_type)(**params)
        else:
            raise ValueError(f"Not Found: {loss_type}")

        criterions.append(c)
        weights.append(weight)

    return Losses(criterions, weights)


def build_optimizer_scheduler(model_params, cfg: Config):
    from inkdet.models.schedulers import GradualWarmupSchedulerV2

    if cfg.scheduler_name == "GradualWarmupSchedulerV2":
        optimizer = getattr(optim, cfg.optimizer_name)(
            model_params,
            lr=cfg.max_lr / cfg.warmup_factor,
            **cfg.optimizer_params,
        )
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            cfg.epochs,
            eta_min=1e-7,
        )
        scheduler = GradualWarmupSchedulerV2(
            optimizer,
            multiplier=10,
            total_epoch=1,
            after_scheduler=scheduler_cosine,
        )
    elif cfg.scheduler_name == "OneCycleLR":
        optimizer = getattr(optim, cfg.optimizer_name)(
            model_params,
            lr=cfg.max_lr,
            **cfg.optimizer_params,
        )
        # NOTE: one step per epoch
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.max_lr,
            total_steps=cfg.epochs,
            **cfg.scheduler_params,
        )
    else:
        optimizer = getattr(optim, cfg.optimizer_name)(
            model_params,
            lr=cfg.max_lr,
            **cfg.optimizer_params,
        )
        scheduler = getattr(optim.lr_scheduler, cfg.scheduler_name)(
            optimizer,
            **cfg.scheduler_params,
        )

    return optimizer, scheduler

# modified from https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py
from typing import Any, Callable, List, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
from loguru import logger
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.video.resnet import (
    BasicBlock,
    BasicStem,
    Bottleneck,
    Conv2Plus1D,
    Conv3DNoTemporal,
    Conv3DSimple,
    MC3_18_Weights,
    R2Plus1D_18_Weights,
    R2Plus1dStem,
    R3D_18_Weights,
)
from torchvision.models.video.resnet import VideoResNet as _VideoResNet


class VideoResNet(_VideoResNet):
    def __init__(self, **kwargs):
        self.in_channels = kwargs.pop("in_channels")
        self.num_classes = kwargs.get("num_classes")
        if "num_classes" in kwargs and kwargs["num_classes"] is None:
            kwargs.pop("num_classes")
        super().__init__(**kwargs)

        self.with_classification = self.num_classes is not None

        if isinstance(self.stem, BasicStem):
            self.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        elif isinstance(self.stem, R2Plus1dStem):
            self.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        else:
            raise ValueError(f"Unknown stem type: {type(self.stem)}")

        if not self.with_classification:
            del self.avgpool
            del self.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]

    def load_checkpoint(self, model_name: str):
        if model_name == "r3d_18":
            weights = R3D_18_Weights.verify("KINETICS400_V1")
        elif model_name == "mc3_18":
            weights = MC3_18_Weights.verify("KINETICS400_V1")
        elif model_name == "r2plus1d_18":
            weights = R2Plus1D_18_Weights.verify("KINETICS400_V1")
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        state_dict = weights.get_state_dict(progress=True)

        if self.in_channels == 1:
            patch_embed_weight = state_dict["stem.0.weight"]
            state_dict["stem.0.weight"] = patch_embed_weight.sum(dim=1, keepdim=True)

        logger.info(f"Load checkpoint {self.load_state_dict(state_dict, strict=False)}")


def _video_resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    conv_makers: Sequence[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
    layers: List[int],
    stem: Callable[..., nn.Module],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> VideoResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = VideoResNet(block=block, conv_makers=conv_makers, layers=layers, stem=stem, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


def r3d_18(*, weights: Optional[R3D_18_Weights] = None, progress: bool = True, **kwargs: Any) -> VideoResNet:
    """Construct 18 layer Resnet3D model.

    .. betastatus:: video module

    Reference: `A Closer Look at Spatiotemporal Convolutions for Action Recognition <https://arxiv.org/abs/1711.11248>`__.

    Args:
        weights (:class:`~torchvision.models.video.R3D_18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.R3D_18_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.resnet.VideoResNet`` base class.
            Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.R3D_18_Weights
        :members:
    """
    weights = R3D_18_Weights.verify(weights)

    return _video_resnet(
        BasicBlock,
        [Conv3DSimple] * 4,
        [2, 2, 2, 2],
        BasicStem,
        weights,
        progress,
        **kwargs,
    )


def mc3_18(*, weights: Optional[MC3_18_Weights] = None, progress: bool = True, **kwargs: Any) -> VideoResNet:
    """Construct 18 layer Mixed Convolution network as in

    .. betastatus:: video module

    Reference: `A Closer Look at Spatiotemporal Convolutions for Action Recognition <https://arxiv.org/abs/1711.11248>`__.

    Args:
        weights (:class:`~torchvision.models.video.MC3_18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.MC3_18_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.resnet.VideoResNet`` base class.
            Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.MC3_18_Weights
        :members:
    """
    weights = MC3_18_Weights.verify(weights)

    return _video_resnet(
        BasicBlock,
        [Conv3DSimple] + [Conv3DNoTemporal] * 3,  # type: ignore[list-item]
        [2, 2, 2, 2],
        BasicStem,
        weights,
        progress,
        **kwargs,
    )


def r2plus1d_18(*, weights: Optional[R2Plus1D_18_Weights] = None, progress: bool = True, **kwargs: Any) -> VideoResNet:
    """Construct 18 layer deep R(2+1)D network as in

    .. betastatus:: video module

    Reference: `A Closer Look at Spatiotemporal Convolutions for Action Recognition <https://arxiv.org/abs/1711.11248>`__.

    Args:
        weights (:class:`~torchvision.models.video.R2Plus1D_18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.R2Plus1D_18_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.resnet.VideoResNet`` base class.
            Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.R2Plus1D_18_Weights
        :members:
    """
    weights = R2Plus1D_18_Weights.verify(weights)

    return _video_resnet(
        BasicBlock,
        [Conv2Plus1D] * 4,
        [2, 2, 2, 2],
        R2Plus1dStem,
        weights,
        progress,
        **kwargs,
    )

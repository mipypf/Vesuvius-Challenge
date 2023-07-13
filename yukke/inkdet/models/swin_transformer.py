# modified from https://github.com/pytorch/vision/blob/v0.15.2/torchvision/models/video/swin_transformer.py
from functools import partial
from typing import Any, List, Optional

import torch
import torch.nn as nn
from loguru import logger
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.video.swin_transformer import PatchEmbed3d, Swin3D_S_Weights, Swin3D_T_Weights
from torchvision.models.video.swin_transformer import SwinTransformer3d as _SwinTransformer3d


class SwinTransformer3d(_SwinTransformer3d):
    def __init__(self, **kwargs):
        self.in_channels = kwargs.pop("in_channels", None)
        self.channel_first = kwargs.pop("channel_first", False)
        assert self.in_channels is None or self.in_channels == 1
        super(SwinTransformer3d, self).__init__(**kwargs)

        if self.in_channels is not None:
            norm_layer = kwargs.get("norm_layer")
            if norm_layer is None:
                norm_layer = partial(nn.LayerNorm, eps=1e-5)

            self.patch_embed = PatchEmbed3d(
                patch_size=kwargs["patch_size"],
                in_channels=self.in_channels,
                embed_dim=kwargs["embed_dim"],
                norm_layer=norm_layer,
            )

        # del self.norm
        del self.avgpool
        del self.head

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: B C T H W
        x = self.patch_embed(x)  # B _T _H _W C
        x = self.pos_drop(x)

        ret: List[torch.Tensor] = []
        for i in range(len(self.features)):
            x = self.features[i](x)  # B _T _H _W C
            if i % 2 == 0:
                ret.append(x)

        ret[-1] = self.norm(ret[-1])

        if self.channel_first:
            ret = [x.permute(0, 4, 1, 2, 3) for x in ret]  # B, C, _T, _H, _W

        return ret

    def load_checkpoint(self, model_name: str):
        if model_name == "swin3d_t":
            weights = Swin3D_T_Weights.verify("KINETICS400_V1")
        elif model_name == "swin3d_s":
            weights = Swin3D_S_Weights.verify("KINETICS400_V1")
        else:
            raise ValueError(f"Unexpected model name: {model_name}")

        state_dict = weights.get_state_dict(progress=True)

        if self.in_channels == 1:
            # Convert 3 channel weights to single channel
            # ref: https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1
            patch_embed_weight = state_dict["patch_embed.proj.weight"]
            state_dict["patch_embed.proj.weight"] = patch_embed_weight.sum(dim=1, keepdim=True)

        logger.info(f"Load checkpoint {self.load_state_dict(state_dict, strict=False)}")


def _swin_transformer3d(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> SwinTransformer3d:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = SwinTransformer3d(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


def swin3d_t(*, weights: Optional[Swin3D_T_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer3d:
    """
    Constructs a swin_tiny architecture from
    `Video Swin Transformer <https://arxiv.org/abs/2106.13230>`_.

    Args:
        weights (:class:`~torchvision.models.video.Swin3D_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.Swin3D_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.Swin3D_T_Weights
        :members:
    """
    weights = Swin3D_T_Weights.verify(weights)

    return _swin_transformer3d(
        patch_size=[2, 4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 7, 7],
        stochastic_depth_prob=0.1,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def swin3d_s(*, weights: Optional[Swin3D_S_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer3d:
    """
    Constructs a swin_small architecture from
    `Video Swin Transformer <https://arxiv.org/abs/2106.13230>`_.

    Args:
        weights (:class:`~torchvision.models.video.Swin3D_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.Swin3D_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.Swin3D_S_Weights
        :members:
    """
    weights = Swin3D_S_Weights.verify(weights)

    return _swin_transformer3d(
        patch_size=[2, 4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 7, 7],
        stochastic_depth_prob=0.1,
        weights=weights,
        progress=progress,
        **kwargs,
    )

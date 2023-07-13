# modified from https://github.com/pytorch/vision/blob/v0.15.2/torchvision/models/video/mvit.py
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.video.mvit import MSBlockConfig
from torchvision.models.video.mvit import MViT as _MViT
from torchvision.models.video.mvit import MViT_V1_B_Weights, MViT_V2_S_Weights, WeightsEnum


def _unsqueeze(x: torch.Tensor, target_dim: int, expand_dim: int) -> Tuple[torch.Tensor, int]:
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim


torch.fx.wrap("_unsqueeze")


class SpatialMaxPool3d(nn.MaxPool3d):
    """the deterministic algorithm replacement of nn.MaxPool3d

    I have checked that outputs are closed with nn.MaxPool3d with the command
    python -m inkdet.tools.check_max_pool3d
    """

    def __init__(self, **kwargs):
        super(SpatialMaxPool3d, self).__init__(**kwargs)

        def check_params(key: Union[int, Tuple[int, int, int]], value: int) -> bool:
            if isinstance(key, int):
                return key == value
            elif isinstance(key, Sequence):
                return key[0] == value
            else:
                return False

        assert check_params(self.kernel_size, 1)
        assert check_params(self.stride, 1)
        assert check_params(self.padding, 0)
        assert check_params(self.dilation, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 5

        def max_pool2d(_x: torch.Tensor) -> torch.Tensor:
            _x = F.max_pool2d(
                _x,
                self.kernel_size[1:] if isinstance(self.kernel_size, Sequence) else self.kernel_size,
                self.stride[1:] if isinstance(self.stride, Sequence) else self.stride,
                self.padding[1:] if isinstance(self.padding, Sequence) else self.padding,
                self.dilation[1:] if isinstance(self.dilation, Sequence) else self.dilation,
                ceil_mode=self.ceil_mode,
                return_indices=self.return_indices,
            )
            return _x

        x = torch.stack([max_pool2d(x[:, :, i]) for i in range(x.shape[2])], dim=2)

        return x


class MViT(_MViT):
    def __init__(self, **kwargs):
        self.in_channels = kwargs.pop("in_channels")
        self.channel_first = kwargs.pop("channel_first", False)
        self.num_classes = kwargs.get("num_classes")
        if "num_classes" in kwargs and kwargs["num_classes"] is None:
            kwargs.pop("num_classes")
        super(MViT, self).__init__(**kwargs)

        assert self.in_channels in [1, 3]

        self.with_classification = self.num_classes is not None

        # Patch Embedding module
        self.conv_proj = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=kwargs["block_setting"][0].input_channels,
            kernel_size=kwargs.get("patch_embed_kernel", (3, 7, 7)),
            stride=kwargs.get("patch_embed_stride", (2, 4, 4)),
            padding=kwargs.get("patch_embed_padding", (1, 3, 3)),
        )

        self.blocks[1].pool_skip.pool = SpatialMaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.blocks[3].pool_skip.pool = SpatialMaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.blocks[14].pool_skip.pool = SpatialMaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        if not self.with_classification:
            del self.head

    def forward(self, x: torch.Tensor) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
        # Convert if necessary (B, C, H, W) -> (B, C, 1, H, W)
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)

        ret: List[torch.Tensor] = []
        ret_ids: List[int] = [0, 2, 13, 15]
        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for i, block in enumerate(self.blocks):
            x, thw = block(x, thw)
            if i in ret_ids:
                # x.shape: (B, L, D)
                # thw: (t, h, w)
                # i=0 : x.shape = (8, 25089,  96), thw = (8, 56, 56)
                # i=2 : x.shape = (8,  6273, 192), thw = (8, 28, 28)
                # i=13: x.shape = (8,  1569, 384), thw = (8, 14, 14)
                # i=15: x.shape = (8,   392, 768), thw = (8,  7,  7)
                b, l, d = x.shape
                pos_emb = l - np.prod(thw)
                if pos_emb == 1:
                    # NOTE: remove a positional embedding
                    # https://github.com/pytorch/vision/blob/v0.15.2/torchvision/models/video/mvit.py#L411-L412
                    xx = x[:, 1:].contiguous()
                elif pos_emb > 1:
                    raise ValueError(f"Unexpected tensor shape: {x.shape}")
                ret.append(xx.view(b, *thw, d))

        ret[-1] = self.norm(ret[-1])

        if self.channel_first:
            ret = [_x.permute(0, 4, 1, 2, 3) for _x in ret]  # B, C, _T, _H, _W

        if self.with_classification:
            x = self.norm(x)
            # classifier "token" as used by standard language architectures
            x = x[:, 0]
            x = self.head(x)
            return ret, x
        else:
            return ret

    def load_checkpoint(self, model_name: str):
        if model_name == "mvit_v1_b":
            weights = MViT_V1_B_Weights.verify("KINETICS400_V1")
        elif model_name == "mvit_v2_s":
            weights = MViT_V2_S_Weights.verify("KINETICS400_V1")
        else:
            raise ValueError(f"Unexpected model name: {model_name}")

        state_dict = weights.get_state_dict(progress=True)

        if self.in_channels == 1:
            patch_embed_weight = state_dict["conv_proj.weight"]
            state_dict["conv_proj.weight"] = patch_embed_weight.sum(dim=1, keepdim=True)

        if self.with_classification:
            # NOTE: to match the shape, initialize head later
            state_dict["head.1.weight"] = state_dict["head.1.weight"][: self.num_classes]
            state_dict["head.1.bias"] = state_dict["head.1.bias"][: self.num_classes]

        logger.info(f"Load checkpoint {self.load_state_dict(state_dict, strict=False)}")


def _mvit(
    block_setting: List[MSBlockConfig],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MViT:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "spatial_size", weights.meta["min_size"])
        _ovewrite_named_param(kwargs, "temporal_size", weights.meta["min_temporal_size"])
    spatial_size = kwargs.pop("spatial_size", (224, 224))
    temporal_size = kwargs.pop("temporal_size", 16)

    model = MViT(
        spatial_size=spatial_size,
        temporal_size=temporal_size,
        block_setting=block_setting,
        residual_pool=kwargs.pop("residual_pool", False),
        residual_with_cls_embed=kwargs.pop("residual_with_cls_embed", True),
        rel_pos_embed=kwargs.pop("rel_pos_embed", False),
        proj_after_attn=kwargs.pop("proj_after_attn", False),
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def mvit_v1_b(*, weights: Optional[MViT_V1_B_Weights] = None, progress: bool = True, **kwargs: Any) -> MViT:
    """
    Constructs a base MViTV1 architecture from
    `Multiscale Vision Transformers <https://arxiv.org/abs/2104.11227>`__.

    .. betastatus:: video module

    Args:
        weights (:class:`~torchvision.models.video.MViT_V1_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.MViT_V1_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.MViT``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.MViT_V1_B_Weights
        :members:
    """
    weights = MViT_V1_B_Weights.verify(weights)

    config: Dict[str, List] = {
        "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
        "input_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
        "output_channels": [192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768, 768],
        "kernel_q": [[], [3, 3, 3], [], [3, 3, 3], [], [], [], [], [], [], [], [], [], [], [3, 3, 3], []],
        "kernel_kv": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "stride_q": [[], [1, 2, 2], [], [1, 2, 2], [], [], [], [], [], [], [], [], [], [], [1, 2, 2], []],
        "stride_kv": [
            [1, 8, 8],
            [1, 4, 4],
            [1, 4, 4],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
    }

    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )

    return _mvit(
        spatial_size=(224, 224),
        temporal_size=16,
        block_setting=block_setting,
        residual_pool=False,
        residual_with_cls_embed=False,
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.2),
        weights=weights,
        progress=progress,
        **kwargs,
    )


def mvit_v2_s(*, weights: Optional[MViT_V2_S_Weights] = None, progress: bool = True, **kwargs: Any) -> MViT:
    """
    Constructs a small MViTV2 architecture from
    `Multiscale Vision Transformers <https://arxiv.org/abs/2104.11227>`__.

    .. betastatus:: video module

    Args:
        weights (:class:`~torchvision.models.video.MViT_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.MViT_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.MViT``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.MViT_V2_S_Weights
        :members:
    """
    weights = MViT_V2_S_Weights.verify(weights)

    config: Dict[str, List] = {
        "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
        "input_channels": [96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768],
        "output_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
        "kernel_q": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "kernel_kv": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "stride_q": [
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
        ],
        "stride_kv": [
            [1, 8, 8],
            [1, 4, 4],
            [1, 4, 4],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
    }

    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )

    return _mvit(
        spatial_size=(224, 224),
        temporal_size=16,
        block_setting=block_setting,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.2),
        weights=weights,
        progress=progress,
        **kwargs,
    )

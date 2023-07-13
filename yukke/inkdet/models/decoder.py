import enum
from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torchvision.ops import MLP

from .swin_unet import PatchExpand4D, PatchExpand5D
from .utils import BatchNorm_Momentum, conv1x1, conv1x1x1, conv1x3x3, conv3x3, conv3x3x3, interpolate


class DecoderCNN2d(nn.Module):
    def __init__(
        self,
        classes: int,
        encoder_dims: List[int],
        upscale_mode: str,
        hypercolumn_indexes: List[int],
        downsample_factor: int,
        logit_upscale_factor: int,
        logit_after_upsample: bool = False,
    ):
        super().__init__()
        assert isinstance(classes, int) and classes > 0
        assert all([isinstance(d, int) and d > 0 for d in encoder_dims])
        assert upscale_mode in ["nearest", "bilinear"]
        assert isinstance(hypercolumn_indexes, list) and all([isinstance(i, int) for i in hypercolumn_indexes])
        assert isinstance(downsample_factor, int) and downsample_factor > 0
        assert isinstance(logit_upscale_factor, int) and logit_upscale_factor > 0
        self.upscale_mode = upscale_mode
        self.hypercolumn_indexes = hypercolumn_indexes
        self.downsample_factor = downsample_factor
        self.logit_upscale_factor = logit_upscale_factor
        self.logit_after_upsample = logit_after_upsample
        self.logit_upscale_params = dict(mode="bilinear", align_corners=True)

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    conv3x3(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1]),
                    nn.BatchNorm2d(encoder_dims[i - 1], momentum=BatchNorm_Momentum),
                    nn.ReLU(inplace=True),
                )
                for i in range(1, len(encoder_dims))
            ]
        )

        self.logit = conv1x1(sum([encoder_dims[i] for i in hypercolumn_indexes]), classes, bias=True)

    def forward(self, features: List[torch.Tensor], feature_only: bool):
        assert not feature_only, "NotImplemented"

        for i in range(len(features) - 1, 0, -1):
            feat_up = interpolate(features[i], scale_factor=2, mode=self.upscale_mode, align_corners=True)
            feat = torch.cat([features[i - 1], feat_up], dim=1)
            feat_down = self.convs[i - 1](feat)
            features[i - 1] = feat_down

        hypercolumns_features = [features[0]]
        hypercolumns_features += [
            interpolate(features[i], scale_factor=self.logit_upscale_factor * i, **self.logit_upscale_params)
            for i in self.hypercolumn_indexes[1:]
        ]
        hypercolumns_features = torch.cat(hypercolumns_features, dim=1)

        if self.logit_after_upsample:
            if self.logit_upscale_factor > 1:
                feature = interpolate(feature, scale_factor=self.logit_upscale_factor, **self.logit_upscale_params)
            logit = self.logit(hypercolumns_features)
        else:
            logit = self.logit(hypercolumns_features)
            if self.logit_upscale_factor > 1:
                logit = interpolate(logit, scale_factor=self.logit_upscale_factor, **self.logit_upscale_params)
        return logit


class UpSampleType(enum.Enum):
    DHW = "dhw"
    HW = "hw"
    D = "d"
    PIXEL_SHUFFLE = "pixel_shuffle"


class DecoderCNN3d(nn.Module):
    def __init__(
        self,
        classes: int,
        encoder_dims: List[int],
        upsample_mode: str,
        upsample_type: UpSampleType,
        skip_downscale_factor: Optional[int],
        skip_d_dim: Optional[int],
        logit_upscale_factor: int,
        logit_channel_scale_factor: int,
    ):
        super().__init__()
        assert isinstance(classes, int) and classes > 0
        assert all([isinstance(d, int) and d > 0 for d in encoder_dims])
        assert upsample_mode in ["nearest", "bilinear", "trilinear"]
        assert skip_downscale_factor is None or (isinstance(skip_downscale_factor, int) and skip_downscale_factor > 0)
        assert skip_d_dim is None or (isinstance(skip_d_dim, int) and skip_d_dim > 0)
        self.upsample_mode = upsample_mode
        self.upsample_type = upsample_type
        self.skip_downscale_factor = skip_downscale_factor
        self.skip_d_dim = skip_d_dim
        self.logit_upscale_factor = logit_upscale_factor
        self.logit_upscale_params = dict(mode="bilinear", align_corners=True)

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    # conv3x3x3(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1]),
                    conv1x3x3(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1]),
                    nn.BatchNorm3d(encoder_dims[i - 1], momentum=BatchNorm_Momentum),
                    nn.ReLU(inplace=True),
                )
                for i in range(1, len(encoder_dims))
            ]
        )

        self.logit = conv1x1(encoder_dims[0] * logit_channel_scale_factor, classes, bias=True)

    def upsample(
        self, x: torch.Tensor, scale_factor: int, upsample_type: Optional[UpSampleType] = None
    ) -> torch.Tensor:
        assert x.ndim == 5
        assert isinstance(scale_factor, int) and scale_factor > 1

        if upsample_type is None:
            upsample_type = self.upsample_type

        if upsample_type == UpSampleType.DHW:
            # encoder_type: resnet3d
            x = interpolate(
                x,
                scale_factor=scale_factor,
                mode=self.upsample_mode,
                align_corners=True,
            )
        elif upsample_type == UpSampleType.HW:
            # encoder_type: swin3d, resnet3d
            _, _, d, h, w = x.shape
            # x = rearrange(x, "b c d h w -> b (c d) h w")
            x = interpolate(
                x,
                size=(d, scale_factor * h, scale_factor * w),
                mode=self.upsample_mode,
                align_corners=True,
            )
            # x = rearrange(x, "b (c d) h w -> b c d h w", c=c, d=d)
        elif upsample_type == UpSampleType.D:
            _, _, d, h, w = x.shape
            x = interpolate(
                x,
                size=(scale_factor * d, h, w),
                mode=self.upsample_mode,
                align_corners=True,
            )
        elif upsample_type == UpSampleType.PIXEL_SHUFFLE:
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown upsample type: {self.upsample_type}")

        return x

    def reshape_skip_connect_feature(self, x: torch.Tensor, target_d_dim: int) -> torch.Tensor:
        assert x.ndim == 5
        assert target_d_dim > 0
        d = x.shape[2]
        if d == target_d_dim:
            return x
        elif d > target_d_dim:
            assert d % target_d_dim == 0
            # >>> a
            # array([0, 1, 2, 3, 4, 5])
            # >>> a.reshape(3, 2)
            # array([[0, 1],
            #       [2, 3],
            #       [4, 5]])
            # >>> a.reshape(3, 2).mean(1)
            # array([0.5, 2.5, 4.5])
            x = rearrange(
                x,
                "b c (d s) h w -> b c d s h w",
                d=target_d_dim,
                s=d // target_d_dim,
            )
            x = torch.mean(x, dim=3)
        elif d < target_d_dim:
            assert target_d_dim % d == 0
            scale_factor = target_d_dim // d
            x = self.upsample(x, scale_factor, UpSampleType.D)

        return x

    def forward(self, features: List[torch.Tensor]):
        # features: [encoder1, encoder2, encoder3, encoder4]
        # features[i]: B C D H W
        if self.skip_d_dim is not None:
            features[-1] = self.reshape_skip_connect_feature(features[-1], target_d_dim=self.skip_d_dim)

        for i in range(len(features) - 1, 0, -1):
            feat = features[i]
            # print(i, "feat", feat.shape)
            feat_up = self.upsample(features[i], scale_factor=2)
            # print(i, "feat_up", features[i - 1].shape, feat_up.shape)
            feat_skip = features[i - 1]
            if self.skip_d_dim is not None:
                feat_skip = self.reshape_skip_connect_feature(feat_skip, target_d_dim=self.skip_d_dim)
            # print(i, feat_skip.shape, feat_up.shape)
            feat = torch.cat([feat_skip, feat_up], dim=1)
            # print(i, "feat_cat", feat.shape)
            feat = self.convs[i - 1](feat)
            features[i - 1] = feat

        feature = features[0]
        # print("feature", feature.shape, self.logit)
        feature = rearrange(feature, "b c d h w -> b (c d) h w")
        logit = self.logit(feature)
        logit = interpolate(logit, scale_factor=self.logit_upscale_factor, **self.logit_upscale_params)

        return logit


class DecoderSimpleMLP(nn.Module):
    # ref: https://github.com/NVlabs/SegFormer/blob/1a8ad5123a4b251f16deb31bacbd1f6f5ce81347/mmseg/models/decode_heads/segformer_head.py
    def __init__(
        self,
        classes: int,
        encoder_dims: List[int],
        embedding_channels: int,
    ):
        super().__init__()

        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(encoder_dims[i], embedding_channels),
                    # nn.LayerNorm(embedding_channels),
                )
                for i in range(len(encoder_dims))
            ]
        )

        # self.norm = nn.LayerNorm(embedding_channels * 4, eps=1e-5)
        self.fuse = nn.Sequential(
            nn.LayerNorm(embedding_channels * 4, eps=1e-5),
            # conv1x1x1(embedding_channels * 4, embedding_channels),
            nn.Linear(embedding_channels * 4, embedding_channels),
            nn.GELU()
            # conv1x1(embedding_channels * 4, embedding_channels),
            # nn.BatchNorm3d(embedding_channels, momentum=BatchNorm_Momentum),
            # nn.ReLU(inplace=True),
        )

        self.logit = nn.Conv2d(embedding_channels * 8, classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # features[i]: B D H W C
        concat_feats: List[torch.Tensor] = []
        for i in reversed(range(len(features))):
            feat = features[i]
            feat = self.mlps[i](feat)
            feat = rearrange(feat, "b d h w c -> b c d h w")
            _, _, d, h, w = feat.shape
            s = 2**i
            feat = interpolate(
                feat,
                size=(d, s * h, s * w),
                mode="nearest",
            )
            concat_feats.append(feat)

        concat_feats = torch.cat(concat_feats, dim=1)
        concat_feats = rearrange(concat_feats, "b c d h w -> b d h w c")
        # concat_feats = self.norm(concat_feats)
        # concat_feats = rearrange(concat_feats, "b d h w c -> b c d h w")
        fused_feat = self.fuse(concat_feats)
        fused_feat = rearrange(fused_feat, "b d h w c -> b (c d) h w")

        logit = self.logit(fused_feat)
        logit = interpolate(logit, scale_factor=4, mode="bilinear", align_corners=True)

        return logit


class DecoderMLP(nn.Module):
    def __init__(
        self,
        classes: int,
        encoder_dims: List[int],
        patch_expand_channel_scale: int,
        logit_interpolate_upscale_factor: int,
        downsample_factor: int,
        num_layers: int,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        assert isinstance(classes, int) and classes > 0
        assert all([isinstance(d, int) and d > 0 for d in encoder_dims])
        assert isinstance(patch_expand_channel_scale, int) and patch_expand_channel_scale in [1, 2]
        assert isinstance(logit_interpolate_upscale_factor, int) and logit_interpolate_upscale_factor in [1, 2, 4]
        assert isinstance(downsample_factor, int) and downsample_factor in [1, 2, 4]
        assert isinstance(num_layers, int) and num_layers > 0
        assert 0 <= drop_rate <= 1
        # assert downsample_factor * logit_interpolate_upscale_factor == 4

        self.downsample_factor = downsample_factor

        if patch_expand_channel_scale == 1:
            self.patch_expands = nn.ModuleList(
                [
                    PatchExpand5D(
                        in_channels=encoder_dims[i],
                        channel_scale=patch_expand_channel_scale,
                        size_scale=2,
                    )
                    for i in range(1, len(encoder_dims))
                ]
            )
            if num_layers == 1:
                self.mlp = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(encoder_dims[i - 1] + encoder_dims[i] // 4),
                            nn.Linear(
                                encoder_dims[i - 1] + encoder_dims[i] // 4,
                                encoder_dims[i - 1],
                            ),
                            nn.GELU(),
                            nn.Dropout(drop_rate),
                        )
                        for i in range(1, len(encoder_dims))
                    ]
                )
            else:
                self.mlp = nn.ModuleList(
                    [
                        MLP(
                            in_channels=encoder_dims[i - 1] + encoder_dims[i] // 4,
                            hidden_channels=[encoder_dims[i - 1] for _ in range(num_layers)],
                            norm_layer=nn.LayerNorm,
                            activation_layer=nn.GELU,
                        )
                        for i in range(1, len(encoder_dims))
                    ]
                )
        elif patch_expand_channel_scale == 2:
            self.patch_expands = nn.ModuleList(
                [
                    PatchExpand5D(
                        in_channels=encoder_dims[i],
                        channel_scale=patch_expand_channel_scale,
                        size_scale=2,
                    )
                    for i in range(1, len(encoder_dims))
                ]
            )
            if num_layers == 1:
                self.mlp = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(encoder_dims[i]),
                            nn.Linear(
                                encoder_dims[i],
                                encoder_dims[i - 1],
                            ),
                            nn.GELU(),
                            nn.Dropout(drop_rate),
                        )
                        for i in range(1, len(encoder_dims))
                    ]
                )
            else:
                self.mlp = nn.ModuleList(
                    [
                        MLP(
                            in_channels=encoder_dims[i],
                            hidden_channels=[encoder_dims[i - 1] for _ in range(num_layers)],
                            norm_layer=nn.LayerNorm,
                            activation_layer=nn.GELU,
                        )
                        for i in range(1, len(encoder_dims))
                    ]
                )

        self.logit_interpolate_upscale_factor = logit_interpolate_upscale_factor
        self.logit_upscale_params = dict(mode="bilinear", align_corners=True)

        if logit_interpolate_upscale_factor == 2:
            # in_channels: 12
            # in_channels = 576
            # logit_in_channels = 576 // 4
            # in_channels: 16
            in_channels = encoder_dims[-1]
            logit_in_channels = encoder_dims[0] * 2

            self.patch_expand = PatchExpand4D(in_channels=in_channels, channel_scale=1, size_scale=2)
            self.logit = nn.Linear(logit_in_channels, classes)
        elif logit_interpolate_upscale_factor == 4:
            self.logit = nn.Linear(encoder_dims[0] * 8, classes)
        elif logit_interpolate_upscale_factor == 1:
            self.logit = nn.Linear(encoder_dims[0] * 8, classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, features: List[torch.Tensor], feature_only: bool):
        for i in range(len(features) - 1, 0, -1):
            feat_up = self.patch_expands[i - 1](features[i])
            feat = torch.cat([features[i - 1], feat_up], dim=-1)
            feat = self.mlp[i - 1](feat)
            features[i - 1] = feat

        if self.logit_interpolate_upscale_factor == 2:
            assert not feature_only, "NotImplemented"
            feature = rearrange(features[0], "b d h w c -> b h w (d c)")
            feature_up = self.patch_expand(feature)
            logit = self.logit(feature_up)
            logit = rearrange(logit, "b h w c -> b c h w")
            if self.downsample_factor == 1:
                logit = interpolate(
                    logit, scale_factor=self.logit_interpolate_upscale_factor, **self.logit_upscale_params
                )
        elif self.logit_interpolate_upscale_factor == 4:
            assert not feature_only, "NotImplemented"
            feature = rearrange(features[0], "b d h w c -> b h w (d c)")
            logit = self.logit(feature)
            logit = rearrange(logit, "b h w c -> b c h w")
            logit = interpolate(logit, scale_factor=self.logit_interpolate_upscale_factor, **self.logit_upscale_params)
        elif self.logit_interpolate_upscale_factor == 1:
            if feature_only:
                return features[0]

            logit = self.head(features[0])

        return logit

    def head(self, feature: torch.Tensor) -> torch.Tensor:
        assert feature.ndim == 5, f"Expected shape of (B, D, H, W, C): {feature.shape}"
        assert feature.shape[2] == feature.shape[3], f"Expected shape of (B, D, H, W, C): {feature.shape}"

        feature = rearrange(feature, "b d h w c -> b h w (d c)")
        logit = self.logit(feature)
        logit = rearrange(logit, "b h w c -> b c h w")
        return logit

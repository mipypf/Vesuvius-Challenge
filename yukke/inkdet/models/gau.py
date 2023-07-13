from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class GAUBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upscale_mode: str = "bilinear"):
        super(GAUBlock, self).__init__()
        self.upscale_mode = upscale_mode
        self.align_corners = True if upscale_mode == "bilinear" else None

        lateral_base_channels: int = 1024
        self.conv_low = nn.Sequential(
            conv3x3(lateral_base_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_high = nn.Sequential(
            # conv3x3(in_channels, out_channels), # v2
            conv1x1(in_channels, out_channels),  # v1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, feat_low, feat_high):
        """
        Args:
            feat_low: low level feature (B, C, D, H, W)
            feat_high: high level feature (B, C', H', W')
        """
        assert feat_low.ndim == 5
        assert feat_high.ndim == 4

        bs, c, d, h, w = feat_low.shape
        feat_low = self.conv_low(feat_low.view(bs, c * d, h, w))

        # v2
        feat_high_up = interpolate(
            self.conv_high(feat_high),
            scale_factor=2,
            mode=self.upscale_mode,
            align_corners=self.align_corners,
        )
        # v1
        # feat_high_up = interpolate(feat_high, scale_factor=2, mode=self.upscale_mode, align_corners=self.align_corners)
        # feat_high_up = self.conv_high(feat_high_up)
        attention = self.att(feat_high)

        return feat_high_up + feat_low * attention


class GAUDecoder2d(nn.Module):
    def __init__(self, classes: int, encoder_dims: List[int], upscale: int):
        super().__init__()
        assert isinstance(classes, int) and classes > 0
        assert all([isinstance(d, int) and d > 0 for d in encoder_dims])
        assert isinstance(upscale, int) and upscale > 0

        self.up = nn.ModuleList(
            [
                GAUBlock(
                    encoder_dims[i],
                    encoder_dims[i - 1],
                )
                for i in range(1, len(encoder_dims))
            ]
        )
        self.fpa = nn.Sequential(
            conv1x1(encoder_dims[-1] * 2, encoder_dims[-1]),
            nn.BatchNorm2d(encoder_dims[-1]),
            nn.ReLU(inplace=True),
        )

        self.logit = nn.Conv2d(encoder_dims[0], classes, 1, 1, 0)
        self.logit_upscale_params = dict(scale_factor=upscale, mode="bilinear", align_corners=True)

    def forward(self, features: List[torch.Tensor]):
        for i in range(len(features) - 1, 0, -1):
            if i == len(features) - 1:
                feature = features[i]
                b, c, d, h, w = feature.shape
                feature_3d = feature.view(b, c * d, h, w)
                feature = self.fpa(feature_3d)
                features[i] = feature

            features[i - 1] = self.up[i - 1](features[i - 1], features[i])

        logit = self.logit(features[0])
        logit = interpolate(logit, **self.logit_upscale_params)
        return logit

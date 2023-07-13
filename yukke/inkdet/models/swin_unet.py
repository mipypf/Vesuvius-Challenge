# modified from https://github.com/HuCaoFighting/Swin-Unet/blob/2aa09a30b9805c780cc482fa43ae19ae17d9c449/networks/swin_transformer_unet_skip_expand_decoder_sys.py
import torch
import torch.nn as nn
from einops import rearrange


class PatchExpand5D(nn.Module):
    def __init__(self, in_channels: int, channel_scale: int, size_scale: int, norm_layer=nn.LayerNorm):
        super().__init__()
        assert isinstance(in_channels, int) and in_channels > 0
        assert isinstance(channel_scale, int) and channel_scale > 0
        assert isinstance(size_scale, int) and size_scale in [1, 2]

        self.channel_scale = channel_scale
        self.size_scale = size_scale
        self.expand = (
            nn.Linear(
                in_channels,
                channel_scale * in_channels,
            )
            if channel_scale > 1
            else nn.Identity()
        )
        self.norm = norm_layer(channel_scale * in_channels // (size_scale**2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B D H W C
        assert x.ndim == 5
        C = x.shape[-1]
        x = self.expand(x)

        C_s = C * self.channel_scale
        x = rearrange(
            x,
            "b d h w (p1 p2 c) -> b d (h p1) (w p2) c",
            p1=self.size_scale,
            p2=self.size_scale,
            c=C_s // (self.size_scale**2),
        )
        x = self.norm(x)

        return x


class PatchExpand4D(nn.Module):
    def __init__(self, in_channels: int, channel_scale: int, size_scale: int, norm_layer=nn.LayerNorm):
        super().__init__()
        assert isinstance(in_channels, int) and in_channels > 0
        assert isinstance(channel_scale, int) and channel_scale > 0
        assert isinstance(size_scale, int) and size_scale in [1, 2]

        self.channel_scale = channel_scale
        self.size_scale = size_scale
        self.expand = (
            nn.Linear(
                in_channels,
                channel_scale * in_channels,
            )
            # if channel_scale > 1
            # else nn.Identity()
        )
        self.norm = norm_layer(channel_scale * in_channels // (size_scale**2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B H W C
        assert x.ndim == 4
        C = x.shape[-1]
        x = self.expand(x)

        C_s = C * self.channel_scale
        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.size_scale,
            p2=self.size_scale,
            c=C_s // (self.size_scale**2),
        )
        x = self.norm(x)

        return x

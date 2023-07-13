import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm_Momentum = 0.1


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> torch.Tensor:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> torch.Tensor:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> torch.Tensor:
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def conv1x3x3(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> torch.Tensor:
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=bias)


def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> torch.Tensor:
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def interpolate(
    x: torch.Tensor, mode: str, size=None, scale_factor: int = None, align_corners: bool = False
) -> torch.Tensor:
    assert scale_factor is None or size is None

    if mode == "nearest":
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode="nearest")
    else:
        # There are an error `upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation`
        # ref: https://discuss.pytorch.org/t/deterministic-behavior-using-bilinear2d/131355/3
        device, dtype = x.device, x.dtype
        x_cpu = x.cpu().float()
        x_cpu = F.interpolate(x_cpu, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        return x_cpu.to(dtype).to(device)

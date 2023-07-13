import torch
import torch.nn as nn

from .resnet import BasicBlock, ResNet, conv1x1x1, get_inplanes


class SEBasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__(in_planes, planes, stride, downsample)
        squeeze_factor: int = 16

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = conv1x1x1(planes, int(planes / squeeze_factor))
        self.fc2 = conv1x1x1(int(planes / squeeze_factor), planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        se_out = self.global_pool(out)
        se_out = self.fc1(se_out)
        se_out = self.relu(se_out)
        se_out = self.fc2(se_out)
        se_out = self.sigmoid(se_out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out * se_out + residual
        out = self.relu(out)

        return out


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18]

    if model_depth == 10:
        model = ResNet(SEBasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(SEBasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)

    return model

import torch
import torch.nn as nn

from inkdet.models.mvit import SpatialMaxPool3d
from inkdet.utils import set_seed


def main():
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_pool3d_1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)).to(device)
    max_pool3d_2 = SpatialMaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)).to(device)
    print(f"nn.MaxPool3d: {max_pool3d_1}")
    print(f"SpatialMaxPool3d: {max_pool3d_2}")

    num_tries: int = 100
    for _ in range(num_tries):
        x = torch.randn((1, 2, 8, 14, 14)).to(device)

        y1 = max_pool3d_1(x)
        y2 = max_pool3d_2(x)

        assert y1.shape == y2.shape
        assert torch.allclose(y1, y2)

    print(f"All outputs are close between nn.MaxPool3d and SpatialMaxPool3d ({num_tries} tries on {device})")
    # passed when device is cpu.


if __name__ == "__main__":
    main()

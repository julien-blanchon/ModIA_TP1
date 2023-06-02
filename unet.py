"""Unet implementation in PyTorch."""
import torch
import torch.nn as nn


def double_conv(in_channels: int, out_channels: int) -> nn.Module:
    """Generate a double convolution block.

    Kernel size is fixed to 3 and padding is fixed to 1.

    Parameters
    ----------
    in_channels : int
        The number of input channel.
    out_channels : int
        The number of output channel.

    Returns
    -------
    nn.Module
        The Double Conv.
    """
    # returns a block compsed of two Convolution layers with ReLU activation function
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

class DownSampleBlock(nn.Module):
    """DownSampleBlock class."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = double_conv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d((2,2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward downsample."""
        x_skip = self.conv_block.forward(x)
        out = self.maxpool.forward(x_skip)

        return out , x_skip

class UpSampleBlock(nn.Module):
    """UpSampleBlock class."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = double_conv(in_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=(2,2), mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1) # concatenates x and x_skip
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """UNet class."""

    def __init__(self):
        super().__init__()

        self.downsample_block_1 = DownSampleBlock(1, 32)
        self.downsample_block_2 = DownSampleBlock(32, 64)
        self.downsample_block_3 = DownSampleBlock(64, 128)
        self.middle_conv_block = double_conv(128, 256)


        self.upsample_block_3 = UpSampleBlock(256+128, 128)
        self.upsample_block_2 = UpSampleBlock(128+64, 64)
        self.upsample_block_1 = UpSampleBlock(64+32, 32)
        self.last_conv = nn.Conv2d(32, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, x_skip1 = self.downsample_block_1.forward(x)
        x, x_skip2 = self.downsample_block_2.forward(x)
        x, x_skip3 = self.downsample_block_3.forward(x)

        x = self.middle_conv_block.forward(x)

        x = self.upsample_block_3.forward(x, x_skip3)
        x = self.upsample_block_2.forward(x, x_skip2)
        x = self.upsample_block_1.forward(x, x_skip1)

        out = self.last_conv(x)

        return out

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.downsample_block_1(x)
        x, _ = self.downsample_block_2(x)
        x, _ = self.downsample_block_3(x)
        return x


if __name__=="__main__":
    x = torch.rand(16,1,224,224)
    net = UNet()
    y = net(x)
    assert y.shape == (16,3,224,224)
    print("Shapes OK")

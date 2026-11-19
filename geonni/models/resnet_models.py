from torch import Tensor, flatten, nn
from torchvision.ops import Conv2dNormActivation


class ResidualBlock(nn.Module):
    """ResNet-style CNN block with residual connection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # the main processing of the input happens here
        self.block_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
        )
        # in case of non-1 stride or a mismatch of input and output channels,
        #  the residual connection requires a resampling layer to match the changed tensor shape
        self.resampling_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
        ) if (stride != 1 or in_channels != out_channels) else None
        # final activation layer after residual addition
        self.final_activation = nn.ReLU(inplace=True)


    def forward(self, input: Tensor) -> Tensor:
        output: Tensor = self.block_layers(input)

        if self.resampling_layer is not None:
            input = self.resampling_layer(input)

        output += input # residual addition
        output = self.final_activation(output)

        return output


class ResidualBottleneck(ResidualBlock):
    """Bottleneck variant of the residual block."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 4):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride)
        reduced_channels = out_channels // reduction
        self.block_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=reduced_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=reduced_channels, out_channels=reduced_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(num_features=reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=reduced_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels),
        )


class ClassificationResNet(nn.Module):
    """
    Image classifying convolutional neural network based on ResNet.
    """
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()

        self.encoder = nn.Sequential( # input img shape: 90x90x3
            Conv2dNormActivation(in_channels=3, out_channels=64, kernel_size=11), # -> 80x80x64
            nn.AdaptiveMaxPool2d(output_size=40), # -> 40x40x64
            ResidualBottleneck(in_channels=64, out_channels=64), # -> 40x40x64
            ResidualBottleneck(in_channels=64, out_channels=64), # -> 40x40x64
            ResidualBlock(in_channels=64, out_channels=128, stride=2), # -> 20x20x128
            ResidualBottleneck(in_channels=128, out_channels=128), # -> 20x20x128
            ResidualBottleneck(in_channels=128, out_channels=128), # -> 20x20x128
            ResidualBottleneck(in_channels=128, out_channels=128), # -> 20x20x128
            ResidualBottleneck(in_channels=128, out_channels=128), # -> 20x20x128
            ResidualBottleneck(in_channels=128, out_channels=128), # -> 20x20x128
            ResidualBottleneck(in_channels=128, out_channels=128), # -> 20x20x128
            ResidualBlock(in_channels=128, out_channels=256, stride=2), # -> 10x10x256
            ResidualBottleneck(in_channels=256, out_channels=256), # -> 10x10x256
            ResidualBottleneck(in_channels=256, out_channels=256), # -> 10x10x256
            ResidualBottleneck(in_channels=256, out_channels=256), # -> 10x10x256
            ResidualBottleneck(in_channels=256, out_channels=256), # -> 10x10x256
            ResidualBottleneck(in_channels=256, out_channels=256), # -> 10x10x256
            ResidualBottleneck(in_channels=256, out_channels=256), # -> 10x10x256
            ResidualBottleneck(in_channels=256, out_channels=256), # -> 10x10x256
            ResidualBottleneck(in_channels=256, out_channels=256), # -> 10x10x256
            ResidualBottleneck(in_channels=256, out_channels=256), # -> 10x10x256
            ResidualBottleneck(in_channels=256, out_channels=256), # -> 10x10x256
            ResidualBlock(in_channels=256, out_channels=512, stride=2), # -> 5x5x512
            ResidualBottleneck(in_channels=512, out_channels=512), # -> 10x10x256
            ResidualBottleneck(in_channels=512, out_channels=512), # -> 10x10x256
        )
        self.pool_layer = nn.AdaptiveAvgPool2d(output_size=1) # -> 1x1x512
        self.classifier = nn.Linear(in_features=512, out_features=num_classes)


    def forward(self, input: Tensor) -> Tensor:
        emb: Tensor = self.encoder(input)
        emb = self.pool_layer(emb)
        emb = flatten(emb, start_dim=1, end_dim=-1)
        emb = self.classifier(emb)
        return emb

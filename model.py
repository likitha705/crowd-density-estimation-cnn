"""CSRNet: VGG-16 front-end + dilated-convolution back-end for density-map regression."""
import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, pretrained_frontend: bool = True):
        super().__init__()
        # Front-end: first 10 conv layers of VGG-16 (output is 1/8 of input size)
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained_frontend else None
        vgg = models.vgg16(weights=weights)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        # Back-end: dilated convolutions keep resolution while enlarging receptive field
        cfg = [512, 512, 512, 256, 128, 64]
        layers, in_ch = [], 512
        for out_ch in cfg:
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2), nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.backend = nn.Sequential(*layers)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        for m in list(self.backend.modules()) + [self.output_layer]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)  # density map; crowd count = density.sum()


if __name__ == "__main__":
    net = CSRNet(pretrained_frontend=False)
    out = net(torch.randn(1, 3, 256, 256))
    print("Output shape:", out.shape)

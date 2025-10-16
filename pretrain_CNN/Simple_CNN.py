import torch
import torch.nn as nn
import torch.nn.functional as F
from TV_activation import TVLeakyReLU

class SimpleCNN(nn.Module):
    def __init__(self, scale_factor=4, use_TVrelu=False):
        super(SimpleCNN, self).__init__()
        self.scale_factor = scale_factor
        self.use_TVrelu = use_TVrelu

        # convolution layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 3 * scale_factor ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        self.relu = nn.ReLU(inplace=True)
        self.tvleaky = TVLeakyReLU(n_channel=3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.conv4.weight, mean=0.0, std=0.001)
        if self.conv4.bias is not None:
            nn.init.constant_(self.conv4.bias, 0)

    def forward(self, x):
        x_skip = x
        x_up = F.interpolate(x_skip, scale_factor=self.scale_factor,
                             mode='bicubic', align_corners=False)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.pixel_shuffle(x)

        # Apply TVLeakyReLU only if enabled in config
        if self.use_TVrelu:
            x = self.tvleaky(x)

        return x + x_up


if __name__ == '__main__':
    x = torch.randn(7, 3, 32, 32)
    net = SimpleCNN(use_TVrelu=True)
    y = net(x)
    print(y.shape)

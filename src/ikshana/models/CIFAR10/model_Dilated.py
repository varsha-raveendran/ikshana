import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            # 32x32x3 -> 32x32x32
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # 32x32x32 -> 32x32x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # 32x32x32 -> 16x16x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )


        self.layer2 =  nn.Sequential(
            # 16x16x32 -> 16x16x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # 16x16x32 -> 16x16x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # 16x16x32 -> 8x8x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )

        self.layer3 =  nn.Sequential(
            # 8x8x32 -> 8x8x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # 8x8x32 -> 8x8x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # 8x8x32-> 4x4x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )

        self.layer4 =  nn.Sequential(
            # 4x4x32 -> 4x4x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # 4x4x32 -> 4x4x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # # 4x4x32 -> 1x1x32
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        x = x.view(-1,10)

        return F.log_softmax(x, dim=1)
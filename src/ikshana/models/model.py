import torch.nn as nn
import torch.nn.functional as F

def norm_func(normalization_type, conv_layer, **kwargs):
    if normalization_type == "BN":
        return nn.BatchNorm2d(conv_layer.out_channels)
    elif normalization_type == "LN":
        return nn.GroupNorm(num_groups=1, num_channels=conv_layer.out_channels)
    elif normalization_type ==  "GN":
        return nn.GroupNorm(num_groups=int((conv_layer.out_channels)/2), num_channels=conv_layer.out_channels)

class Net(nn.Module):
    def __init__(self, normalization_type: str ="BN", **kwargs):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm_func(normalization_type, self.conv1, **kwargs)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1,bias=False)
        self.norm2 = norm_func(normalization_type, self.conv2, **kwargs)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, bias=False)
        self.norm3 = norm_func(normalization_type, self.conv3, **kwargs)

        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, bias=False)
        self.norm4 = norm_func(normalization_type, self.conv4, **kwargs)

        self.conv5 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, bias=False)
        self.norm5 = norm_func(normalization_type, self.conv5, **kwargs)

        self.conv6 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, bias=False)
        self.norm6 = norm_func(normalization_type, self.conv6, **kwargs)

        self.conv7 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, bias=False)
        self.norm7 = norm_func(normalization_type, self.conv7, **kwargs)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=1),
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=1)
        )

    def forward(self, net):
        net = F.relu(self.norm1(self.conv1(net)))
        net = self.maxpool1(F.relu(self.norm2(self.conv2(net))))
        net = F.relu(self.norm3(self.conv3(net)))
        net = F.relu(self.norm4(self.conv4(net)))
        net = F.relu(self.norm5(self.conv5(net)))
        net = F.relu(self.norm6(self.conv6(net)))
        net = F.relu(self.norm7(self.conv7(net)))
        net = self.gap(net)
        net = self.classifier(net)
        net = net.view(-1,10)

        return F.log_softmax(net, dim=1)
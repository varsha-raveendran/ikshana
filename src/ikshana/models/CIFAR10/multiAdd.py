import torch
import torch.nn as nn
import torch.nn.functional as F


class WeirdBlock(nn.Module):

    def __init__(self, in_planes, out_planes, norm=nn.BatchNorm2d, act=nn.ReLU):
        super(DoubleConvBlock, self).__init__()

        
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size= 3, stride= 1, padding= 1, bias= False)
        self.pool = nn.MaxPool2d(2,2)
        self.bn1 = norm(out_planes)
        self.act = act()

        self.DoubleConv = nn.Sequential(
                                nn.Conv2d(out_planes, out_planes, kernel_size= 3, stride= 1, padding= 1, bias= False),
                                norm(out_planes),
                                act(),
                                nn.Conv2d(out_planes, out_planes, kernel_size= 3, stride= 1, padding= 1, bias= False),
                                norm(out_planes),
                                act()
                            )
        
        

    def forward(self, x):

        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU
        out = self.act(self.bn1(self.pool(self.conv(x))))
        identity = out

        # R1 = ResBlock((Conv-BN-ReLU-Conv-BN-ReLU))(X)
        out = self.DoubleConv(out)
        out += identity

        # Add(X, R1)
        out += identity

        return out


class SomeNet(nn.Module):

    def __init__(self, block):
        super(SomeNet, self).__init__()

        self.prep = nn.Sequential(
                            nn.Conv2d(3, 64, kernel_size= 3, stride= 1, padding=1, bias= False),
                            nn.BatchNorm2d(64),
                            nn.ReLU()
                            )

        self.layer1 = block(64,128)
        
        self.layer2 = nn.Sequential(
                            nn.Conv2d(128, 256, kernel_size= 3, stride= 1, padding=1, bias= False),
                            nn.MaxPool2d(2,2),
                            nn.BatchNorm2d(256),
                            nn.ReLU()
                            )

        self.layer3 = block(256,512)

        self.global_pool = nn.MaxPool2d(4)
        self.linear = nn.Linear(512,256)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)
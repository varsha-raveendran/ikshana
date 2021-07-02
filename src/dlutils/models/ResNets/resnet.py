'''
Pre-activation ResNet - V2
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
    (Pre Activated Resnet(V2))

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
    (ResNet (V1))
'''

__all__ = ['resnet18']

import torch
import torch.nn as nn
import torch.nn.functional as F
import typing


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, norm=nn.BatchNorm2d, act=nn.ReLU):
        super(BasicBlock, self).__init__()

        self.norm1 = norm(in_planes)
        self.act1 = act()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size= 3, stride= stride, padding= 1, bias= False)
        
        self.norm2 = norm(planes)
        self.act2 = act()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size= 3, stride= 1, padding=1, bias= False)        

        if stride != 1 and in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size= 1, stride= stride, bias= False)
            )
        
    
    def forward(self, x):

        identity = self.downsample(x) if hasattr(self, 'downsample') else x
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))

        out += identity

        return out



class ResNet(nn.Module):
    def __init__(self, block:nn.Module, num_block:typing.List[int], num_classes:int= 10, in_channels:int= 3, 
                    layer0= None, norm= nn.BatchNorm2d, act= nn.ReLU, **kwargs):
        super(ResNet, self).__init__()

        self.norm = norm
        self.act = act

        self.stride = kwargs.get('stride', [1,2,2,2])

        self.in_planes = 64
        if layer0 is None:
            self.layer0 = nn.Conv2d(in_channels, self.in_planes, kernel_size= 3, padding= 1, bias= False)

        self.layer1 = self._make_layer(block, self.in_planes, 64, num_block[0], self.stride[0])
        self.layer2 = self._make_layer(block, self.in_planes, 128, num_block[1], self.stride[1])
        self.layer3 = self._make_layer(block, self.in_planes, 256, num_block[2], self.stride[2])
        self.layer4 = self._make_layer(block, self.in_planes, 512, num_block[3], self.stride[3])
        self.linear = nn.Linear(512, num_classes)

        
    
    def _make_layer(self, block, in_planes, planes, num_block, stride):
        
        strides = [stride] + [1]*(num_block-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm, self.act))
            self.in_planes = planes
        
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)



def resnet18(**kwargs): return ResNet(BasicBlock, num_block= [2,2,2,2], num_classes= 10, **kwargs)
def resnet34(**kwargs): return ResNet(BasicBlock, num_block= [3,4,6,3], num_classes= 10, **kwargs)

if __name__ == '__main__':
    a = torch.rand(2,3,32,32)
    m = resnet18()
    output = m(a)
    print(output.shape)
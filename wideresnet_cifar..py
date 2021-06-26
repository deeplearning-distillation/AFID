from __future__ import absolute_import
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1):
    """
    one 3x3 convolution with padding\n
    in_planes:  输入通道数\n
    out_planes: 输出通道数
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # print("--------------ca------------"+ str(in_planes))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        scale = self.sigmoid(x)
        return scale


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        print("--------------------"+ str(in_planes) +"," + str(planes))        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class AFID_Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(FFL_Wide_ResNet, self).__init__()
        self.inplanes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        # am add position
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()

        fix_inplanes=self.inplanes        
        self.layer1_1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.inplanes = fix_inplanes  ##reuse self.inplanes        
        self.layer1_2 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        
        fix_inplanes=self.inplanes   
        self.layer2_1 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.inplanes = fix_inplanes  ##reuse self.inplanes            
        self.layer2_2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        
        fix_inplanes=self.inplanes   
        self.layer3_1 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.inplanes = fix_inplanes  ##reuse self.inplanes            
        self.layer3_2 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        
        # am add position
        self.ca1 = ChannelAttention(self.inplanes)
        self.sa1 = SpatialAttention()

        self.bn1 = nn.BatchNorm2d(nStages[3])
        
        self.classfier3_1 = nn.Linear(nStages[3], num_classes)
        self.classfier3_2 = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.inplanes, planes, dropout_rate, stride))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        fmap = []

        # lower layers
        x_1 = self.conv1(x)
        x_2 = self.conv1(x)

        # AM
        am = self.ca(x_2) * x_2
        am = self.sa(am)     
        
        cm = self.ca(x_1) * x_1
        cm = self.sa(cm) 

        x_1 = x_1 * am
        x_2 = x_2 * cm
        x_1_1 = self.layer1_1(x_1)  
        x_1_2 = self.layer1_2(x_2)  

        x_2_1 = self.layer2_1(x_1_1)
        x_2_2 = self.layer2_2(x_1_2)

        x_3_1 = self.layer3_1(x_2_1)  # 8x8 
        x_3_2 = self.layer3_2(x_2_2)        
        fmap.append(x_3_1)
        fmap.append(x_3_2)

        am1 = self.ca1(x_3_2) * x_3_2
        am2 = self.sa1(am1)     
        
        cm1 = self.ca1(x_3_1) * x_3_1
        cm2 = self.sa1(cm1) 
       
        x_3_1 = x_3_1 * am2
        x_3_2 = x_3_2 * cm2

        x_3_1 = F.relu(self.bn1(x_3_1))
        x_3_2 = F.relu(self.bn1(x_3_2))

        # output layer
        x_3_1 = F.avg_pool2d(x_3_1, 8)
        x_3_1 = x_3_1.view(x_3_1.size(0), -1)
        x_3_2 = F.avg_pool2d(x_3_2, 8)
        x_3_2 = x_3_2.view(x_3_2.size(0), -1)

        x_3_1 = self.classfier3_1(x_3_1)
        x_3_2 = self.classfier3_2(x_3_2)
        return x_3_1, x_3_2, fmap

class Fusion_module(nn.Module):
    def __init__(self,channel,numclass,sptial):
        super(Fusion_module, self).__init__()
        self.fc2   = nn.Linear(channel, numclass)
        self.conv1 =  nn.Conv2d(channel*2, channel*2, kernel_size=3, stride=1, padding=1, groups=channel*2, bias=False)
        self.bn1 = nn.BatchNorm2d(channel * 2)
        self.conv1_1 = nn.Conv2d(channel*2, channel, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channel)

        self.sptial = sptial

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #self.avg = channel
    def forward(self, x,y):
        bias = False
        atmap = []
        input = torch.cat((x,y),1)

        # print("------------------------------"+ str(x.size())+"," + str(y.size())+ ","+ str(input.size()))


        x = F.relu(self.bn1((self.conv1(input))))
        # print("----------------1--------------"+ str(x.size()))
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        # print("----------------2--------------"+ str(x.size()))

        atmap.append(x)
        # print("----------------3--------------"+ str(x.size()))           
        x = F.avg_pool2d(x, self.sptial)
        # print("----------------4--------------"+ str(x.size()))        
        x = x.view(x.size(0), -1)

        out = self.fc2(x)
        atmap.append(out)

        return out


def afid_wide_resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return AFID_Wide_ResNet(**kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Unet(nn.Module):

    def __init__(self, kernel, num_classes=None, input_channels=1):
        super(Unet, self).__init__()
        self.kernel = kernel
        self.maxpool = nn.MaxPool2d(2)

        self.downblock1 = convolution_block(input_channels, 64, self.kernel)
        self.downblock2 = nn.Sequential(*[self.maxpool, convolution_block(64, 128, self.kernel)])
        self.downblock3 = nn.Sequential(*[self.maxpool, convolution_block(128, 256, self.kernel)])
        self.downblock4 = nn.Sequential(*[self.maxpool, convolution_block(256, 512, self.kernel)])

        self.bridge     = nn.Sequential(*[self.maxpool, convolution_block(512, 1024, self.kernel), Transposed_convolution(1024, 512, 2, 2)])
        
        self.upblock4   = nn.Sequential(*[convolution_block(1024, 512, self.kernel), Transposed_convolution(512, 256, 2, 2)])
        self.upblock3   = nn.Sequential(*[convolution_block(512, 256, self.kernel), Transposed_convolution(256, 128, 2, 2)])
        self.upblock2   = nn.Sequential(*[convolution_block(256, 128, self.kernel), Transposed_convolution(128, 64, 2, 2)])
        self.upblock1   = convolution_block(128, 64, self.kernel)
        
        self.output     = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        x1 = self.downblock1(x)
        x2 = self.downblock2(x1)
        x3 = self.downblock3(x2)
        x4 = self.downblock4(x3)

        x5 = self.bridge(x4)

        x5 = self.concat(x5,x4)
        x6 = self.upblock4(x5)
        
        x6 = self.concat(x6,x3)
        x7 = self.upblock3(x6)
      
        x7 = self.concat(x7,x2)
        x8 = self.upblock2(x7)

        x8 = self.concat(x8,x1)
        x9 = self.upblock1(x8)
        
        out = self.output(x9)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def concat(self, small, big):
        small_h, small_w = small.size()[2:]
        big_h,   big_w   = big.size()[2:]
        half_h = (big_h - small_h) // 2
        half_w = (big_w - small_w) // 2
        crop = big[:,:,half_h:half_h+small_h,half_w:half_w+small_w]

        return torch.cat([small,crop],dim=1)


class convolution_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(convolution_block, self).__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel         = kernel
        self.block1         = self.make_block(self.in_channels, self.out_channels)
        self.block2         = self.make_block(self.out_channels, self.out_channels)

    def make_block(self, in_channels, out_channels):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = self.kernel, stride=1, padding=0, bias=True)
        return nn.Sequential(self.conv, nn.ReLU())

    def forward(self, x):
        y = self.block1(x)
        out = self.block2(y)
        return out


class Transposed_convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Transposed_convolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride     = stride
        self.upsampling = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, 
                                             stride=self.stride)
    
    def forward(self, x):
        x = self.upsampling(x)
        return x


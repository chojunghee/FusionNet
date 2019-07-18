import torch.nn as nn
import torch.nn.functional as F
import math

# modified for image segmentation other than EM dataset
class FusionNet(nn.Module):

    def __init__(self, kernel, num_classes=None, input_channels=1):
        super(FusionNet, self).__init__()
        self.kernel = kernel
        self.maxpool = nn.MaxPool2d(2)
        self.downblock1 = residual_block(input_channels, 64, self.kernel)
        self.downblock2 = nn.Sequential(*[self.maxpool, residual_block(64, 128, self.kernel)])
        self.downblock3 = nn.Sequential(*[self.maxpool, residual_block(128, 256, self.kernel)])
        self.downblock4 = nn.Sequential(*[self.maxpool, residual_block(256, 512, self.kernel)])
        self.bridge     = nn.Sequential(*[self.maxpool, residual_block(512, 1024, self.kernel), Transposed_convolution(1024, 512, 2, 2)])
        self.upblock4   = nn.Sequential(*[residual_block(512, 512, self.kernel), Transposed_convolution(512, 256, 2, 2)])
        self.upblock3   = nn.Sequential(*[residual_block(256, 256, self.kernel), Transposed_convolution(256, 128, 2, 2)])
        self.upblock2   = nn.Sequential(*[residual_block(128, 128, self.kernel), Transposed_convolution(128, 64, 2, 2)])
        self.upblock1   = residual_block(64, 64, self.kernel)
        self.output     = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=self.kernel, stride=1, padding=1)

        """ self.output     = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=self.kernel, stride=1, padding=1)
        self.sigmoid    = nn.Sigmoid() """
        self._initialize_weights()

    def forward(self, x):
        x1 = self.downblock1(x)
        x2 = self.downblock2(x1)
        x3 = self.downblock3(x2)
        x4 = self.downblock4(x3)

        x5 = self.bridge(x4)
        x4, x5 = self.size_matching(x4,x5)
        x5 = x5 + x4

        x6 = self.upblock4(x5)
        x3, x6 = self.size_matching(x3,x6)
        x6 = x6 + x3

        x7 = self.upblock3(x6)
        x2, x7 = self.size_matching(x2,x7)
        x7 = x7 + x2

        x8 = self.upblock2(x7)
        x1, x8 = self.size_matching(x1,x8)
        x8 = x8 + x1

        x9 = self.upblock1(x8)
        out = self.output(x9)
        # out = self.sigmoid(out)

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

    def size_matching(self, im1, im2):
        if im1.size(2) < im2.size(2):
            pad = nn.ReflectionPad2d((0, 0, im2.size(2)-im1.size(2), 0))
            im1 = pad(im1)
        else:
            pad = nn.ReflectionPad2d((0, 0, im1.size(2)-im2.size(2), 0))
            im2 = pad(im2)

        if im1.size(3) < im2.size(3):
            pad = nn.ReflectionPad2d((im2.size(3)-im1.size(3), 0, 0, 0))
            im1 = pad(im1)
        else:
            pad = nn.ReflectionPad2d((im1.size(3)-im2.size(3), 0, 0, 0))
            im2 = pad(im2)
        return im1, im2


class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(residual_block, self).__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel         = kernel
        self.block1         = self.make_block(self.in_channels, self.out_channels)
        self.block2         = self.make_block(self.out_channels, self.out_channels)
        self.mid_block      = nn.Sequential(*[self.make_block(self.out_channels, self.out_channels) for _ in range(3)])

    def make_block(self, in_channels, out_channels):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = self.kernel, stride=1, padding=int((self.kernel-1)/2), bias=True)
        self.bn   = nn.BatchNorm2d(out_channels)
        return nn.Sequential(self.conv, nn.ReLU(), self.bn)

    def forward(self, x):
        residual = self.block1(x)
        y        = self.mid_block(residual)
        y        = y + residual
        out      = self.block2(y)
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


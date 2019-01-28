import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from base import BaseModel

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class Unet3DModel(BaseModel):
    def __init__(self, input_nc, output_nc, acti_type, norm_type, num_groups=0):
        super(Unet3DModel, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.get_norm_layer(norm_type, num_groups)
        self.get_acti_func(acti_type)
        use_bias = True
        
        block1 = [nn.Conv3d(self.input_nc, 32, kernel_size=3, padding=1, bias=use_bias),
                  self.norm_layer(32),
                  self.acti_func(),
                  nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=use_bias),
                  self.norm_layer(64),
                  self.acti_func()]
        block2 = [nn.MaxPool3d(2),
                  nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
                  self.norm_layer(64),
                  self.acti_func(),
                  nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
                  self.norm_layer(128),
                  self.acti_func()]
        block3 = [nn.MaxPool3d(2),
                  nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
                  self.norm_layer(128),
                  self.acti_func(),
                  nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
                  self.norm_layer(256),
                  self.acti_func()]
        block4 = [nn.MaxPool3d(2),
                  nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
                  self.norm_layer(256),
                  self.acti_func(),
                  nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
                  self.norm_layer(512),
                  self.acti_func()]
        
        up_block3 = [nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2, bias=use_bias)]
        up_block2 = [nn.Conv3d(768, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
                     self.norm_layer(256),
                     self.acti_func(),
                     nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
                     self.norm_layer(256),
                     self.acti_func(),
                     nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2, bias=use_bias)]
        up_block1 = [nn.Conv3d(384, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
                     self.norm_layer(128),
                     self.acti_func(),
                     nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
                     self.norm_layer(128),
                     self.acti_func(),
                     nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2, bias=use_bias)]
        out_block = [nn.Conv3d(192, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
                     self.norm_layer(64),
                     self.acti_func(),
                     nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
                     self.norm_layer(64),
                     self.acti_func(),
                     nn.Conv3d(64, self.output_nc, kernel_size=1, bias=use_bias),
                     nn.Softmax(dim=1)]
        
        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(*block2)
        self.block3 = nn.Sequential(*block3)
        self.block4 = nn.Sequential(*block4)
        self.up_block3 = nn.Sequential(*up_block3)
        self.up_block2 = nn.Sequential(*up_block2)
        self.up_block1 = nn.Sequential(*up_block1)
        self.out_block = nn.Sequential(*out_block)
    
    def forward(self, input):
        l1 = self.block1(input)
        l2 = self.block2(l1)
        l3 = self.block3(l2)
        l4 = self.block4(l3)
        l5 = self.up_block3(l4)
        l5 = self.pad_and_concat(l3, l5)
        l6 = self.up_block2(l5)
        l6 = self.pad_and_concat(l2, l6)
        l7 = self.up_block1(l6)
        l7 = self.pad_and_concat(l1, l7)
        output = self.out_block(l7)
        return output
        
    def pad_and_concat(self, layer1, layer2):
        diff_D = layer1.size(2) - layer2.size(2)
        diff_H = layer1.size(3) - layer2.size(3)
        diff_W = layer1.size(4) - layer2.size(4)
        _layer2 = F.pad(layer2, (diff_W // 2, diff_W - diff_W // 2,
                                 diff_H // 2, diff_H - diff_H // 2,
                                 diff_D // 2, diff_D - diff_D // 2,))
        return torch.cat((layer1, _layer2), 1)
        
    def get_norm_layer(self, norm_type='batch', num_groups=0):
        if norm_type == 'batch':
            self.norm_layer = functools.partial(nn.BatchNorm3d)
        elif norm_type == 'instance':
            self.norm_layer = functools.partial(nn.InstanceNorm3d)
        elif norm_type == 'group':
            self.norm_layer = functools.partial(nn.GroupNorm, num_groups)
        elif norm_type == 'none':
            self.norm_layer = None
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    
    def get_acti_func(self, acti_type='relu'):
        if acti_type == 'relu':
            self.acti_func = nn.ReLU
        elif acti_type == 'prelu':
            self.acti_func = nn.PReLU
        else:
            raise NotImplementedError('activative function [%s] is not found' % acti_type)
            
            
class UNet3D(BaseModel):
    """UNet3D: The implementation of 3D U-Net with some modifications
    
    """
    def __init__(self, in_channels, num_classes, num_features=[32, 64, 128, 256, 512], acti_func='ReLU', norm_func='BatchNorm'):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_features = num_features
        
        self.conv_func = functools.partial(nn.Conv3d, kernel_size=3, padding=1)
        
        if acti_func == 'ReLU':
            self.acti_func = nn.ReLU
        elif acti_func == 'PReLU':
            self.acti_func = nn.PReLU
        else:
            raise ValueError("Unsupported activation function {}".format(acti_func))
            
        if norm_func == 'BatchNorm':
            self.norm_func = nn.BatchNorm3d
        elif norm_func == 'GroupNorm':
            self.norm_func = functools.partial(nn.GroupNorm, num_features[0]//2) # num_groups=num_features[0]//2
        else:
            raise ValueError("Unsupported normalization function {}".format(norm_func))
            
        self.inc = inconv(self.in_channels, self.num_features[0], self.num_features[1], self.conv_func, self.acti_func, self.norm_func)
        self.down1 = down(self.num_features[1], self.num_features[2], self.conv_func, self.acti_func, self.norm_func)
        self.down2 = down(self.num_features[2], self.num_features[3], self.conv_func, self.acti_func, self.norm_func)
        self.down3 = down(self.num_features[3], self.num_features[4], self.conv_func, self.acti_func, self.norm_func)
        self.up1 = up(self.num_features[4], self.num_features[3], self.conv_func, self.acti_func, self.norm_func)
        self.up2 = up(self.num_features[3], self.num_features[2], self.conv_func, self.acti_func, self.norm_func)
        self.up3 = up(self.num_features[2], self.num_features[1], self.conv_func, self.acti_func, self.norm_func)
        self.outc = outconv(self.num_features[1], self.num_classes)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return F.softmax(x, dim=1)

class inconv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, conv_func, acti_func, norm_func):
        super(inconv, self).__init__()
        self.conv_func = conv_func
        self.acti_func = acti_func
        self.norm_func = norm_func
        self.conv = nn.Sequential(self.conv_func(in_ch, mid_ch),
                                  self.acti_func(inplace=True),
                                  self.norm_func(mid_ch),
                                  self.conv_func(mid_ch, out_ch),
                                  self.acti_func(inplace=True),
                                  self.norm_func(out_ch))
    def forward(self, x):
        return self.conv(x)

class down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_func, acti_func, norm_func):
        super(down, self).__init__()
        self.conv_func = conv_func
        self.acti_func = acti_func
        self.norm_func = norm_func
        self.mpconv = nn.Sequential(nn.MaxPool3d(2),
                                    self.conv_func(in_ch, in_ch),
                                    self.acti_func(inplace=True),
                                    self.norm_func(in_ch),
                                    self.conv_func(in_ch, out_ch),
                                    self.acti_func(inplace=True),
                                    self.norm_func(out_ch))
    def forward(self, x):
        return self.mpconv(x)
    
class up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_func, acti_func, norm_func):
        super(up, self).__init__()
        self.conv_func = conv_func
        self.acti_func = acti_func
        self.norm_func = norm_func
        self.deconv = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(self.conv_func(in_ch + out_ch, out_ch),
                                  self.acti_func(inplace=True),
                                  self.norm_func(out_ch),
                                  self.conv_func(out_ch, out_ch),
                                  self.acti_func(inplace=True),
                                  self.norm_func(out_ch))

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        
        # input is (N, C, D, H, W)
        d_diff = x2.size(2) - x1.size(2)
        h_diff = x2.size(3) - x1.size(3)
        w_diff = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, (w_diff // 2, w_diff - w_diff//2,
                        h_diff // 2, h_diff - h_diff//2,
                        d_diff // 2, d_diff - d_diff//2))
        x = torch.cat([x1, x2], dim=1)
        y = self.conv(x)
        return y

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
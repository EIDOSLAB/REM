#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

import torch.nn.functional as F
from torch import nn
from torchvision import models


class SegNetTranspose(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNetTranspose, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.num_channels = input_channels
        
        # Encoder layers
        
        self.encoder_conv_00 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.input_channels,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ])
        self.encoder_conv_01 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ])
        self.encoder_conv_10 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ])
        self.encoder_conv_11 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ])
        self.encoder_conv_20 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.encoder_conv_21 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.encoder_conv_22 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.encoder_conv_30 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.encoder_conv_31 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.encoder_conv_32 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.encoder_conv_40 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.encoder_conv_41 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.encoder_conv_42 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        
        self.init_vgg_weights()
        
        # Decoder layers
        
        self.decoder_convtr_42 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.decoder_convtr_41 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.decoder_convtr_40 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.decoder_convtr_32 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.decoder_convtr_31 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.decoder_convtr_30 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.decoder_convtr_22 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.decoder_convtr_21 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.decoder_convtr_20 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ])
        self.decoder_convtr_11 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ])
        self.decoder_convtr_10 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ])
        self.decoder_convtr_01 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ])
        self.decoder_convtr_00 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=self.output_channels,
                               kernel_size=3,
                               padding=1)
        ])
    
    def forward(self, input_img):
        """
        Forward pass `input_img` through the network
        """
        
        # Encoder
        
        # Encoder Stage - 1
        dim_0 = input_img.size()
        x_00 = self.encoder_conv_00(input_img)
        x_01 = self.encoder_conv_01(x_00)
        x_0, indices_0 = F.max_pool2d(x_01, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = self.encoder_conv_10(x_0)
        x_11 = self.encoder_conv_11(x_10)
        x_1, indices_1 = F.max_pool2d(x_11, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = self.encoder_conv_20(x_1)
        x_21 = self.encoder_conv_21(x_20)
        x_22 = self.encoder_conv_22(x_21)
        x_2, indices_2 = F.max_pool2d(x_22, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = self.encoder_conv_30(x_2)
        x_31 = self.encoder_conv_31(x_30)
        x_32 = self.encoder_conv_32(x_31)
        x_3, indices_3 = F.max_pool2d(x_32, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 5
        dim_4 = x_3.size()
        x_40 = self.encoder_conv_40(x_3)
        x_41 = self.encoder_conv_41(x_40)
        x_42 = self.encoder_conv_42(x_41)
        x_4, indices_4 = F.max_pool2d(x_42, kernel_size=2, stride=2, return_indices=True)
        
        # Decoder
        
        # Decoder Stage - 5
        x_4d = F.interpolate(x_4, scale_factor=2)
        x_42d = self.decoder_convtr_42(x_4d)
        x_41d = self.decoder_convtr_41(x_42d)
        x_40d = self.decoder_convtr_40(x_41d)
        
        # Decoder Stage - 4
        x_3d = F.interpolate(x_40d, scale_factor=2)
        x_32d = self.decoder_convtr_32(x_3d)
        x_31d = self.decoder_convtr_31(x_32d)
        x_30d = self.decoder_convtr_30(x_31d)
        
        # Decoder Stage - 3
        x_2d = F.interpolate(x_30d, scale_factor=2)
        x_22d = self.decoder_convtr_22(x_2d)
        x_21d = self.decoder_convtr_21(x_22d)
        x_20d = self.decoder_convtr_20(x_21d)
        
        # Decoder Stage - 2
        x_1d = F.interpolate(x_20d, scale_factor=2)
        x_11d = self.decoder_convtr_11(x_1d)
        x_10d = self.decoder_convtr_10(x_11d)
        
        # Decoder Stage - 1
        x_0d = F.interpolate(x_10d, scale_factor=2)
        x_01d = self.decoder_convtr_01(x_0d)
        x_00d = self.decoder_convtr_00(x_01d)
        
        x_softmax = F.softmax(x_00d, dim=1)
        
        return x_00d
    
    def init_vgg_weights(self):
        vgg16 = models.vgg16(pretrained=True)
        
        assert self.encoder_conv_00[0].weight.size() == vgg16.features[0].weight.size()
        self.encoder_conv_00[0].weight.data = vgg16.features[0].weight.data
        assert self.encoder_conv_00[0].bias.size() == vgg16.features[0].bias.size()
        self.encoder_conv_00[0].bias.data = vgg16.features[0].bias.data
        
        assert self.encoder_conv_01[0].weight.size() == vgg16.features[2].weight.size()
        self.encoder_conv_01[0].weight.data = vgg16.features[2].weight.data
        assert self.encoder_conv_01[0].bias.size() == vgg16.features[2].bias.size()
        self.encoder_conv_01[0].bias.data = vgg16.features[2].bias.data
        
        assert self.encoder_conv_10[0].weight.size() == vgg16.features[5].weight.size()
        self.encoder_conv_10[0].weight.data = vgg16.features[5].weight.data
        assert self.encoder_conv_10[0].bias.size() == vgg16.features[5].bias.size()
        self.encoder_conv_10[0].bias.data = vgg16.features[5].bias.data
        
        assert self.encoder_conv_11[0].weight.size() == vgg16.features[7].weight.size()
        self.encoder_conv_11[0].weight.data = vgg16.features[7].weight.data
        assert self.encoder_conv_11[0].bias.size() == vgg16.features[7].bias.size()
        self.encoder_conv_11[0].bias.data = vgg16.features[7].bias.data
        
        assert self.encoder_conv_20[0].weight.size() == vgg16.features[10].weight.size()
        self.encoder_conv_20[0].weight.data = vgg16.features[10].weight.data
        assert self.encoder_conv_20[0].bias.size() == vgg16.features[10].bias.size()
        self.encoder_conv_20[0].bias.data = vgg16.features[10].bias.data
        
        assert self.encoder_conv_21[0].weight.size() == vgg16.features[12].weight.size()
        self.encoder_conv_21[0].weight.data = vgg16.features[12].weight.data
        assert self.encoder_conv_21[0].bias.size() == vgg16.features[12].bias.size()
        self.encoder_conv_21[0].bias.data = vgg16.features[12].bias.data
        
        assert self.encoder_conv_22[0].weight.size() == vgg16.features[14].weight.size()
        self.encoder_conv_22[0].weight.data = vgg16.features[14].weight.data
        assert self.encoder_conv_22[0].bias.size() == vgg16.features[14].bias.size()
        self.encoder_conv_22[0].bias.data = vgg16.features[14].bias.data
        
        assert self.encoder_conv_30[0].weight.size() == vgg16.features[17].weight.size()
        self.encoder_conv_30[0].weight.data = vgg16.features[17].weight.data
        assert self.encoder_conv_30[0].bias.size() == vgg16.features[17].bias.size()
        self.encoder_conv_30[0].bias.data = vgg16.features[17].bias.data
        
        assert self.encoder_conv_31[0].weight.size() == vgg16.features[19].weight.size()
        self.encoder_conv_31[0].weight.data = vgg16.features[19].weight.data
        assert self.encoder_conv_31[0].bias.size() == vgg16.features[19].bias.size()
        self.encoder_conv_31[0].bias.data = vgg16.features[19].bias.data
        
        assert self.encoder_conv_32[0].weight.size() == vgg16.features[21].weight.size()
        self.encoder_conv_32[0].weight.data = vgg16.features[21].weight.data
        assert self.encoder_conv_32[0].bias.size() == vgg16.features[21].bias.size()
        self.encoder_conv_32[0].bias.data = vgg16.features[21].bias.data
        
        assert self.encoder_conv_40[0].weight.size() == vgg16.features[24].weight.size()
        self.encoder_conv_40[0].weight.data = vgg16.features[24].weight.data
        assert self.encoder_conv_40[0].bias.size() == vgg16.features[24].bias.size()
        self.encoder_conv_40[0].bias.data = vgg16.features[24].bias.data
        
        assert self.encoder_conv_41[0].weight.size() == vgg16.features[26].weight.size()
        self.encoder_conv_41[0].weight.data = vgg16.features[26].weight.data
        assert self.encoder_conv_41[0].bias.size() == vgg16.features[26].bias.size()
        self.encoder_conv_41[0].bias.data = vgg16.features[26].bias.data
        
        assert self.encoder_conv_42[0].weight.size() == vgg16.features[28].weight.size()
        self.encoder_conv_42[0].weight.data = vgg16.features[28].weight.data
        assert self.encoder_conv_42[0].bias.size() == vgg16.features[28].bias.size()
        self.encoder_conv_42[0].bias.data = vgg16.features[28].bias.data
        
        del vgg16


class SegNetConv(nn.Module):
    def __init__(self, input_channels, output_channels, bn=False):
        super(SegNetConv, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Encoder layers
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=self.output_channels, kernel_size=3, padding=1),
        )
        
        self.init_vgg_weights()
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
    
    def init_vgg_weights(self):
        vgg16 = models.vgg16(pretrained=True)
        
        vgg_layers = [m for m in vgg16.modules() if isinstance(m, nn.Conv2d)]
        segnet_layers = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        
        for v, s in zip(vgg_layers, segnet_layers):
            assert s.weight.size() == v.weight.size()
            s.weight.data = v.weight.data
            assert s.bias.size() == v.bias.size()
            s.bias.data = v.bias.data
        
        del vgg16, vgg_layers, segnet_layers


if __name__ == '__main__':
    model = SegNetConv(3, 1)

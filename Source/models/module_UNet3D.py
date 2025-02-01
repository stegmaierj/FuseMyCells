# -*- coding: utf-8 -*-
"""
Implementation of the 3D UNet architecture.
https://arxiv.org/pdf/1606.06650.pdf
"""

import torch
import torch.nn as nn


class module_UNet3D(nn.Module):
    """
        Implementation of the 3D U-Net architecture.
    """

    def __init__(self, patch_size, in_channels, out_channels, feat_channels=16, out_activation='sigmoid', layer_norm='none', **kwargs):
        super(module_UNet3D, self).__init__()
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.layer_norm = layer_norm # instance | batch | none
        self.out_activation = out_activation # relu | leakyrelu | sigmoid | tanh | hardtanh | none
        
        self.norm_methods = {
            'instance': nn.InstanceNorm3d,
            'batch': nn.BatchNorm3d,
            'none': nn.Identity
            }
        
        self.out_activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(negative_slope=0.2, inplace=True),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'hardtanh': nn.Hardtanh(0,1),
            'none': nn.Identity()
            })     
                
        
        # Define layer instances       
        self.c1 = nn.Sequential(
            nn.Conv3d(in_channels, feat_channels//2, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels//2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels//2, feat_channels, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.d1 = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=4, stride=2, padding=1),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )


        self.c2 = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels*2, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.d2 = nn.Sequential(
            nn.Conv3d(feat_channels*2, feat_channels*2, kernel_size=4, stride=2, padding=1),
            self.norm_methods[self.layer_norm](feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.c3 = nn.Sequential(
            nn.Conv3d(feat_channels*2, feat_channels*2, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*2, feat_channels*4, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.d3 = nn.Sequential(
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=4, stride=2, padding=1),
            self.norm_methods[self.layer_norm](feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )


        self.c4 = nn.Sequential(
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*4, feat_channels*8, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.u1 = nn.Sequential(
            nn.ConvTranspose3d(feat_channels*8, feat_channels*8, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.norm_methods[self.layer_norm](feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.c5 = nn.Sequential(
            nn.Conv3d(feat_channels*12, feat_channels*4, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.u2 = nn.Sequential(
            nn.ConvTranspose3d(feat_channels*4, feat_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.norm_methods[self.layer_norm](feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.c6 = nn.Sequential(
            nn.Conv3d(feat_channels*6, feat_channels*2, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*2, feat_channels*2, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.u3 = nn.Sequential(
            nn.ConvTranspose3d(feat_channels*2, feat_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.norm_methods[self.layer_norm](feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*2, feat_channels*2, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.c7 = nn.Sequential(
            nn.Conv3d(feat_channels*3, feat_channels, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.out = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, out_channels, kernel_size=1),
            self.out_activations[self.out_activation]
            )
        

    def forward(self, img):
        
        c1 = self.c1(img)
        d1 = self.d1(c1)
        
        c2 = self.c2(d1)
        d2 = self.d2(c2)
        
        c3 = self.c3(d2)
        d3 = self.d3(c3)
        
        c4 = self.c4(d3)
        
        u1 = self.u1(c4)
        c5 = self.c5(torch.cat((u1,c3),1))
        
        u2 = self.u2(c5)
        c6 = self.c6(torch.cat((u2,c2),1))
        
        u3 = self.u3(c6)
        c7 = self.c7(torch.cat((u3,c1),1))
        
        out = self.out(c7)
        
        return out
        
# -*- coding: utf-8 -*-
"""
Implementation of the 3D ResNet architecture with bottleneck blocks.
https://arxiv.org/pdf/1512.03385.pdf
"""

import torch
import torch.nn as nn


class module_ResNet3D_bottleneck(nn.Module):

    def __init__(self, patch_size, in_channels, out_channels, feat_channels=16, out_activation='sigmoid', layer_norm='none', **kwargs):
        super(module_ResNet3D_bottleneck, self).__init__()
        
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
        
        self.input = nn.Sequential(
            nn.Conv3d(in_channels, feat_channels, kernel_size=7, padding=1),
            self.norm_methods[self.layer_norm](feat_channels)
            )
        
        self.conv1 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels//4, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels//4, feat_channels//4, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels//4, feat_channels, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels)
            )
        
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels//4, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels//4, feat_channels//4, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels//4, feat_channels, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels),
            )
        
        self.project1 = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels*2, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels*2)
            )
        
        self.conv3 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*2, feat_channels//2, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels//2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels//2, feat_channels//2, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels//2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels//2, feat_channels*2, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels*2)
            )
        
        self.conv4 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*2, feat_channels//2, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels//2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels//2, feat_channels//2, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels//2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels//2, feat_channels*2, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels*2)
            )
        
        self.project2 = nn.Sequential(
            nn.Conv3d(feat_channels*2, feat_channels*4, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels*4)
            )
        
        self.conv5 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*4, feat_channels, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels*4, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels*4)
            )
        
        self.conv6 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*4, feat_channels, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels*4, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels*4)
            )
        
        self.project3 = nn.Sequential(
            nn.Conv3d(feat_channels*4, feat_channels*8, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels*8)
            )
        
        self.out = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*8, feat_channels*2, kernel_size=1),
            self.norm_methods[self.layer_norm](feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*2, feat_channels*2, kernel_size=3, padding=1),
            self.norm_methods[self.layer_norm](feat_channels*2),
            nn.Conv3d(feat_channels*2, out_channels, kernel_size=1),
            self.out_activations[self.out_activation]
            )
       
        

    def forward(self, img):
        
        inp = self.input(img)
        
        conv1 = self.conv1(inp)
        skip1 = torch.add(inp, conv1)
        
        conv2 = self.conv2(skip1)
        skip2 = torch.add(skip1, conv2)
        
        proj1 = self.project1(skip2)
        
        conv3 = self.conv3(proj1)
        skip3 = torch.add(proj1, conv3)
        
        conv4 = self.conv4(skip3)
        skip4 = torch.add(skip3, conv4)
        
        proj2 = self.project2(skip4)
        
        conv5 = self.conv5(proj2)
        skip5 = torch.add(proj2, conv5)
        
        conv6 = self.conv6(skip5)
        skip4 = torch.add(skip5, conv6)
        
        proj3 = self.project3(skip4)
                
        out = self.out(proj3)
        
        return out
        
    
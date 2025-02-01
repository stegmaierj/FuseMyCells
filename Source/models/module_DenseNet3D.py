'''
Adapted from https://arxiv.org/abs/1608.06993
'''


import torch
import torch.nn as nn


class module_DenseNet3D(nn.Module):
    
    #Implementing denseley connected Network
    def __init__(self, in_channels, out_channels,feat_channels, coefficients=1,out_activation='sigmoid', layer_norm='batch', **kwargs):
        super(module_DenseNet3D,self).__init__()
      
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.coefficients=coefficients
        self.feat_channels=feat_channels
        self.layer_norm=layer_norm
        self.out_activation = out_activation

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
            
        #Preperation
        self.c1 = nn.Sequential(
            nn.Conv3d(in_channels, feat_channels, kernel_size=3, stride=1, padding=1),
            self.norm_methods[self.layer_norm](num_features=feat_channels),
            nn.ReLU(),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1)
        )

        #Denseblock 1
        self.c2 = nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels),
            nn.ReLU(),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1)
        )

        self.c3 = nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels*2),
            nn.ReLU(),
            nn.Conv3d(feat_channels*2, feat_channels, kernel_size=3, stride=1, padding=1)
        )

        self.c4 = nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels*3),
            nn.ReLU(),
            nn.Conv3d(feat_channels*3,feat_channels, kernel_size=3, stride=1, padding=1)
        )

        self.c5=nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels*4),
            nn.ReLU(),
            nn.Conv3d(feat_channels*4,feat_channels, kernel_size=3, stride=1, padding=1)
        )
        
        #Transistion Layer 1
        self.t1 = nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels*5),
            nn.ReLU(),
            nn.Conv3d(feat_channels*5,feat_channels, kernel_size=1),
            self.norm_methods[self.layer_norm](num_features=feat_channels*4),
            nn.ReLU(),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1)
        )

        #Denseblock 2
        self.c6 = nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels),
            nn.ReLU(),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1)
        )

        self.c7 = nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels*2),
            nn.ReLU(),
            nn.Conv3d(feat_channels*2, feat_channels, kernel_size=3, stride=1, padding=1)
        )

        self.c8 = nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels*3),
            nn.ReLU(),
            nn.Conv3d(feat_channels*3,feat_channels, kernel_size=3, stride=1, padding=1)
        )

        self.c9=nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels*4),
            nn.ReLU(),
            nn.Conv3d(feat_channels*4,feat_channels, kernel_size=3, stride=1, padding=1)
        )
        
        #Transistion Layer 2
        self.t2 = nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels*5),
            nn.ReLU(),
            nn.Conv3d(feat_channels*5,feat_channels, kernel_size=1),
            self.norm_methods[self.layer_norm](num_features=feat_channels*4),
            nn.ReLU(),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1)
        )

        #Denseblock 3
        self.c10 = nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels),
            nn.ReLU(),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1),
        )

        self.c11 = nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels*2),
            nn.ReLU(),
            nn.Conv3d(feat_channels*2, feat_channels, kernel_size=3, stride=1, padding=1)
        )

        self.c12 = nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels*3),
            nn.ReLU(),
            nn.Conv3d(feat_channels*3,feat_channels, kernel_size=3, stride=1, padding=1)
        )

        self.c13 = nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels*4),
            nn.ReLU(),
            nn.Conv3d(feat_channels*4,feat_channels, kernel_size=3, stride=1, padding=1)
        )

        #Transistion Layer 3
        self.t3 = nn.Sequential(
            self.norm_methods[self.layer_norm](num_features=feat_channels*5),
            nn.ReLU(),
            nn.Conv3d(feat_channels*5,out_channels, kernel_size=1),
            self.norm_methods[self.layer_norm](num_features=out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )



        

    def forward(self, img):
        
        # Preparations
        c1 = self.c1(img)
        
        # Denseblock 1
        c2=self.c2(c1)
        c3=self.c3(torch.cat((c1,c2),1))
        c4=self.c4(torch.cat((c1,c2,c3),1))
        c5=self.c5(torch.cat((c1,c2,c3,c4),1))

        # Transition
        t1=self.t1(torch.cat((c1,c2,c3,c4,c5),1))

        # Denseblock 2
        c6=self.c6(t1)
        c7=self.c7(torch.cat((t1,c6),1))
        c8=self.c8(torch.cat((t1,c6,c7),1))
        c9=self.c9(torch.cat((t1,c6,c7,c8),1))

        # Transition
        t2=self.t2(torch.cat((t1,c6,c7,c8,c9),1))

        # Denseblock 3
        c10=self.c10(t2)
        c11=self.c11(torch.cat((t2,c10),1))
        c12=self.c12(torch.cat((t2,c10,c11),1))
        c13=self.c13(torch.cat((t2,c10,c11,c12),1))

        # Out-transition
        out=self.t3(torch.cat((t2,c10,c11,c12,c13),1))

        return out

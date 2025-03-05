# -*- coding: utf-8 -*-
"""
Implementation of the multi-class pipeline to support watershed-based segmentation.
https://arxiv.org/pdf/1810.06933.pdf
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchmetrics.image import StructuralSimilarityIndexMeasure

from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from torch.utils.data import DataLoader
from resources.dataloader.h5_dataloader_docker import MeristemH5Dataset

class ImageFusionUNet3D(pl.LightningModule):
    
    def __init__(self, hparams):
        super(ImageFusionUNet3D, self).__init__()
        
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)
        self.augmentation_dict = {}
        self.ssim_weight = hparams.ssim_weight

        # load the backbone network architecture
        if self.hparams.backbone.lower() == 'unet3d':        
            from resources.models.module_UNet3D import module_UNet3D as backbone
        elif self.hparams.backbone.lower() == 'unet3d_pixelshuffle':
            from models.module_UNet3D_pixelshuffle import module_UNet3D_pixelshuffle as backbone
        elif self.hparams.backbone.lower() == 'resnet3d':
            from models.module_ResNet3D import module_ResNet3D as backbone
        elif self.hparams.backbone.lower() == 'resnet3d_bottleneck':
            from models.module_ResNet3D_bottleneck import module_ResNet3D_bottleneck as backbone
        elif self.hparams.backbone.lower() == 'densenet3d':
            from models.module_DenseNet3D import module_DenseNet3D as backbone
        else:
            raise ValueError('Unknown backbone architecture {0}!'.format(self.hparams.backbone))
            
        self.network = backbone(patch_size=hparams.patch_size, in_channels=hparams.in_channels, out_channels=hparams.out_channels, feat_channels=hparams.feat_channels, out_activation=hparams.out_activation, layer_norm=hparams.layer_norm)
        
        # cache for generated images
        self.last_predictions = None
        self.last_imgs = None
        self.last_masks = None


    def forward(self, z):
        return self.network(z)
    
    
    def load_pretrained(self, pretrained_file, strict=True, verbose=True):
        
        # Load the state dict
        state_dict = torch.load(pretrained_file)['state_dict']
        
        # Make sure to have a weight dict
        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)
            
        # Get parameter dict of current model
        param_dict = dict(self.network.named_parameters())
        
        layers = []
        for layer in param_dict:
            if strict and not 'network.'+layer in state_dict:
                if verbose:
                    print('Could not find weights for layer "{0}"'.format(layer))
                continue
            try:
                param_dict[layer].data.copy_(state_dict['network.'+layer].data)
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print('Error at layer {0}:\n{1}'.format(layer, e))
        
        self.network.load_state_dict(param_dict)
        
        if verbose:
            print('Loaded weights for the following layers:\n{0}'.format(layers))
        
        
    def background_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)


    def boundary_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    
    def seed_loss(self, y_hat, y):
        loss = F.mse_loss(y_hat, y, reduction='none')
        weight = torch.clamp(y, min=0.1, max=1.0)
        loss = torch.mul(loss, weight)        
        loss = torch.sum(loss)/torch.sum(weight)
        return loss


    # loss based on the ssim between prediction and target
    def ssim_loss(self, y_hat, y):

        # get the losses
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        ssim.cuda()

        loss_ssim = self.ssim_weight * (1.0 - ssim(y_hat, y))
        loss_l1 = F.l1_loss(y_hat, y)

        loss = loss_ssim + loss_l1

        return loss, loss_l1, loss_ssim

    # loss based on the ssim ratio between ground truth and input
    # n_ssim = (prediction_ssim - reference_ssim) / (1 - reference_ssim)
    def fmc_loss(self, x, y_hat, y):

        if x.shape[1] > 1:
            x = x[:,[0], ...]

        # get the losses
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        ssim.cuda()

        prediction_ssim = ssim(y_hat, y)
        reference_ssim = ssim(x, y)

        loss = (prediction_ssim - reference_ssim) / (1.0 - reference_ssim)

        # return the negative variant of this loss to minimize it
        return -loss + F.l1_loss(y_hat, y)


    def training_step(self, batch, batch_idx):
        
        # Get image ans mask of current batch
        self.last_imgs, self.last_masks = batch['image'], batch['mask']
        
        # generate images
        self.predictions = self.forward(self.last_imgs)
                        
        if not self.hparams.use_fmc_loss:

            loss, loss_l1, loss_ssim = self.ssim_loss(self.predictions, self.last_masks)

            self.logger.experiment.add_scalar('ssim', loss_ssim, self.current_epoch)
            self.logger.experiment.add_scalar('l1', loss_l1, self.current_epoch)
            self.logger.experiment.add_scalar('loss', loss, self.current_epoch)
        else:
            loss = self.fmc_loss(self.last_imgs, self.predictions, self.last_masks)
            self.logger.experiment.add_scalar('loss', loss, self.current_epoch)
        
        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_hat = self.forward(x)

        if not self.hparams.use_fmc_loss:
            test_loss, test_loss_l1, test_loss_ssim = self.ssim_loss(y_hat, y)
            return {'test_loss_ssim': test_loss_ssim, 'test_loss_l1': test_loss_l1, 'test_loss': test_loss}
        else:
            test_loss = self.fmc_loss(x, y_hat, y)
            return {'test_loss': test_loss}


    def test_end(self, outputs):

        if not self.hparams.use_fmc_loss:
            avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
            avg_loss_ssim = torch.stack([x['test_loss_ssim'] for x in outputs]).mean()
            avg_loss_l1 = torch.stack([x['test_loss_l1'] for x in outputs]).mean()

            tensorboard_logs = {'test_loss': avg_loss, 'test_loss_l1': avg_loss_l1, 'test_loss_ssim': avg_loss_ssim}
            return {'avg_test_loss': avg_loss, 'avg_test_loss_l1': avg_loss_l1, 'avg_test_loss_ssim': avg_loss_ssim, 'log': tensorboard_logs}

        else:
            avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

            tensorboard_logs = {'test_loss': avg_loss}
            return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_hat = self.forward(x)

        if not self.hparams.use_fmc_loss:
            val_loss, val_loss_l1, val_loss_ssim = self.ssim_loss(y_hat, y)
            return {'val_loss_ssim': val_loss_ssim, 'val_loss_l1': val_loss_l1, 'val_loss': val_loss}
        else:
            val_loss = self.fmc_loss(x, y_hat, y)
            return {'val_loss': val_loss}

    def validation_end(self, outputs):
        if not self.hparams.use_fmc_loss:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            avg_loss_ssim = torch.stack([x['val_loss_ssim'] for x in outputs]).mean()
            avg_loss_l1 = torch.stack([x['val_loss_l1'] for x in outputs]).mean()

            tensorboard_logs = {'val_loss': avg_loss, 'val_loss_l1': avg_loss_l1, 'val_loss_ssim': avg_loss_ssim}
            return {'avg_val_loss': avg_loss, 'avg_val_loss_l1': avg_loss_l1, 'avg_val_loss_ssim': avg_loss_ssim, 'log': tensorboard_logs}
        else:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            tensorboard_logs = {'val_loss': avg_loss}
            return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        opt = torch.optim.RAdam(self.network.parameters(), lr=self.hparams.learning_rate)
        return [opt], []

    def train_dataloader(self):
         if self.hparams.train_list is None:
            return None
         else:
            dataset = MeristemH5Dataset(self.hparams.train_list, self.hparams.data_root, patch_size=self.hparams.patch_size,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, augmentation_dict=self.augmentation_dict,\
                                        dist_handling=self.hparams.dist_handling, seed_handling='float', data_norm=self.hparams.data_norm, samples_per_epoch=self.hparams.samples_per_epoch, binary_mask=False, patches_from_fg=0.75)
            return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)
    
    def test_dataloader(self):
        if self.hparams.test_list is None:
            return None
        else:
            dataset = MeristemH5Dataset(self.hparams.test_list, self.hparams.data_root, patch_size=self.hparams.patch_size,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, augmentation_dict={},\
                                        dist_handling=self.hparams.dist_handling, seed_handling='float', data_norm=self.hparams.data_norm, binary_mask=False, patches_from_fg=0.75)
            return DataLoader(dataset, batch_size=self.hparams.batch_size)
    
    def val_dataloader(self):
        if self.hparams.val_list is None:
            return None
        else:
            dataset = MeristemH5Dataset(self.hparams.val_list, self.hparams.data_root, patch_size=self.hparams.patch_size,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, augmentation_dict={},\
                                        dist_handling=self.hparams.dist_handling, seed_handling='float', data_norm=self.hparams.data_norm, binary_mask=False, patches_from_fg=0.75)
            return DataLoader(dataset, batch_size=self.hparams.batch_size)


    def on_train_epoch_end(self):
        
        # log sampled images
        predictions = self.forward(self.last_imgs)

        selected_prediction = predictions[:,:,int(self.hparams.patch_size[0]//2),:,:]
        selected_prediction = (selected_prediction - torch.min(selected_prediction[:])) / (torch.max(selected_prediction[:]) - torch.min(selected_prediction[:]))
        selected_prediction = torch.clip(selected_prediction, torch.tensor(0.0).cuda(), torch.tensor(1.0).cuda())

        prediction_grid = torchvision.utils.make_grid(selected_prediction)
        self.logger.experiment.add_image('generated_images', prediction_grid, self.current_epoch)

        selected_raw_image = self.last_imgs[:,0,int(self.hparams.patch_size[0]//2),:,:]
        selected_raw_image = (selected_raw_image - torch.min(selected_raw_image[:])) / (torch.max(selected_raw_image[:]) - torch.min(selected_raw_image[:]))
        selected_raw_image = torch.clip(selected_raw_image, torch.tensor(0.0).cuda(), torch.tensor(1.0).cuda())
        
        img_grid = torchvision.utils.make_grid(selected_raw_image)
        self.logger.experiment.add_image('raw_images', img_grid, self.current_epoch)

        selected_target_image = self.last_masks[:,:,int(self.hparams.patch_size[0]//2),:,:]
        selected_target_image = (selected_target_image - torch.min(selected_target_image[:])) / (torch.max(selected_target_image[:]) - torch.min(selected_target_image[:]))
        selected_target_image = torch.clip(selected_target_image, torch.tensor(0.0).cuda(), torch.tensor(1.0).cuda())
        
        mask_grid = torchvision.utils.make_grid(selected_target_image)
        self.logger.experiment.add_image('target_masks', mask_grid, self.current_epoch)
        
        
    def set_augmentations(self, augmentation_dict_file):
        if not augmentation_dict_file is None:
            self.augmentation_dict = json.load(open(augmentation_dict_file))
        
        
    @staticmethod
    def add_model_specific_args(parent_parser): 
        """
        Parameters you define here will be available to your model through self.hparams
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        parser.add_argument('--backbone', default='UNet3D', type=str, help='which model to load (UNet3D | UNet3D_PixelShuffle | ResNet3D | ResNet3D_bottleneck | DenseNet3D)')
        parser.add_argument('--in_channels', default=4, type=int)
        parser.add_argument('--out_channels', default=1, type=int)
        parser.add_argument('--feat_channels', default=16, type=int)
        parser.add_argument('--patch_size', default=(64,256,256), type=int, nargs='+')
        parser.add_argument('--out_activation', default='none', type=str)
        parser.add_argument('--layer_norm', default='instance', type=str)

        # data
        parser.add_argument('--data_norm', default='fmc_percentile', type=str)        
        parser.add_argument('--data_root', default=r'/netshares/BiomedicalImageAnalysis/Resources/FuseMyCells_ISBIChallenge/Data/', type=str) 
        parser.add_argument('--train_list', default=r'/work/scratch/stegmaier/Projects/2025/FuseMyCellsISBI_ImageFusion/Source/Source/data/Study1_train.csv', type=str)
        parser.add_argument('--test_list', default=r'/work/scratch/stegmaier/Projects/2025/FuseMyCellsISBI_ImageFusion/Source/Source/data/Study1_test.csv', type=str)
        parser.add_argument('--val_list', default=r'/work/scratch/stegmaier/Projects/2025/FuseMyCellsISBI_ImageFusion/Source/Source/data/Study1_val.csv', type=str)
        parser.add_argument('--image_groups', default=('data/raw_image', 'data/normalized_intensity', 'data/surface_distance', 'data/light_map'), type=str, nargs='+')
        parser.add_argument('--mask_groups', default=('data/raw_image',), type=str, nargs='+')
        parser.add_argument('--dist_handling', default='none', type=str)
        
        # training params
        parser.add_argument('--samples_per_epoch', default=-1, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--background_weight', default=1, type=float)
        parser.add_argument('--seed_weight', default=100, type=float)
        parser.add_argument('--boundary_weight', default=1, type=float)
        parser.add_argument('--ssim_weight', default=100, type=float)
        parser.add_argument('--use_fmc_loss', dest='use_fmc_loss', action='store_true', default=False, help='Use FMC loss based on SSIM ratios')
        
        return parser
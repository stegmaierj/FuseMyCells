#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:46:20 2020

@author: eschweiler
"""

import os
import numpy as np
import torch
import csv
from skimage import io
from argparse import ArgumentParser
from torch.autograd import Variable

from resources.dataloader.h5_dataloader_docker import MeristemH5Tiler as Tiler
from resources.utils.utils import print_timestamp

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)


def fmc_process_image(hparams, network):
    
    # set hyperparameters for FMC
    hparams.crop = (5, 15, 15)
    hparams.overlap = (5, 15, 15)
    hparams.gpus = 1
    hparams.patch_size = (64, 256, 256)

    if not torch.cuda.is_available():
        hparams.gpus = 0

    #hparams.test_list = "/work/scratch/stegmaier/Projects/2025/FuseMyCellsISBI_ImageFusion/Source/Source/data/Study1_membrane_test_singleFile.csv"

    """
    Main testing routine specific for this project
    :param hparams:
    """

    # ------------------------
    # 0 SANITY CHECKS
    # ------------------------
    if not isinstance(hparams.overlap, (tuple, list)):
        hparams.overlap = (hparams.overlap,) * len(hparams.patch_size)
    if not isinstance(hparams.crop, (tuple, list)):
        hparams.crop = (hparams.crop,) * len(hparams.patch_size)
    assert all([p-2*o-2*c>0 for p,o,c in zip(hparams.patch_size, hparams.overlap, hparams.crop)]), 'Invalid combination of patch size, overlap and crop size.'
    
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = network(hparams=hparams)
    #model = model.load_from_checkpoint(hparams.ckpt_path)
    model.load_pretrained(pretrained_file=hparams.ckpt_path)

    if hparams.gpus > 0:
        model = model.cuda()
    
    # ------------------------
    # 2 INIT DATA TILER
    # ------------------------
    tiler = Tiler(hparams.test_list, no_mask=hparams.input_batch=='image', no_img=hparams.input_batch=='mask', **vars(hparams))
    fading_map = tiler.get_fading_map()
    fading_map = np.repeat(fading_map[np.newaxis,...], hparams.out_channels, axis=0)
    
    # ------------------------
    # 3 FILE AND FOLDER CHECKS
    # ------------------------
    #os.makedirs(hparams.output_path, exist_ok=True)
    #file_checklist = []
    
    # ------------------------
    # 4 PROCESS EACH IMAGE
    # ------------------------
    if hparams.num_files is None or hparams.num_files < 0:
        hparams.num_files = len(tiler.data_list)
    else:
        hparams.num_files = np.minimum(len(tiler.data_list), hparams.num_files)
        
    for image_idx in range(hparams.num_files):
        
        # Check if current file has already been processed
        # Initialize current file        
        tiler.set_data_idx(image_idx)
        
        # Determine if the patch size exceeds the image size
        working_size = tuple(np.max(np.array(tiler.locations), axis=0) - np.min(np.array(tiler.locations), axis=0) + np.array(hparams.patch_size))
            
        # Initialize maps      
        predicted_img = np.full((hparams.out_channels,)+working_size, 0, dtype=np.float32)        
        norm_map = np.full((hparams.out_channels,)+working_size, 0, dtype=np.float32)
                
        for patch_idx in range(tiler.__len__()):
            
            print_timestamp('Processing patch {0}/{1}...',(patch_idx+1, tiler.__len__()))
            
            # Get the input
            sample = tiler.__getitem__(patch_idx)
            data = None
            if hparams.gpus > 0:
                data = Variable(torch.from_numpy(sample[hparams.input_batch][np.newaxis,...]).cuda())
            else:
                data = Variable(torch.from_numpy(sample[hparams.input_batch][np.newaxis,...]))
            data = data.float()
            
            # Add noise, if necessary
            if hparams.add_noise_channel > -2:
                if hparams.gpus > 0:
                    noise = torch.randn(data.shape[0:1]+(1,)+data.shape[2:]).cuda(data.device.index)
                else:
                    noise = torch.randn(data.shape[0:1]+(1,)+data.shape[2:])
                    
                if hparams.add_noise_channel == -1:
                    data = torch.cat((data, noise), 1)
                elif hparams.add_noise_channel > -1:
                    data[:,hparams.add_noise_channel:hparams.add_noise_channel+1,...] += noise
            
            # Predict the output
            pred_patch = model(data)
            pred_patch = pred_patch.cpu().data.numpy()
            pred_patch = np.squeeze(pred_patch)
            pred_patch = np.clip(pred_patch, hparams.clip[0], hparams.clip[1])
                        
            # Get the current slice position
            slicing = tuple(map(slice, (0,)+tuple(tiler.patch_start+tiler.global_crop_before), (hparams.out_channels,)+tuple(tiler.patch_end+tiler.global_crop_before)))
                        
            # Add predicted patch and fading weights to the corresponding maps
            predicted_img[slicing] = predicted_img[slicing]+pred_patch*fading_map
            norm_map[slicing] = norm_map[slicing]+fading_map
            
        # Normalize the predicted image
        norm_map = np.clip(norm_map, 1e-5, np.inf)
        predicted_img = predicted_img / norm_map          
        
        # Crop the predicted image to its original size
        slicing = tuple(map(slice, (0,)+tuple(tiler.global_crop_before), (hparams.out_channels,)+tuple(np.array(predicted_img.shape[1:])+np.array(tiler.global_crop_after))))
        predicted_img = predicted_img[slicing]
        
        # Save the predicted image
        predicted_img = np.transpose(predicted_img, (1,2,3,0))
        predicted_img = predicted_img.astype(np.float32)

        return predicted_img

def fmc_entry_point(input_image_file, ckpt_path, in_channels, image_groups):

    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '--output_path',
        type=str,
        default=r'/work/scratch/stegmaier/Projects/2025/FuseMyCellsISBI_ImageFusion/Processing/ApplyTest',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--ckpt_path',
        type=str,
        default=r'/work/scratch/stegmaier/Projects/2025/FuseMyCellsISBI_ImageFusion/Processing/Study1_Membrane_1_1/imagefusionunet3d-epoch=291-step=16936.ckpt',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of GPUs to use'
    )
    
    parent_parser.add_argument(
        '--overlap',
        type=int,
        default=(5,5,5),
        help='overlap of adjacent patches',
        nargs='+'
    )
    
    parent_parser.add_argument(
        '--crop',
        type=int,
        default=(5,5,5),
        help='safety crop of patches',
        nargs='+'
    )
    
    parent_parser.add_argument(
        '--input_batch',
        type=str,
        default='image',
        help='which part of the batch is used as input (image | mask)'
    )
    
    parent_parser.add_argument(
        '--clip',
        type=float,
        default=(-1.0, 1.0),
        help='clipping values for network outputs',
        nargs='+'
    )
    
    parent_parser.add_argument(
        '--num_files',
        type=int,
        default=-1,
        help='number of files to process'
    )
    
    parent_parser.add_argument(
        '--add_noise_channel',
        type=int,
        default=-2,
        help='add noise to patch'
    )
    
    parent_parser.add_argument(
        '--pipeline',
        type=str,
        default='ImageFusionUNet3D',
        help='which pipeline to load ImageFusionUNet3D'
    )
        
    parent_args = parent_parser.parse_known_args()[0]
    
    # load the desired network architecture
    if parent_args.pipeline.lower() == 'imagefusionunet3d':
        from resources.models.ImageFusionUNet3D import ImageFusionUNet3D as network
    else:
        raise ValueError('Unknown pipeline "{0}".'.format(parent_args.pipeline))
        
    # each LightningModule defines arguments relevant to it
    parser = network.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    hyperparams.test_list = input_image_file
    hyperparams.ckpt_path = ckpt_path
    hyperparams.image_groups = image_groups
    hyperparams.in_channels = in_channels

    # ---------------------
    # RUN TRAINING
    # ---------------------
    return fmc_process_image(hyperparams, network)

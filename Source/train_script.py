#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:46:20 2020

@author: eschweiler
"""

from argparse import ArgumentParser

import numpy as np
import torch
import glob
import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):

    """
    Main training routine specific for this project
    :param hparams:
    """

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = network(hparams=hparams)
    os.makedirs(hparams.output_path, exist_ok=True)
    
    # Load pretrained weights if available
    if not hparams.pretrained is None:
        model.load_pretrained(hparams.pretrained)

    # Resume from checkpoint if available    
    resume_ckpt = None
    if hparams.resume:
        checkpoints = glob.glob(os.path.join(hparams.output_path,'*.ckpt'))
        checkpoints.sort(key=os.path.getmtime)
        if len(checkpoints)>0:
            resume_ckpt = checkpoints[-1]
            print('Resuming from checkpoint: {0}'.format(resume_ckpt))
            
    # Set the augmentations if available
    model.set_augmentations(hparams.augmentations)
        
    # Save a few samples for sanity checks
    if hparams.dataloader_sanity_check:
        print('Saving 20 data samples for sanity checks...')
        model.train_dataloader().dataset.test(os.path.join(hparams.output_path, 'samples'), num_files=20)

    print('use_fmc_loss is set to: %i' % hparams.use_fmc_loss)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.output_path,
        filename=hparams.pipeline+'-{epoch:03d}-{step}',
        save_top_k=1,
        monitor='step',
        mode='max',
        verbose=True,
        every_n_epochs=1
    )
    
    logger = TensorBoardLogger(
        save_dir=hparams.log_path,
        name='lightning_logs_'+hparams.pipeline.lower()
    )
    
    trainer = Trainer(
        logger=logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        devices=hparams.devices,
        min_epochs=hparams.epochs,
        max_epochs=hparams.epochs,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model, ckpt_path=resume_ckpt)



if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
        '--output_path',
        type=str,
        default=r'/work/scratch/stegmaier/Projects/2025/FuseMyCellsISBI_ImageFusion/Processing/',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--log_path',
        type=str,
        default=r'/work/scratch/stegmaier/Projects/2025/FuseMyCellsISBI_ImageFusion/Logs/',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--devices',
        type=int,
        default=1,
        help='number of devices / GPUs to use'
    )
    
    parent_parser.add_argument(
        '--no_resume',
        dest='resume',
        action='store_false',
        default=True,
        help='Do not resume training from latest checkpoint'
    )

    parent_parser.add_argument(
        '--no_dataloader_sanity_check',
        dest='dataloader_sanity_check',
        action='store_false',
        default=True,
        help='Do not save sanity check data upon first run'
    )
    
    parent_parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        nargs='+',
        help='path to pretrained model weights'
    )
    
    parent_parser.add_argument(
        '--augmentations',
        type=str,
        default=None,#r'/work/scratch/eschweiler/augmentation_settings/augments_2d_default_05.json',
        help='path to augmentation dict file'
    )
    
    parent_parser.add_argument(
        '--epochs',
        type=int,
        default=5000,
        help='number of epochs'
    )
    
    parent_parser.add_argument(
        '--pipeline',
        type=str,
        default='ImageFusionUNet3D',
        help='which pipeline to load (MulticlassSeg3D | Cellpose3D | Cellpose3D_sampling | Cellpose3D_samplingAug | SynthGAN3D | SynthGAN3D_noise | SynthGAN3D_sampling | Autoencoder3D | Autoencoder3D_extract | Autoencoder3D_variational | Autoencoder3D_adversarial | HarmonicNet | GABi3D | Mapping2D | Mapping3D | DiffusionGAN3D | DiffusionModel3D | DiffusionModel2D)'
    )
    
    parent_args = parent_parser.parse_known_args()[0]
    
    # load the desired network architecture
    if parent_args.pipeline.lower() == 'multiclassseg3d':        
        from models.MulticlassSeg3D import MulticlassSeg3D as network
    elif parent_args.pipeline.lower() == 'cellpose3d':        
        from models.Cellpose3D import Cellpose3D as network
    elif parent_args.pipeline.lower() == 'cellpose3d_sampling':        
        from models.Cellpose3D_sampling import Cellpose3D_sampling as network
    elif parent_args.pipeline.lower() == 'cellpose3d_samplingaug':        
        from models.Cellpose3D_samplingAug import Cellpose3D_samplingAug as network
    elif parent_args.pipeline.lower() == 'synthgan3d':
        from models.SynthGAN3D import SynthGAN3D as network
    elif parent_args.pipeline.lower() == 'synthgan3d_noise':
        from models.SynthGAN3D_noise import SynthGAN3D_noise as network
    elif parent_args.pipeline.lower() == 'synthgan3d_sampling':
        from models.SynthGAN3D_sampling import SynthGAN3D_sampling as network
    elif parent_args.pipeline.lower() == 'synthgan2d':
        from models.SynthGAN2D import SynthGAN2D as network
    elif parent_args.pipeline.lower() == 'synthgan2d_noise':
        from models.SynthGAN2D_noise import SynthGAN2D_noise as network
    elif parent_args.pipeline.lower() == 'diffusiongan3d':
        from models.DiffusionGAN3D import DiffusionGAN3D as network
    elif parent_args.pipeline.lower() == 'diffusionmodel3d':
        from models.DiffusionModel3D import DiffusionModel3D as network
    elif parent_args.pipeline.lower() == 'diffusionmodel2d':
        from models.DiffusionModel2D import DiffusionModel2D as network
    elif parent_args.pipeline.lower() == 'autoencoder3d':
        from models.Autoencoder3D import Autoencoder3D as network
    elif parent_args.pipeline.lower() == 'autoencoder3d_extract':
        from models.Autoencoder3D_extract import Autoencoder3D_extract as network
    elif parent_args.pipeline.lower() == 'autoencoder3d_variational':
        from models.Autoencoder3D_variational import Autoencoder3D_variational as network
    elif parent_args.pipeline.lower() == 'autoencoder3d_adversarial':
        from models.Autoencoder3D_adversarial import Autoencoder3D_adversarial as network
    elif parent_args.pipeline.lower() == 'harmonicnet':
        from models.HarmonicNet import HarmonicNet as network
    elif parent_args.pipeline.lower() == 'gabi3d':
        from models.GABi3D import GABi3D as network
    elif parent_args.pipeline.lower() == 'mapping2d':
        from models.Mapping2D import Mapping2D as network
    elif parent_args.pipeline.lower() == 'imagefusionunet3d':
        from models.ImageFusionUNet3D import ImageFusionUNet3D as network
    elif parent_args.pipeline.lower() == 'mapping3d':
        from models.Mapping3D import Mapping3D as network
    else:
        raise ValueError('Pipeline {0} unknown.'.format(parent_args.pipeline))
    
    # each LightningModule defines arguments relevant to it
    parser = network.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)

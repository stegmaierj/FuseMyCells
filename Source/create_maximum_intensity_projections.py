#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31

@author: stegmaier
"""

from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import isfile, join, isdir
from utils.fmc_utils import create_maximum_intensity_projections
from skimage import io, filters


def main(hparams):
    
    input_file = hparams.input_file
    output_path = hparams.output_path
    output_path_xy = output_path + 'XY/'
    output_path_xz = output_path + 'XZ/'
    output_path_yz = output_path + 'YZ/'

    if not isdir(output_path):
        mkdir(output_path)

    if not isdir(output_path_xy):
        mkdir(output_path_xy)

    if not isdir(output_path_xz):
        mkdir(output_path_xz)

    if not isdir(output_path_yz):
        mkdir(output_path_yz)

    input_image = io.imread(input_file)
    projection_x, projection_y, projection_z = create_maximum_intensity_projections(input_image)

    io.imsave(output_path_xy + input_file.replace('.tif', '_MaxProjXY.tif'), projection_x)
    io.imsave(output_path_xz + input_file.replace('.tif', '_MaxProjXZ.tif'), projection_y)
    io.imsave(output_path_yz + input_file.replace('.tif', '_MaxProjYZ.tif'), projection_z)

    print(input_file)
    test = 1
   

if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '--input_file',
        type=str,
        default=r'/Users/jstegmaier/Downloads/Study3/Study1/image_91_nucleus_angle.tif',
        help='A single input file'
    )
    
    parent_parser.add_argument(
        '--output_path',
        type=str,
        default=r'path/to/output/images',
        help='Output directory'
    )    
        
    hyperparams = parent_parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)

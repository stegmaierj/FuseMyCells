#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31

@author: stegmaier
"""

from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import isfile, join, isdir
from fmc_utils import create_maximum_intensity_projections
from skimage import io, filters


def main(hparams):
    
    input_folder = hparams.input_folder
    output_folder = hparams.output_folder

    if not isdir(output_folder):
        mkdir(output_folder)

    input_files = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]

    for input_image_file in input_files:

        current_file = input_folder + input_image_file

        input_image = io.imread(current_file)
        projection_x, projection_y, projection_z = create_maximum_intensity_projections(input_image)

        io.imsave(output_folder + input_image_file.replace('.tif', '_MaxProjXY.tif'), projection_x)
        io.imsave(output_folder + input_image_file.replace('.tif', '_MaxProjXZ.tif'), projection_y)
        io.imsave(output_folder + input_image_file.replace('.tif', '_MaxProjYZ.tif'), projection_z)

        test = 1
   

if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '--input_folder',
        type=str,
        default=r'/Users/jstegmaier/Downloads/Study3/',
        help='Directory of raw images'
    )

    parent_parser.add_argument(
        '--input_file',
        type=str,
        default=r'/Users/jstegmaier/Downloads/Study3/Study1/image_91_nucleus_angle.tif',
        help='A single input file'
    )
    
    parent_parser.add_argument(
        '--output_folder',
        type=str,
        default=r'path/to/output/images',
        help='Output directory'
    )    
        
    hyperparams = parent_parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)

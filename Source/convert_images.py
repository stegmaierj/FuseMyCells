import os
import h5py
import glob
import numpy as np

from skimage import io, filters
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import binary_opening, binary_erosion

from scipy.ndimage import zoom, distance_transform_edt

import matplotlib.pyplot as plt

from os import listdir, mkdir
from os.path import isfile, join, isdir

from PIL import Image
from PIL.TiffTags import TAGS

from utils.fmc_utils import get_fmc_metadata, compute_convex_image, get_fmc_gradient_info, get_fmc_light_direction
from utils.h5_converter import prepare_image_fmc
from argparse import ArgumentParser

def main(hparams):
    
    input_file = hparams.input_file
    output_path = hparams.output_path

    if not isdir(output_path):
        mkdir(output_path)

    if "_fused" not in input_file:
        prepare_image_fmc(input_file, output_path=output_path, identifier='*.tif', descriptor='', normalize=[1,99],\
                       get_surfacedistance=True, get_lightmap=True, use_fmc_percentile_normalization=True, overwrite=False)
    else:
        prepare_image_fmc(input_file, output_path=output_path, identifier='*.tif', descriptor='', normalize=[1,99],\
                       get_surfacedistance=False, get_lightmap=False, use_fmc_percentile_normalization=True, overwrite=False)

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

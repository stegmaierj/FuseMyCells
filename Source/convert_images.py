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
import tifffile

from utils.fmc_utils import get_fmc_metadata, compute_convex_image, get_fmc_gradient_info, get_fmc_light_direction
from utils.h5_converter import prepare_image_fmc
from argparse import ArgumentParser

def standardize_metadata(metadata : dict):
    key_map = {
        "spacing": ["spacing"],
        "PhysicalSizeX": ["PhysicalSizeX", "physicalsizex", "physical_size_x"],
        "PhysicalSizeY": ["PhysicalSizeY", "physicalsizey", "physical_size_y"],
        "PhysicalSizeZ": ["PhysicalSizeZ", "physicalsizez", "physical_size_z"],
        "unit": ["unit"],
        "axes": ["axes"],
        "channel": ["channel"],
        "shape": ["shape"],
        "study": ["study"],
    }

    # Normalize metadata by looking up possible keys
    standardized_metadata = {}
    for standard_key, possible_keys in key_map.items():
        for key in possible_keys:
            if key in metadata:
                standardized_metadata[standard_key] = metadata[key]
                break  # Stop once we find the first available key

    return standardized_metadata



def read_image(location): # WARNING IMAGE DATA EN ZYX
    import tifffile
    # Read the TIFF file and get the image and metadata
    with tifffile.TiffFile(location) as tif:

        image_data = tif.asarray() # Extract image array data

        if tif.shaped_metadata is not None:
            shp_metadata = tif.shaped_metadata[0]
            metadata = standardize_metadata(shp_metadata)

            return image_data, metadata
        else:
            if tif.imagej_metadata is not None:
                shape = list(image_data.shape)
                imgj_metadata = tif.imagej_metadata
                imgj_metadata['shape'] = shape
                metadata = standardize_metadata(imgj_metadata)

                return image_data, metadata

            else:
                metadata = tif.pages[0].tags['ImageDescription'].value
                print(f"error loading metadata: {metadata}, type of object : {type(metadata)}")

"""
def read_image(location): # WARNING IMAGE DATA EN ZYX
    # Read the TIFF file and get the image and metadata
    with tifffile.TiffFile(location) as tif:
        image_data = tif.asarray() # Extract image data
        metadata = tif.shaped_metadata  # Get the existing metadata in a DICT

        if metadata == None:
            metadata = get_fmc_metadata(location)
            metadata['PhysicalSizeX'] = float(metadata['physicalsizex'])
            metadata['PhysicalSizeY'] = float(metadata['physicalsizey'])
            metadata['PhysicalSizeZ'] = float(metadata['physicalsizez'])
            return image_data, metadata
        else:
            return image_data, metadata[0]
"""


def save_image(*, location, array, metadata):

    PhysicalSizeX = metadata['PhysicalSizeX']
    PhysicalSizeY = metadata['PhysicalSizeY']
    tifffile.imwrite(
        location,
        array,
        bigtiff=True, #Keep it for 3D images
        resolution=(1. / PhysicalSizeX, 1. / PhysicalSizeY),
        metadata=metadata,
        tile=(128, 128),
        )


def main(hparams):
    
    input_file = hparams.input_file
    output_path = hparams.output_path

    if not isdir(output_path):
        mkdir(output_path)

    image_input, metadata = read_image(input_file)

    if "_fused" not in input_file:
        prepare_image_fmc(input_file, image_input, metadata, output_path=output_path, identifier='*.tif', descriptor='', normalize=[1,99],\
                       get_surfacedistance=True, get_lightmap=True, use_fmc_percentile_normalization=True, overwrite=False)
    else:
        prepare_image_fmc(input_file, image_input, metadata, output_path=output_path, identifier='*.tif', descriptor='', normalize=[1,99],\
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
"""

The following is a simple example algorithm inference
where the images predicted are only full black images.

"""

print(" START IMPORT ")

import os
from os import listdir, mkdir
from os.path import basename, isdir, join
from pathlib import Path
import csv

import numpy as np
import tifffile

# from tools import percentile_normalization
from resources.utils.fmc_utils import get_fmc_metadata, fmc_guess_dataset
from resources.utils.h5_converter import prepare_image_fmc
from resources.apply_script_docker import fmc_entry_point

print(" END IMPORT ")

localDebugPrefix = "/work/scratch/stegmaier/Projects/2025/FuseMyCellsISBI_ImageFusion/Source/Docker/"
#localDebugPrefix = ""

INPUT_PATH = Path(localDebugPrefix + "/input/images/fluorescence-lightsheet-3D-microscopy")
print(f" INPUT_PATH IS   " + str(INPUT_PATH))
os.system("ls -l " + str(INPUT_PATH))

OUTPUT_PATH = Path(localDebugPrefix + "/output")
if not isdir(join(OUTPUT_PATH,"images")):
    mkdir(join(OUTPUT_PATH, "images"))

OUTPUT_PATH = Path(localDebugPrefix + "/output/images")
if not isdir(join(OUTPUT_PATH, "fused-fluorescence-lightsheet-3D-microscopy")):
    mkdir(join(OUTPUT_PATH, "fused-fluorescence-lightsheet-3D-microscopy"))

OUTPUT_PATH = Path(localDebugPrefix + "/output/images/fused-fluorescence-lightsheet-3D-microscopy")
print(" OUTPUT IS  " + str(OUTPUT_PATH))

RESOURCE_PATH = Path("resources") #WEIGHTS NEED TO BE PUT IN RESOURCE_PATH
print(" RESOURCE_PATH IS   " + str(RESOURCE_PATH))
os.system("ls -l " + str(RESOURCE_PATH))

TMP_PATH = Path(localDebugPrefix + "/tmp/")
print(" TMP PATH IS  " + str(TMP_PATH))

def run():

    print(f" LIST IMAGES IN  {INPUT_PATH} ")

    for input_file_name in listdir(INPUT_PATH):
        if input_file_name.endswith("tiff") or input_file_name.endswith("tif"):

            # load input image and get meta data
            print(" --> Predict " + input_file_name)
            image_input, metadata = read_image(join(INPUT_PATH,input_file_name))

            skip_processing = False

            if not skip_processing:
                # determine which model to use
                study_number = -1
                model_suffix = "Nucleus" if metadata['channel'] == 'nucleus' else "Membrane"
                if 'study' in metadata:
                    study_number = int(metadata['study'])
                else:
                    study_number, model_suffix = fmc_guess_dataset(metadata)

                # find the checkpoint to be used
                print("File %s will be processed with model for study %i and %s" % (input_file_name, study_number, model_suffix))     
                ckpt_path = "%s%s/weights/Study%i_%s.ckpt" % (localDebugPrefix, RESOURCE_PATH, study_number, model_suffix)
                print("Using checkpoint path %s" % (ckpt_path))

                image_groups = ["data/raw_image", "data/surface_distance"]
                in_channels = 2

                #if study_number == 2:
                #    image_groups = ["data/raw_image", "data/light_map"]
                #elif study_number == 3 and metadata['channel'] == 'membrane':
                #    in_channels = 1
                #    image_groups = ["data/raw_image"]

                input_path_tif = join(INPUT_PATH,input_file_name)
                input_name_h5 = join(str(TMP_PATH), input_file_name.replace(".tiff", ".h5").replace(".tif", ".h5"))
                input_name_csv = join(str(TMP_PATH), input_file_name.replace(".tiff", ".csv").replace(".tif", ".csv"))

                print("Input path Tiff is set to %s" % (input_path_tif))
                print("Input path H5 is set to %s" % (input_name_h5))
                print("Input path CSV is set to %s" % (input_name_csv))

                with open(input_name_csv, 'w', newline='') as fh:
                    writer = csv.writer(fh, delimiter=';')
                    writer.writerow([input_name_h5, input_name_h5])

                print("Trying to perform preprocessing ... ")
                prepare_image_fmc(input_path_tif, image_input, metadata, output_path=join(str(TMP_PATH),''), identifier='*.tif', descriptor='', normalize=[1,99],\
                        get_surfacedistance=True, get_lightmap=True, use_fmc_percentile_normalization=True, overwrite=False)

                print("Successfully finished preprocessing ... ")

                # Prediction
                print("Trying to perform the prediction ... ")
                image_predict = fmc_entry_point(input_name_csv, ckpt_path, in_channels, image_groups)
                print("Successfully finished prediction ... ")

                print("Trying to remove temporary files ... ")
                os.remove(input_name_csv)
                #os.remove(input_name_h5)
                print("Successfully removed temporary files ...")
                
            else:
                image_predict = image_input

            print("Trying to save result image ... ")
            save_image(location = join(OUTPUT_PATH, basename(input_file_name)),
                       array = image_predict,
                       metadata = metadata
                       )
            print("Successfully saved result image ... ")           


    print(" --> LIST OUTPUT IMAGES IN "+str(OUTPUT_PATH))

    for output_images in listdir(OUTPUT_PATH):
        print(" --> FOUND "+str(output_images))
    return 0


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


if __name__ == "__main__":
    raise SystemExit(run())

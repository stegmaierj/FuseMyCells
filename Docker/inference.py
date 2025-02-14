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
from resources.utils.fmc_utils import get_fmc_metadata
from resources.utils.h5_converter import prepare_image_fmc
from resources.apply_script_docker import fmc_entry_point

print(" END IMPORT ")

#localDebugPrefix = "/work/scratch/stegmaier/Projects/2025/FuseMyCellsISBI_ImageFusion/Source/Docker"
localDebugPrefix = ""

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
    print(" LOAD NETWORK ")
    # model = mynetwork
    # weight_file = join(RESOURCE_PATH, your_model.keras")
    # model.load_weights(weight_file)

    print(f" LIST IMAGES IN  {INPUT_PATH} ")

    for input_file_name in listdir(INPUT_PATH):
        if input_file_name.endswith("tiff") or input_file_name.endswith("tif"):
            print(" --> Predict " + input_file_name)
            image_input, metadata = read_image(join(INPUT_PATH,input_file_name))

            image_groups = ["data/raw_image", "data/surface_distance", "data/light_map"]
            in_channels = 3

            study_number = int(metadata['study'])
            model_suffix = "Nucleus" if "nucleus" in input_file_name else "Membrane"

            ckpt_path = "%s%s/weights/Study%i_%s.ckpt" % (localDebugPrefix, RESOURCE_PATH, study_number, model_suffix)

            input_path_tif = join(INPUT_PATH,input_file_name)
            input_name_h5 = join(str(TMP_PATH), input_file_name.replace(".tif", ".h5"))
            input_name_csv = join(str(TMP_PATH), input_file_name.replace(".tif", ".csv"))

            with open(input_name_csv, 'w', newline='') as fh:
                writer = csv.writer(fh, delimiter=';')
                writer.writerow([input_name_h5, input_name_h5])

            prepare_image_fmc(input_path_tif, output_path=join(str(TMP_PATH),''), identifier='*.tif', descriptor='', normalize=[1,99],\
                       get_surfacedistance=True, get_lightmap=True, use_fmc_percentile_normalization=True, overwrite=False)

            # Prediction
            image_predict = fmc_entry_point(input_name_csv, ckpt_path, in_channels, image_groups)

            save_image(location = join(OUTPUT_PATH, basename(input_file_name)),
                       array = image_predict,
                       metadata = metadata
                       )
            
            os.remove(input_name_csv)
            #os.remove(input_name_h5)

    print(" --> LIST OUTPUT IMAGES IN "+str(OUTPUT_PATH))

    for output_images in listdir(OUTPUT_PATH):
        print(" --> FOUND "+str(output_images))
    return 0



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

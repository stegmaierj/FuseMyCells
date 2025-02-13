"""

The following is a simple example algorithm inference
where the images predicted are only full black images.

"""

print(" START IMPORT ")

import os
from os import listdir, mkdir
from os.path import basename, isdir, join
from pathlib import Path

import numpy as np
import tifffile

# from tools import percentile_normalization


print(" END IMPORT ")

INPUT_PATH = Path("/input/images/fluorescence-lightsheet-3D-microscopy")
print(f" INPUT_PATH IS   " + str(INPUT_PATH))
os.system("ls -l " + str(INPUT_PATH))

OUTPUT_PATH = Path("/output")
if not isdir(join(OUTPUT_PATH,"images")):
    mkdir(join(OUTPUT_PATH, "images"))

OUTPUT_PATH = Path("/output/images")
if not isdir(join(OUTPUT_PATH, "fused-fluorescence-lightsheet-3D-microscopy")):
    mkdir(join(OUTPUT_PATH, "fused-fluorescence-lightsheet-3D-microscopy"))

OUTPUT_PATH = Path("/output/images/fused-fluorescence-lightsheet-3D-microscopy")
print(" OUTPUT IS  " + str(OUTPUT_PATH))

RESOURCE_PATH = Path("resources") #WEIGHTS NEED TO BE PUT IN RESOURCE_PATH
print(" RESOURCE_PATH IS   " + str(RESOURCE_PATH))
os.system("ls -l " + str(RESOURCE_PATH))


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

            # Prediction
            image_predict = np.zeros(image_input.shape, dtype = np.uint16) # model.predict()

            save_image(location = join(OUTPUT_PATH, basename(input_file_name)),
                       array = image_predict,
                       metadata = metadata
                       )

    print(" --> LIST OUTPUT IMAGES IN "+str(OUTPUT_PATH))

    for output_images in listdir(OUTPUT_PATH):
        print(" --> FOUND "+str(output_images))
    return 0



def read_image(location): # WARNING IMAGE DATA EN ZYX
    # Read the TIFF file and get the image and metadata
    with tifffile.TiffFile(location) as tif:
        image_data = tif.asarray() # Extract image data
        metadata = tif.shaped_metadata  # Get the existing metadata in a DICT
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

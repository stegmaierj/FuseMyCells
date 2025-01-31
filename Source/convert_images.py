import os
import h5py
import glob
import numpy as np

from skimage import io, filters
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import binary_opening, binary_erosion

from scipy.ndimage import zoom, distance_transform_edt

import matplotlib.pyplot as plt

from PIL import Image
from PIL.TiffTags import TAGS

from fmc_utils import get_fmc_metadata, compute_convex_image, get_fmc_gradient_info, get_fmc_light_direction



selected_series = 1

#input_image_file = '/Users/jstegmaier/Downloads/Study3/Study1/image_100_membrane_angle.tif'
#input_image_file = '/Users/jstegmaier/Downloads/Study3/Study2/image_120_nucleus_angle.tif'
#input_image_file = '/Users/jstegmaier/Downloads/Study3/Study3/image_192_membrane_angle.tif'
input_image_file = '/Users/jstegmaier/Downloads/Study3/Study4/image_305_nucleus_fused.tif'
#input_image_file = '/Users/jstegmaier/Downloads/Study3/Study5/image_394_nucleus_angle.tif'

output_image_file_edt = input_image_file.replace('.tif', '_edt.tif')
output_image_file_lmi = input_image_file.replace('.tif', '_lmi.tif')


meta_data = get_fmc_metadata(input_image_file)
    

current_study = int(meta_data['study'])
gradient_z, gradient_y, gradient_x = get_fmc_gradient_info(current_study)
    

input_image = io.imread(input_image_file)


slopes = get_fmc_light_direction(input_image, debug_figures=False)

original_size = input_image.shape


image_spacing = [float(meta_data['physical_size_z']), float(meta_data['physical_size_y']), float(meta_data['physical_size_x'])]

input_image = zoom(input_image, (image_spacing[0], image_spacing[1], image_spacing[2]))

## create the light distance image
light_map_image = np.ones_like(input_image)

#gradient_z = False
#gradient_y = False
#gradient_x = True

#slopes[0] = -1
#slopes[1] = -1
#slopes[2] = -1

if gradient_z and slopes[0] > 0:
    light_map_image[0, ...] = 0
elif gradient_z and slopes[0] < 0:
    light_map_image[-1, ...] = 0

if gradient_y and slopes[1] > 0:
    light_map_image[:, 0, :] = 0
elif gradient_y and slopes[1] < 0:
    light_map_image[:, -1, :] = 0

if gradient_x and slopes[2] > 0:
    light_map_image[:, :, 0] = 0
elif gradient_x and slopes[2] < 0:
    light_map_image[:, :, -1] = 0

light_map_image = distance_transform_edt(light_map_image)
io.imsave(output_image_file_lmi, light_map_image.astype(np.uint16))

downsampled_size = input_image.shape
upsampling_factors = np.array(original_size) / np.array(downsampled_size)

convex_image = compute_convex_image(input_image, image_spacing)

edt_image = distance_transform_edt(convex_image)

edt_image = zoom(edt_image.astype(np.uint16), (upsampling_factors[0], upsampling_factors[1], upsampling_factors[2]))

io.imsave(output_image_file_edt, edt_image)





#find_light_direction(input_image)
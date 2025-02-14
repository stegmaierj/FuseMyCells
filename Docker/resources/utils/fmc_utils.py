import os
import h5py
import glob
import numpy as np

from skimage import io, filters

from skimage.measure import label, regionprops, regionprops_table

from skimage.morphology import binary_opening, binary_erosion

from scipy.ndimage import zoom

import matplotlib.pyplot as plt

from PIL import Image
from PIL.TiffTags import TAGS


def get_fmc_metadata(input_image_file):
    img = Image.open(input_image_file)

    meta_data_string = img.tag[270][0].split('\n')

    meta_data = {}
    for tag in meta_data_string:
        split_string = tag.split('=')

        if len(split_string[0]) > 0:
            meta_data[split_string[0]] = split_string[1]

    return meta_data

def get_fmc_gradient_info(study_number):
    x_gradient = False
    y_gradient = False
    z_gradient = False
    if study_number==1:
        z_gradient = True
    elif study_number==2:
        z_gradient = True
    elif study_number==3:
        x_gradient = True
        y_gradient = True
        z_gradient = True
    elif study_number==4:
        x_gradient = True
    elif study_number==5:
        x_gradient = True

    return z_gradient, y_gradient, x_gradient

def get_fmc_light_direction(input_image, debug_figures=False):

    # get the intensity statistics
    #min_intensity = np.min(input_image)
    #max_intensity = np.max(input_image)
    #print(min_intensity)
    #print(max_intensity)

    # compute the maximum projections
    projection_xy = np.max(input_image, axis=0)
    projection_xz = np.max(input_image, axis=1)

    # compute the line projections
    projection_x = np.mean(projection_xy, axis=0)
    projection_y = np.mean(projection_xy, axis=1)
    projection_z = np.mean(projection_xz, axis=1)

    otsu_x = filters.threshold_otsu(projection_x)
    otsu_y = filters.threshold_otsu(projection_y)
    otsu_z = filters.threshold_otsu(projection_z)

    projection_x = projection_x[projection_x > otsu_x]
    projection_y = projection_y[projection_y > otsu_y]
    projection_z = projection_z[projection_z > otsu_z]

    range_x = np.linspace(np.min(projection_x), np.max(projection_x), len(projection_x))
    range_y = np.linspace(np.min(projection_y), np.max(projection_y), len(projection_y))
    range_z = np.linspace(np.min(projection_z), np.max(projection_z), len(projection_z))

    slope_x = np.polyfit(range_x, projection_x, 1)
    slope_y = np.polyfit(range_y, projection_y, 1)
    slope_z = np.polyfit(range_z, projection_z, 1)

    if debug_figures:
        line_x = range_x * slope_x[0] + slope_x[1]
        line_y = range_y * slope_y[0] + slope_y[1]
        line_z = range_z * slope_z[0] + slope_z[1]

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        ax1.plot(projection_z)
        ax1.plot(line_z)
        ax1.set_xlabel("Z-Profile")
        ax1.set_ylabel("Z-Profile")
        ax1.set_title("%f" % slope_z[0])

        ax2.plot(projection_y)
        ax2.plot(line_y)
        ax2.set_xlabel("Y-Profile")
        ax2.set_ylabel("Intensity")
        ax2.set_title("%f" % slope_y[0])

        ax3.plot(projection_x)
        ax3.plot(line_x)
        ax3.set_xlabel("X-Profile")
        ax3.set_ylabel("Intensity")
        ax3.set_title("%f" % slope_x[0])

        plt.show()


    test = 1

    slopes = -np.sign([slope_z[0], slope_y[0], slope_x[0]])
    return slopes

def compute_convex_image(input_image, image_spacing, gaussian_sigma=1.0):

    # smooth image to remove some noise before thresholding    
    smoothed_image = filters.gaussian(input_image, (gaussian_sigma*image_spacing[0], gaussian_sigma*image_spacing[1], gaussian_sigma*image_spacing[2]))

    binary_image = np.zeros_like(smoothed_image)
    binary_image[smoothed_image > np.mean(smoothed_image[:])] = 1

    binary_image = binary_opening(binary_image)

    label_image = label(binary_image)

    region_props = regionprops(label_image)

    max_label = None
    max_volume = 0
    for region in region_props:
        if region.area > max_volume:
            max_volume = region.area
            max_label = region.label


    binary_image[:] = 0
    binary_image[label_image == max_label] = 1

    binary_image = binary_image.astype(np.uint8) - binary_erosion(binary_image).astype(np.uint8)

    region_props = regionprops(binary_image)
    bounding_box = region_props[0].bbox

    convex_image = np.zeros_like(binary_image)
    convex_image[bounding_box[0]:bounding_box[3], bounding_box[1]:bounding_box[4], bounding_box[2]:bounding_box[5]] = region_props[0].convex_image
    
    return convex_image

"""
def compute_convex_image_slow(input_image, image_spacing, gaussian_sigma=1.0, zoom_factor=None):

    input_image_size = input_image.shape

    input_image_small = input_image
    upsampling_factors = np.array([1.0,1.0,1.0])
    if not zoom_factor == None:
        input_image_small = zoom(input_image, (zoom_factor, zoom_factor, zoom_factor))
        input_image_small_size = input_image_small.shape
        upsampling_factors = np.array(input_image_size) / np.array(input_image_small_size)
      
    smoothed_image = filters.gaussian(input_image_small, (gaussian_sigma*image_spacing[0], gaussian_sigma*image_spacing[1], gaussian_sigma*image_spacing[2]))

    binary_image = np.zeros_like(smoothed_image)
    binary_image[smoothed_image > np.mean(smoothed_image[:])] = 1

    binary_image = binary_opening(binary_image)


    label_image = label(binary_image)

    region_props = regionprops(label_image)

    max_label = None
    max_volume = 0
    for region in region_props:
        if region.area > max_volume:
            max_volume = region.area
            max_label = region.label


    binary_image[:] = 0
    binary_image[label_image == max_label] = 1

    binary_image = binary_image.astype(np.uint8) - binary_erosion(binary_image).astype(np.uint8)

    region_props = regionprops(binary_image)
    bounding_box = region_props[0].bbox

    convex_image = np.zeros_like(binary_image)
    convex_image[bounding_box[0]:bounding_box[3], bounding_box[1]:bounding_box[4], bounding_box[2]:bounding_box[5]] = region_props[0].convex_image

    if not zoom_factor == None:
        convex_image = zoom(convex_image, (upsampling_factors[0], upsampling_factors[1], upsampling_factors[2]))
    
    return convex_image
"""

def create_maximum_intensity_projections(input_image):
    
    # compute the maximum projections
    projection_xy = np.max(input_image, axis=0)
    projection_xz = np.max(input_image, axis=1)
    projection_yz = np.max(input_image, axis=2)

    return projection_xy, projection_xz, projection_yz


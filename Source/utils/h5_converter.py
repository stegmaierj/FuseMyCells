# -*- coding: utf-8 -*-


import os
import h5py
import glob
import numpy as np

from os import listdir, mkdir
from os.path import isfile, join, isdir

from skimage import io, morphology, measure, filters
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom, generic_filter
from scipy.ndimage.morphology import binary_fill_holes
from scipy.spatial import ConvexHull, Delaunay

from utils.utils import print_timestamp
from utils.fmc_utils import get_fmc_gradient_info, get_fmc_light_direction, get_fmc_metadata, compute_convex_image

import SimpleITK as sitk



def h5_writer(data_list, save_path, group_root='data', group_names=['image']):
    
    save_path = os.path.abspath(save_path)
    
    assert(len(data_list)==len(group_names)), 'Each data matrix needs a group name'
    
    with h5py.File(save_path, 'w') as f_handle:
        grp = f_handle.create_group(group_root)
        for data, group_name in zip(data_list, group_names):
            grp.create_dataset(group_name, data=data, chunks=True, compression='gzip')
            
            
            
def h5_reader(file, source_group='data/image'):
    
    with h5py.File(file, 'r') as file_handle:
       data = file_handle[source_group][:]
                            
    return data




def h52tif(file_dir='', identifier='*', group_names=['data/image']):
    
    # Get all files within the given directory
    filelist = glob.glob(os.path.join(file_dir, identifier+'.h5'))
    
    # Create saving folders
    for group in group_names:
        os.makedirs(os.path.join(file_dir, ''.join(s for s in group if s.isalnum())), exist_ok=True)
    
    # Save each desired group
    for num_file,file in enumerate(filelist):
        print_timestamp('Processing file {0}/{1}', (num_file+1, len(filelist)))
        with h5py.File(file, 'r') as file_handle:
            for group in group_names:
                data = file_handle[group][:]
                io.imsave(os.path.join(file_dir, ''.join(s for s in group if s.isalnum()), os.path.split(file)[-1][:-2]+'tif'), data)
                


def replace_h5_group(source_list, target_list, source_group='data/image', target_group=None):
    
    assert len(target_list)==len(source_list), 'There needs to be one target ({0}) for each source ({1})!'.format(len(target_list), len(source_list))
    if target_group is None: target_group=source_group

    for num_pair, pair in enumerate(zip(source_list, target_list)):
        print_timestamp('Processing file {0}/{1}...', [num_pair+1, len(target_list)])
        
        # Load the source mask
        with h5py.File(pair[0], 'r') as source_handle:
            source_data = source_handle[source_group][...]
            
        # Save the data to the target file
        with h5py.File(pair[1], 'r+') as target_handle:
            target_data = target_handle[target_group]
            target_data[...] = source_data




def add_group(file, data, target_group='data/image'):
    
    with h5py.File(file, 'a') as file_handle:
        file_handle.create_dataset(target_group, data=data, chunks=True, compression='gzip')
        
        
        
            
def add_h5_group(source_list, target_list, source_group='data/distance', target_group=None):
    
    assert len(target_list)==len(source_list), 'There needs to be one target ({0}) for each source ({1})!'.format(len(target_list), len(source_list))
    if target_group is None: target_group=source_group
    
    for num_pair, pair in enumerate(zip(source_list, target_list)):
        
        print_timestamp('Processing file {0}/{1}...', [num_pair+1, len(source_list)])
            
        # Get the data from the source file
        with h5py.File(pair[0], 'r') as source_handle:
            source_data = source_handle[source_group][...]
            
        # Save the data to the target file
        try:
            with h5py.File(pair[1], 'a') as target_handle:
                target_handle.create_dataset(target_group, data=source_data, chunks=True, compression='gzip')
        except:
            print_timestamp('Skipping file "{0}"...', [os.path.split(pair[1])[-1]])
                    



def add_tiff_group(source_list, target_list, target_group='data/newgroup'):
    
    assert len(target_list)==len(source_list), 'There needs to be one target ({0}) for each source ({1})!'.format(len(target_list), len(source_list))
    assert target_group is not None, 'There needs to be a target group name!'
    
    for num_pair, pair in enumerate(zip(source_list, target_list)):
        
        print_timestamp('Processing file {0}/{1}...', [num_pair+1, len(source_list)])
            
        # Get the data from the source file
        source_data = io.imread(pair[0])
            
        # Save the data to the target file
        with h5py.File(pair[1], 'a') as target_handle:
            target_handle.create_dataset(target_group, data=source_data-np.min(source_data), chunks=True, compression='gzip')



def remove_h5_group(file_list, source_group='data/nuclei'):
    
    for num_file, file in enumerate(file_list):
        
        print_timestamp('Processing file {0}/{1}...', [num_file+1, len(file_list)])
        
        with h5py.File(file, 'a') as file_handle:
            del file_handle[source_group]

            
            
def flood_fill_hull(image):    
    
    # Credits: https://stackoverflow.com/questions/46310603/how-to-compute-convex-hull-image-volume-in-3d-numpy-arrays
    points = np.transpose(np.where(image))
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    
    return out_img



def calculate_flows(instance_mask, bg_label=0):
    
    flow_x = np.zeros(instance_mask.shape, dtype=np.float32)
    flow_y = np.zeros(instance_mask.shape, dtype=np.float32)
    flow_z = np.zeros(instance_mask.shape, dtype=np.float32)
    regions = measure.regionprops(instance_mask)
    for props in regions:
        
        if props.label == bg_label:
            continue
        
        # get all coordinates within instance
        c = props.centroid
        coords = np.where(instance_mask==props.label)
        
        # calculate minimum extend in all spatial directions
        norm_x = np.maximum(1, np.minimum(np.abs(c[0]-props.bbox[0]),np.abs(c[0]-props.bbox[3]))/3)
        norm_y = np.maximum(1, np.minimum(np.abs(c[1]-props.bbox[1]),np.abs(c[1]-props.bbox[4]))/3)
        norm_z = np.maximum(1, np.minimum(np.abs(c[2]-props.bbox[2]),np.abs(c[2]-props.bbox[5]))/3)
        
        # calculate flows
        flow_x[coords] = np.tanh((coords[0]-c[0])/norm_x)
        flow_y[coords] = np.tanh((coords[1]-c[1])/norm_y)
        flow_z[coords] = np.tanh((coords[2]-c[2])/norm_z)
    
    return flow_x, flow_y, flow_z
        
        

def rescale_data(data, zoom_factor, order=0):
    
    if any([zf!=1 for zf in zoom_factor]):
        data_shape = data.shape
        data = zoom(data, zoom_factor, order=order)
        print_timestamp('Rescaled image from size {0} to {1}'.format(data_shape, data.shape))
        
    return data



def foreground_from_mip(img):
    
    mip = np.ones_like(img)
    
    for ndim in range(img.ndim):
        mip_tmp = np.max(img, axis=ndim)
        mip_tmp = mip_tmp>0.05
        mip_tmp = np.expand_dims(mip_tmp, ndim)
        mip_tmp = np.repeat(mip_tmp, img.shape[ndim], ndim)
        
        mip *= mip_tmp
        
    return mip
        

def fmc_percentile_normalization(image, pmin=2, pmax=99.8, axis=None):
    """
    Compute a percentile normalization for the given image.
    Parameters:
    - image (array): array (2D or 3D) of the image file.
    - pmin  (int or float): the minimal percentage for the percentiles to compute.
                            Values must be between 0 and 100 inclusive.
    - pmax  (int or float): the maximal percentage for the percentiles to compute.
                            Values must be between 0 and 100 inclusive.
    - axis : Axis or axes along which the percentiles are computed.
             The default (=None) is to compute it along a flattened version of the array.
    - dtype (dtype): type of the wanted percentiles (uint16 by default)

    Returns:
    Normalized image (np.ndarray): An array containing the normalized image.
    """

    if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100 ):
        raise ValueError("Invalid values for pmin and pmax")

    low_percentile = np.percentile(image, pmin, axis = axis, keepdims = True)
    high_percentile = np.percentile(image, pmax, axis = axis, keepdims = True)

    if low_percentile == high_percentile:
        print(f"Same min {low_percentile} and high {high_percentile}, image may be empty")
        return image

    return (image - low_percentile) / (high_percentile - low_percentile)

def resample_image_itk(input_image, image_spacing=[1.0, 1.0, 1.0], resize_factor=1.0):
   
    original_CT = sitk.GetImageFromArray(input_image)
    original_CT.SetSpacing(image_spacing)

    dimension = original_CT.GetDimension()
    reference_physical_size = np.zeros(original_CT.GetDimension())
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]
    
    reference_origin = original_CT.GetOrigin()
    reference_direction = original_CT.GetDirection()

    reference_size = [round(sz/resize_factor) for sz in original_CT.GetSize()] 
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(original_CT.GetDirection())

    transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)
  
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.CompositeTransform(transform)
    centered_transform.AddTransform(centering_transform)

    # sitk.Show(sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0))
    
    return sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0)


def prepare_image_fmc(input_path, output_path=None, identifier='*.tif', descriptor='', normalize=[0,100],\
                   get_surfacedistance=False, get_lightmap=False, use_fmc_percentile_normalization=False, overwrite=False):
    
    meta_data = get_fmc_metadata(input_path)

    current_study = 0
    if 'study' in meta_data.keys():
        current_study = int(meta_data['study'])
    else:
        current_study = 4
        print('Meta information for study not found for current file! Setting it hard-coded to 4!')

    image_spacing = [float(meta_data['physical_size_z']), float(meta_data['physical_size_y']), float(meta_data['physical_size_x'])]

    # save the data
    head, tail = os.path.split(input_path)
    save_name = ''
    if (output_path == None):
        save_name = input_path.replace('.tif', '.h5')
    else:
        if not isdir(output_path):
            mkdir(output_path)
        save_name = output_path + tail.replace('.tif', '.h5')

    if isfile(save_name) and not overwrite:
        return

    # load the image
    input_image = io.imread(input_path)
    input_image = input_image.astype(np.float32)

    #print_timestamp('Resampling Image with ITK')
    #small_input_image = resample_image_itk(input_image, image_spacing=image_spacing, resize_factor=0.5)
    #print_timestamp('Done')

    #test = 1

    #print_timestamp('Resampling Image with skimage')
    small_input_image = zoom(input_image, (image_spacing[0], image_spacing[1], image_spacing[2]))
    #print_timestamp('Done')

    original_size = input_image.shape
    downsampled_size = small_input_image.shape
    upsampling_factors = np.array(original_size) / np.array(downsampled_size)

    save_imgs = []
    if use_fmc_percentile_normalization:
        save_imgs = [fmc_percentile_normalization(input_image),] 
    else:
        save_imgs = [input_image,]

    # save raw image
    save_groups = ['raw_image',]
    
    if get_lightmap:
        
        print_timestamp('Extracting light map image...')

        ## create the light distance image
        light_map_image = np.ones_like(small_input_image)

        gradient_z, gradient_y, gradient_x = get_fmc_gradient_info(current_study)
        slopes = get_fmc_light_direction(input_image, debug_figures=False)

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
        light_map_image = zoom(light_map_image.astype(np.uint16), (upsampling_factors[0], upsampling_factors[1], upsampling_factors[2]))
        
        save_imgs.append(light_map_image.astype(np.uint16))
        save_groups.append('light_map')
        print_timestamp('Done extracting light map image...')
        
    if get_surfacedistance:
        
        print_timestamp('Extracting distance image...')

        zoom_factor = None
        if np.max(small_input_image.shape) > 500:
            zoom_factor = 0.5

        convex_image = compute_convex_image(small_input_image, image_spacing, zoom_factor=zoom_factor)
        edt_image = distance_transform_edt(convex_image)

        edt_image = zoom(edt_image.astype(np.uint16), (upsampling_factors[0], upsampling_factors[1], upsampling_factors[2]))

        
        save_imgs.append(edt_image.astype(np.uint16))
        save_groups.append('surface_distance')

        print_timestamp('Done extracting distance map image...')

    """
    if get_normalized_intensity:
        
        print_timestamp('Extracting normalized intensity image...')
        
        # normalize the image
        perc1, perc2 = np.percentile(input_image.astype(np.float32), list(normalize))
        input_image = input_image.astype(np.float32)
        input_image -= perc1
        input_image /= (perc2-perc1)
        input_image = np.clip(input_image, 0, 1)
        input_image = input_image.astype(np.float32)
        
        save_imgs.append(input_image.astype(np.float32))
        save_groups.append('normalized_intensity')
    """        

    h5_writer(save_imgs, save_name, group_root='data', group_names=save_groups)
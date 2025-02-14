# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:53:11 2021

@author: Nutzer
"""

import time
import os
import glob
from skimage import io




def print_timestamp(msg, args=None):
    
    print('[{0:02.0f}:{1:02.0f}:{2:02.0f}] '.format(time.localtime().tm_hour,time.localtime().tm_min,time.localtime().tm_sec) +\
          msg.format(*args if not args is None else ''))




def extract_videoframes(data_path='', identifier='*.tif', slice_num=0):
    
    # Create frame folder
    os.makedirs(os.path.join(data_path, 'frames'), exist_ok=True)
    
    # Get files
    files = glob.glob(os.path.join(data_path, identifier))
    
    # Extract frames
    for num_file,file in enumerate(files):        
        print_timestamp('Processing file {0}/{1}...', (num_file+1, len(files)))
        img = io.imread(file)[slice_num,...]
        io.imsave(os.path.join(data_path, 'frames', os.path.split(file)[-1]), img, check_contrast=False)
    

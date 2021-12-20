#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:54:37 2021

@author: rv43
"""

import logging
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s %(message)s')

#  PROCESSING NF GRAINS WITH MISORIENTATION
#==============================================================================
import os
import sys
import re
import pyinputplus as pyip
import numpy as np
import scipy as sp
#import matplotlib.pyplot as plt

import msnc_tools as msnc
import tomofunctions as tf

config = msnc.readConfigFile('config.txt')
detector = None
if config.get('detector_id'):
    detector = msnc.readDetectorConfig(config['detector_id'] + '.yml')
assert(detector)

#%%============================================================================
#% DETECTOR AND SCAN PARAMETERS
#==============================================================================
num_rows = detector['detector']['pixels']['rows']
logging.debug(f'num_rows = {num_rows}')
num_columns = detector['detector']['pixels']['columns']
logging.debug(f'num_columns = {num_columns}')
lens_magnification = config.get('lens_magnification', 1.)
pixel_size = detector['detector']['pixels']['size'][0]/lens_magnification
logging.debug(f'pixel_size = {pixel_size}')

start_angle = config.get('start_angle', 0.)
logging.debug(f'start_angle = {start_angle}')
end_angle = config.get('end_angle', 180.)
logging.debug(f'end_angle = {end_angle}')
if 'num_angles' in config:
    num_angles = config['num_angles']
else:
    num_angles = pyip.inputInt('Enter the number of angles (>0): ', greaterThan=0)
    config = msnc.addtoConfigFile('config.txt', 'Scan parameters', f'num_angles = {num_angles}')
logging.debug(f'num_angles = {num_angles}')

#%%============================================================================
#% DATA LOCATIONS - 3A AND ID1A3 SAVING SCHEMA
#==============================================================================
indexRegex = re.compile(r'\d+')

# Tomography dark field images
tdf_data_folder = config.get('tdf_data_folder')
assert(tdf_data_folder)
logging.info(f'tdf_data_folder = {tdf_data_folder}')
tdf_files = [f for f in os.listdir(tdf_data_folder)
    if os.path.isfile(os.path.join(tdf_data_folder, f)) and 
            f.endswith('.tif') and indexRegex.search(f)]
tdf_files.sort()
# RV give it a default of up to 20 right now
tdf_num_imgs = msnc.get_num_files(tdf_files, 'dark field image', 20)
logging.debug(f'tdf_num_imgs = {tdf_num_imgs}')
tdf_img_start = int(indexRegex.search(tdf_files[0]).group())
logging.debug(f'tdf_img_start = {tdf_img_start}')

# Tomography bright field images
tbf_data_folder = config.get('tbf_data_folder')
assert(tbf_data_folder)
logging.info(f'tbf_data_folder = {tbf_data_folder}')
tbf_files = [f for f in os.listdir(tbf_data_folder)
    if os.path.isfile(os.path.join(tbf_data_folder, f)) and 
            f.endswith('.tif') and indexRegex.search(f)]
tbf_files.sort()
# RV give it a default of up to 20 right now
tbf_num_imgs = msnc.get_num_files(tbf_files, 'bright field image', 20)
logging.debug(f'tbf_num_imgs = {tbf_num_imgs}')
tbf_img_start = int(indexRegex.search(tbf_files[0]).group())
logging.debug(f'tbf_img_start = {tbf_img_start}')

# Tomography images
num_tomo_data_sets = config.get('num_tomo_data_sets', 1)
logging.info(f'num_tomo_data_sets = {num_tomo_data_sets}')
tomo_data_folders = []
tomo_img_starts = []
tomo_ref_heights = []
for i in range(num_tomo_data_sets):
    if num_tomo_data_sets == 1:
        if 'tomo_data_folder' in config:
            tomo_data_folder = config['tomo_data_folder']
        else:
            tomo_data_folder = config.get(f'tomo_data_folder_{i+1}')
        if 'z_pos' in config:
            tomo_ref_height = config['z_pos']
        else:
            tomo_ref_height = config.get(f'z_pos_{i+1}')
    else:
        tomo_data_folder = config.get(f'tomo_data_folder_{i+1}')
        # Set reference heights relative to the first set
        tomo_ref_height = config.get(f'z_pos_{i+1}')
    assert(tomo_data_folder)
    assert(tomo_ref_height)
    logging.info(f'tomo_data_folder = {tomo_data_folder}')
    logging.info(f'tomo_ref_height = {tomo_ref_height}')
    # Set the reference heights relative to the first set
    if i: tomo_ref_height -= tomo_ref_heights[0]
    tomo_data_folders.append(tomo_data_folder)
    tomo_ref_heights.append(tomo_ref_height)
    tomo_files = [f for f in os.listdir(tomo_data_folder)
        if os.path.isfile(os.path.join(tomo_data_folder, f)) and 
                f.endswith('.tif') and indexRegex.search(f)]
    tomo_files.sort()
    tomo_num_imgs = msnc.get_num_files(tomo_files, 'tomography image', num_angles)
    logging.debug(f'tomo_num_imgs = {tomo_num_imgs}')
    if num_angles != tomo_num_imgs:
        sys.exit(f'Inconsistent number of angles: num_angles = {num_angles}' + 
                f' and tomo_num_imgs = {tomo_num_imgs}')
    tomo_img_start = int(indexRegex.search(tomo_files[0]).group())
    logging.debug(f'tomo_img_start = {tomo_img_start}')
    tomo_img_starts.append(tomo_img_start)
# Set the origin for the reference height at the first set
tomo_ref_heights[0] = 0.

#%%============================================================================
#% GENERATE DARK FIELD
#==============================================================================
tdf = tf.genDark(tdf_data_folder, num_rows, num_columns, tdf_img_start, 
        tdf_num_imgs)
# RV make input of some kind (not always needed)
tdf_cutoff = 21
tdf_below_cutoff = np.where(tdf<=tdf_cutoff, tdf, np.nan)
tdf_mean = np.nanmean(tdf_below_cutoff)
logging.debug(f'tdf_cutoff = {tdf_cutoff}')
logging.debug(f'tdf_mean = {tdf_mean}')
#np.nan_to_num(tdf, nan=tdf_mean) # RV not in this numpi version
where = np.where(tdf>tdf_cutoff)
for i in range(0, where[0].size):
    tdf[where[0][i], where[1][i]] = tdf_mean
msnc.quick_imshow(tdf, 'dark field')

#%%============================================================================
#% GENERATE BRIGHT FIELD
#==============================================================================
tbf = tf.genBright(tbf_data_folder, tdf, num_rows, num_columns, tbf_img_start, 
        tbf_num_imgs, 1)
msnc.quick_imshow(tbf, 'bright field')

#%%============================================================================
#% SET BOUNDS FOR IMAGE STACK
#==============================================================================
if 'x_low' not in config or 'x_upp' not in config:
    print('\nSelect image bounds from bright field')
tbf_x_sum = np.sum(tbf, 1)
x_low = config.get('x_low')
if not x_low:
    x_min = 0
    x_max = tbf.shape[0]
    while True:
        msnc.quick_xyplot(range(x_min, x_max), tbf_x_sum[x_min:x_max], 
                'sum over theta and y', True)
        zoom_flag = pyip.inputInt('Set lower image bound (0) or zoom in (1)?: ', min=0, max=1);
        if zoom_flag:
            x_min = pyip.inputInt(f'    Set lower zoom index [{x_min}, {x_max-1}]: ', 
                    min=x_min, max=x_max-1)
            x_max = pyip.inputInt(f'    Set upper zoom index [{x_min+1}, {x_max}]: ', 
                    min=x_min+1, max=x_max)
        else:
            x_low = pyip.inputInt(f'    Set lower image bound [{x_min}, {x_max}]: ', 
                    min=x_min, max=x_max)
            break
x_upp = config.get('x_upp')
if not x_upp:
    x_min = x_low+1
    x_max = tbf.shape[0]
    while True:
        msnc.quick_xyplot(range(x_min, x_max), tbf_x_sum[x_min:x_max], 
                'sum over theta and y', True)
        if not pyip.inputInt('Set upper image bound (0) or zoom in (1)?: ', min=0, max=1):
            x_upp = pyip.inputInt(f'    Set upper image bound [{x_min}, {x_max}]: ', 
                    min=x_min, max=x_max)
            break
        else:
            x_min = pyip.inputInt(f'    Set lower zoom index [{x_min}, {x_max-1}]: ', 
                    min=x_min, max=x_max-1)
            x_max = pyip.inputInt(f'    Set upper zoom index [{x_min+1}, {x_max}]: ', 
                    min=x_min+1, max=x_max)
img_x_bounds = np.array([x_low, x_upp])
img_y_bounds = np.array([0, num_columns])
logging.debug(f'img_x_bounds: {img_x_bounds}')
logging.debug(f'img_y_bounds: {img_y_bounds}')
msnc.quick_xyplot(range(x_low, x_upp), tbf_x_sum[x_low:x_upp], 
        'sum over theta and y', True)

#%%============================================================================
#% GENERATE RADIOGRAPHS
#==============================================================================
# nor required for analysis, only performed to safe memory
if 'zoom_perc' in config:
    zoom_perc = int(config['zoom_perc'])
else:
    if pyip.inputYesNo('\nDo you want to zoom in to reduce memory requirement (y/n)? '):
        zoom_perc = pyip.inputInt('Enter zoom percentage [1, 100]: ', min=1, max=100)
    else:
        zoom_perc = 100
logging.info(f'zoom_perc = {zoom_perc}')
if 'num_angle_skip' in config:
    num_angle_skip = int(config['num_angle_skip'])
else:
    if pyip.inputYesNo('Do you want to skip angles to reduce memory requirement (y/n)? ') == 'y':
        num_angle_skip = pyip.inputInt('Enter the number skip angle interval' + 
                f' [0, {num_angles-1}]: ', min=0, max=num_angles-1)
    else:
        num_angle_skip = 0
logging.info(f'num_angle_skip = {num_angle_skip}')

#%%============================================================================
for i in range(num_tomo_data_sets):
    logging.info(f'loading set {i+1}/{num_tomo_data_sets}')
    tomo_stack = tf.genTomo(tomo_data_folders[i], tdf, tbf, img_x_bounds, img_y_bounds,
            tomo_img_starts[i], num_angles, num_angle_skip)
    # get rid of infs that may be introduced by normalization
    # RV do this internally?
    tomo_stack = np.where(tomo_stack == np.inf, 0, tomo_stack)

    # adjust sample reference height
    tomo_ref_heights[i] += img_x_bounds[0]*pixel_size

    # Check bounds of image stack
#    tomo_sum = np.sum(tomo_stack, axis=(1,2))
#    tomo_x_sum = np.sum(tomo_stack, 2)
#    logging.debug('Verify selected image bounds')
#    plt.clf()
#    plt.ion()
#    plt.plot([img_x_bounds[0], img_x_bounds[0]], [tomo_x_sum.min(), tomo_x_sum.max()], 'k-', linewidth=2)
#    plt.plot([img_x_bounds[1], img_x_bounds[1]], [tomo_x_sum.min(), tomo_x_sum.max()], 'k-', linewidth=2)
#    plt.plot(range(img_x_bounds[0], img_x_bounds[1]), tomo_x_sum)
#    plt.pause(1)
    
    # Convert tomo_stack to correct shap and axis for quick tomo
    tomo_stack = np.swapaxes(tomo_stack, 0, 1)
#    msnc.quick_imshow(tomo_stack[0,:,:], 'red_stack_fullres_{i+1}', 
#            vmin=0.0, vmax=1.2)
    msnc.quick_imshow(tomo_stack[0,:,:], f'red_stack_fullres_{i+1}')
    # Downsize image to smaller size
    if zoom_perc != 100:
        logging.info('zooming in...')
        tomo_stack = sp.ndimage.zoom(tomo_stack, (1, 0.01*zoom_perc, 0.01*zoom_perc))
        logging.info('... done!')
#        msnc.quick_imshow(tomo_stack[0,:,:], f'red_stack_zoom_{zoom_perc}p_{i+1}', 
#                vmin=0.0, vmax=1.2)
        msnc.quick_imshow(tomo_stack[0,:,:], f'red_stack_zoom_{zoom_perc}p_{i+1}')

    if zoom_perc == 100:
        filename = f'red_stack_fullres_{i+1}.npy'
    else:
        filename = f'red_stack_{zoom_perc}p_{i+1}.npy'
    logging.info('saving ' + filename + ' ...')
    np.save(filename, tomo_stack)
    logging.info('... done!')

#%%============================================================================
#% UPDATE CONFIG FILE
#==============================================================================
if not msnc.searchConfigFile('config.txt', 'Reduced stack parameters'):
    config = msnc.appendConfigFile('config.txt', '# Reduced stack parameters')
if 'x_low' in config:
    config = msnc.updateConfigFile('config.txt', 'x_low', f'{x_low}')
else:
    config = msnc.addtoConfigFile(
            'config.txt', 'Reduced stack parameters', f'x_low = {x_low}')
if 'x_upp' in config:
    config = msnc.updateConfigFile('config.txt', 'x_upp', f'{x_upp}')
else:
    config = msnc.addtoConfigFile('config.txt', 'x_low', f'x_upp = {x_upp}')
if 'zoom_perc' in config:
    config = msnc.updateConfigFile('config.txt', 'zoom_perc', f'{zoom_perc}')
else:
    config = msnc.addtoConfigFile('config.txt', 'x_upp', f'zoom_perc = {zoom_perc}')
if 'num_angle_skip' in config:
    config = msnc.updateConfigFile('config.txt', 'num_angle_skip', f'{num_angle_skip}')
else:
    config = msnc.addtoConfigFile('config.txt', 'zoom_perc', 
            f'num_angle_skip = {num_angle_skip}')
if 'z_ref_1' in config:
    config = msnc.updateConfigFile('config.txt', 'z_ref_1', f'{tomo_ref_heights[0]}')
else:
    config = msnc.addtoConfigFile('config.txt', 'num_angle_skip', 
            f'z_ref_1 = {tomo_ref_heights[0]}')
#RV increase sig figs if needed
for i in range(1, num_tomo_data_sets):
    if 'z_ref_{i+1}' in config:
        config = msnc.updateConfigFile('config.txt', f'z_ref_{i+1}', f'{tomo_ref_heights[i]}')
    else:
        config = msnc.addtoConfigFile('config.txt', f'z_ref_{i}', 
                f'z_ref_{i+1} = {tomo_ref_heights[i]}')



































#%%============================================================================
input('Press any key to continue')
#%%============================================================================

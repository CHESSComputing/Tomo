#!/usr/bin/env python2
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
#import pyinputplus as pyip
import numpy as np
#import scipy as sp
import scipy.ndimage as img
#import matplotlib.pyplot as plt

import msnc_tools as msnc
import tomofunctions as tf

config = msnc.readConfigFile('config.txt')
detector = msnc.readDetectorConfig(config['detector_id'] + '.yml')

#%%============================================================================
#% DETECTOR PARAMETERS
#==============================================================================
num_rows = detector['detector']['pixels']['rows']
logging.debug('num_rows = ' + str(num_rows))
num_columns = detector['detector']['pixels']['columns']
logging.debug('num_columns = ' + str(num_columns))
pixel_size = detector['detector']['pixels']['size'][0]/config['lens_magnification']
logging.debug('pixel_size = ' + str(pixel_size))

start_angle = config['start_angle']
logging.debug('start_angle = ' + str(start_angle))
end_angle = config['end_angle']
logging.debug('end_angle = ' + str(end_angle))
num_angles = config['num_angles']
logging.debug('num_angles = ' + str(num_angles))

#%%============================================================================
#% DATA LOCATIONS - 3A AND ID1A3 SAVING SCHEMA
#==============================================================================
indexRegex = re.compile(r'\d+')

# Tomography dark field images
tdf_data_folder = config['tdf_data_folder']
logging.info('tdf_data_folder = ' + tdf_data_folder)
tdf_files = [f for f in os.listdir(tdf_data_folder)
    if os.path.isfile(os.path.join(tdf_data_folder, f)) and 
            f.endswith('.tif') and indexRegex.search(f)]
tdf_files.sort()
# RV give it a default of up to 20 right now
tdf_num_imgs = msnc.get_num_files(tdf_files, 'dark field image', 20)
logging.debug('tdf_num_imgs = ' + str(tdf_num_imgs))
tdf_img_start = int(indexRegex.search(tdf_files[0]).group())
logging.debug('tdf_img_start = ' + str(tdf_img_start))

# Tomography bright field images
tbf_data_folder = config['tbf_data_folder']
logging.info('tbf_data_folder = ' + tbf_data_folder)
tbf_files = [f for f in os.listdir(tbf_data_folder)
    if os.path.isfile(os.path.join(tbf_data_folder, f)) and 
            f.endswith('.tif') and indexRegex.search(f)]
tbf_files.sort()
# RV give it a default of up to 20 right now
tbf_num_imgs = msnc.get_num_files(tbf_files, 'bright field image', 20)
logging.debug('tbf_num_imgs = ' + str(tbf_num_imgs))
tbf_img_start = int(indexRegex.search(tbf_files[0]).group())
logging.debug('tbf_img_start = ' + str(tbf_img_start))

# Tomography images
if config['num_tomo_data_sets']:
    num_tomo_data_sets = config['num_tomo_data_sets']
else:
    num_tomo_data_sets = 1
logging.info('num_tomo_data_sets = ' + str(num_tomo_data_sets))
tomo_data_folders = []
tomo_img_starts = []
tomo_ref_heights = []
for i in range(num_tomo_data_sets):
    if num_tomo_data_sets == 1:
        if config['tomo_data_folder']:
            tomo_data_folder = config['tomo_data_folder']
        else:
            tomo_data_folder = config['tomo_data_folder_' + str(i+1)]
        if config['z_pos']:
            tomo_ref_height = config['z_pos']
        else:
            tomo_ref_height = config['z_pos_' + str(i+1)]
    else:
        tomo_data_folder = config['tomo_data_folder_' + str(i+1)]
        # Set reference heights relative to the first set
        tomo_ref_height = config['z_pos_' + str(i+1)]
    logging.info('tomo_data_folder = ' + tomo_data_folder)
    logging.info('tomo_ref_height = ' + str(tomo_ref_height))
    # Set the reference heights relative to the first set
    if i: tomo_ref_height -= tomo_ref_heights[0]
    tomo_data_folders.append(tomo_data_folder)
    tomo_ref_heights.append(tomo_ref_height)
    tomo_files = [f for f in os.listdir(tomo_data_folder)
        if os.path.isfile(os.path.join(tomo_data_folder, f)) and 
                f.endswith('.tif') and indexRegex.search(f)]
    tomo_files.sort()
    tomo_num_imgs = msnc.get_num_files(tomo_files, 'tomography image', num_angles)
    logging.debug('tomo_num_imgs = ' + str(tomo_num_imgs))
    if num_angles != tomo_num_imgs:
        sys.exit('Inconsistent number of angles: num_angles = ' + str(num_angles) + 
                ' and tomo_num_imgs = ' + str(tomo_num_imgs))
    tomo_img_start = int(indexRegex.search(tomo_files[0]).group())
    logging.debug('tomo_img_start = ' + str(tomo_img_start))
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
logging.debug('tdf_cutoff = ', str(tdf_cutoff))
logging.debug('tdf_mean = ', str(tdf_mean))
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
if not (config.has_key('x_low') and config.has_key('x_upp')):
    print('\nSelect image bounds from bright field')
tbf_x_sum = np.sum(tbf, 1)
x_low = 0
x_upp = tbf.shape[0]
if config.has_key('x_low'):
    x_low = config['x_low']
else:
    x_min = x_low
    x_max = x_upp
    while True:
        msnc.quick_xyplot(range(x_min, x_max), tbf_x_sum[x_min:x_max], 
                'sum over y and layers')
        zoom_flag = msnc.get_int_in_range(
                'Set lower image bound (0) or zoom in (1)?: ', 0, 1);
        if zoom_flag:
            x_min = msnc.get_int_in_range('    Set lower zoom index (between ' + 
                    str(x_min) + ' and ' + str(x_max-1) + '): ', x_min, x_max-1)
            x_max = msnc.get_int_in_range('    Set upper zoom index (between ' + 
                    str(x_min+1) + ' and ' + str(x_max) + '): ', x_min+1, x_max)
        else:
            x_low = msnc.get_int_in_range('    Set lower image bound (between ' + 
                    str(x_min) + ' and ' + str(x_max) + '): ', x_min, x_max)
            break
if config.has_key('x_upp'):
    x_upp = config['x_upp']
else:
    x_min = x_low
    x_max = x_upp
    while True:
        msnc.quick_xyplot(range(x_min, x_max), tbf_x_sum[x_min:x_max], 
                'sum over y and layers')
        if msnc.get_int_in_range(
                'Set upper image bound (0) or zoom in (1)?: ', 0, 1) == 0:
            x_upp = msnc.get_int_in_range('    Set upper image bound (between ' + 
                    str(x_min) + ' and ' + str(x_max) + '): ', x_min, x_max)
            break
        else:
            x_min = msnc.get_int_in_range('    Set lower zoom index (between ' + 
                    str(x_min) + ' and ' + str(x_max-1) + '): ', x_min, x_max-1)
            x_max = msnc.get_int_in_range('    Set upper zoom index (between ' + 
                    str(x_min+1) + ' and ' + str(x_max) + '): ', x_min+1, x_max)
img_x_bounds = np.array([x_low, x_upp])
img_y_bounds = np.array([0, num_columns])
logging.debug('img_x_bounds' + str(img_x_bounds))
logging.debug('img_y_bounds' + str(img_y_bounds))
msnc.quick_xyplot(range(x_low, x_upp), tbf_x_sum[x_low:x_upp], 'sum over theta and y')

#%%============================================================================
#% GENERATE RADIOGRAPHS
#==============================================================================
# nor required for analysis, only performed to safe memory
if config.has_key('zoom_perc'):
    zoom_perc = int(config['zoom_perc'])
else:
    if msnc.get_yes_no('\nDo you want to zoom in to reduce memory requirement (y/n)? '):
        zoom_perc = msnc.get_int_in_range('Enter zoom percentage [1, 100]: ', 1, 100)
    else:
        zoom_perc = 100
logging.info('zoom_perc = ' + str(zoom_perc))
if config.has_key('num_angle_skip'):
    num_angle_skip = int(config['num_angle_skip'])
else:
    if msnc.get_yes_no('Do you want to skip angles to reduce memory requirement (y/n)? '):
        num_angle_skip = msnc.get_int_in_range('Enter the number skip angle interval [0, ' + 
                str(num_angles-1) + ']: ', 0, num_angles-1)
    else:
        num_angle_skip = 0
logging.info('num_angle_skip = ' + str(num_angle_skip))

#%%============================================================================
for i in range(num_tomo_data_sets):
    logging.info('loading set ' + str(i+1) + '/' + str(num_tomo_data_sets))
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
#    msnc.quick_imshow(tomo_stack[0,:,:], 'red_stack_fullres_' + str(i+1), 
#            vmin=0.0, vmax=1.2)
    msnc.quick_imshow(tomo_stack[0,:,:], 'red_stack_fullres_' + str(i+1))
    # Downsize image to smaller size
    if zoom_perc != 100:
        logging.info('zooming in...')
        tomo_stack = img.zoom(tomo_stack, (1, 0.01*zoom_perc, 0.01*zoom_perc))
        logging.info('... done!')
        msnc.quick_imshow(tomo_stack[0,:,:], 'red_stack_zoom_' + 
                str(zoom_perc) + 'p_' + str(i+1), vmin=0.0, vmax=1.2)

    if zoom_perc == 100:
        filename = 'red_stack_fullres_' + str(i+1) + '.npy'
    else:
        filename = 'red_stack_' + str(zoom_perc) + 'p_' + str(i+1) + '.npy'
    logging.info('saving ' + filename + ' ...')
    np.save(filename, tomo_stack)
    logging.info('... done!')

#%%============================================================================
#% UPDATE CONFIG FILE
#==============================================================================
if not msnc.searchConfigFile('config.txt', 'Reduced stack parameters'):
    config = msnc.appendConfigFile('config.txt', '# Reduced stack parameters')
if config.has_key('x_low'):
    config = msnc.updateConfigFile('config.txt', 'x_low', str(x_low))
else:
    config = msnc.addtoConfigFile(
            'config.txt', 'Reduced stack parameters', 'x_low = ' + str(x_low))
if config.has_key('x_upp'):
    config = msnc.updateConfigFile('config.txt', 'x_upp', str(x_upp))
else:
    config = msnc.addtoConfigFile('config.txt', 'x_low', 'x_upp = ' + str(x_upp))
if config.has_key('zoom_perc'):
    config = msnc.updateConfigFile('config.txt', 'zoom_perc', str(zoom_perc))
else:
    config = msnc.addtoConfigFile('config.txt', 'x_upp', 'zoom_perc = ' + str(zoom_perc))
if config.has_key('num_angle_skip'):
    config = msnc.updateConfigFile('config.txt', 'num_angle_skip', str(num_angle_skip))
else:
    config = msnc.addtoConfigFile('config.txt', 'zoom_perc', 'num_angle_skip = ' + 
            str(num_angle_skip))
if config.has_key('z_ref_1'):
    config = msnc.updateConfigFile('config.txt', 'z_ref_1', str(tomo_ref_heights[0]))
else:
    config = msnc.addtoConfigFile('config.txt', 'num_angle_skip', 'z_ref_1 = ' + 
            str(tomo_ref_heights[0]))
#RV increase sig figs if needed
for i in range(1, num_tomo_data_sets):
    z_ref_key = 'z_ref_' + str(i+1)
    if config.has_key('z_ref_' + str(i+1)):
        config = msnc.updateConfigFile('config.txt', 'z_ref_' + str(i+1), 
                str(tomo_ref_heights[i]))
    else:
        config = msnc.addtoConfigFile('config.txt', 'z_ref_' + str(i), 'z_ref_' + 
                str(i+1) + ' = ' + str(tomo_ref_heights[i]))

#%%============================================================================
raw_input('Press any key to continue')
#%%============================================================================

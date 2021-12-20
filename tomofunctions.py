# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:54:37 2021

@author: rv43

Based on tomoFunctions2.py of Tue Jul 31 09:08:56 2018 by jk989
""" 
#import os
import sys
#import re
import numpy as np
#import scipy as sp
import scipy.ndimage as img
#import matplotlib.pyplot as plt
import logging
import tomopy

#####F2 LOAD FUNCTIONS####
def genDark(tdf_data_folder, num_rows, num_columns, tdf_img_start, tdf_num_imgs,
        num_img_skip = None):
    if num_img_skip is None: num_img_skip = 0
    tdf_img_range = np.arange(tdf_img_start, tdf_img_start+tdf_num_imgs, num_img_skip+1)

    logging.debug('Loading data to get median dark field...')
    tdf_stack = np.zeros([len(tdf_img_range), num_rows, num_columns])
    for i in range(len(tdf_img_range)):
        logging.debug('    loading ' + str(i+1) + '/' + str(len(tdf_img_range)) + ' ' + tdf_data_folder + 
                'nf_%0.6d.tif'%(tdf_img_range[i]))
        tdf_stack[i, :, :] = img.imread(tdf_data_folder + 'nf_%0.6d.tif'%(tdf_img_range[i]))
        #image_stack[i, :, :] = np.flipud(tmp_img>threshold)

    # take the median
    tdf = np.median(tdf_stack, axis=0)
    logging.debug('... done!')
    return tdf

def genBright(tbf_data_folder, tdf, num_rows, num_columns, tbf_img_start, tbf_num_imgs,
        num_img_skip = None):
    if num_img_skip is None: num_img_skip = 0
    tbf_img_range = np.arange(tbf_img_start, tbf_img_start+tbf_num_imgs, num_img_skip+1)

    logging.debug('Loading data to get median bright field...')
    tbf_stack = np.zeros([len(tbf_img_range), num_rows, num_columns])
    for i in np.arange(len(tbf_img_range)):
        logging.debug('    loading ' + str(i+1) + '/' + str(len(tbf_img_range)) + ' ' + tbf_data_folder + 
                'nf_%0.6d.tif'%(tbf_img_range[i]))
        tbf_stack[i, :, :] = img.imread(tbf_data_folder + 'nf_%0.6d.tif'%(tbf_img_range[i])) - tdf
        #image_stack[ii, :, :] = np.flipud(tmp_img>threshold)

    # take the median
    tbf = np.median(tbf_stack, axis=0)
    logging.debug('... done!')
    return tbf

def genTomo(tomo_data_folder, tdf, tbf, img_x_bounds, img_y_bounds, tomo_img_start, 
        tomo_num_imgs, num_img_skip = None, int_corr = None):
    if num_img_skip is None: num_img_skip = 0
    if int_corr is None: int_corr = np.ones(tomo_num_imgs)
    if tomo_num_imgs != len(int_corr):
        sys.exit('Inconsistent number of images: tomo_num_imgs = ' + str(tomo_num_imgs) + 
                ' and len(int_corr) = ' + str(len(int_corr)))

    tomo_img_range = np.arange(tomo_img_start, tomo_img_start+tomo_num_imgs, num_img_skip+1)
    int_corr_range = int_corr[0:tomo_num_imgs:num_img_skip+1]

    #numbers for intensity corrections values of ic0/median of ic0
    logging.debug('Loading tomography images, applying data window, removing negative values, ' +
            'applying intensity correction, and building radiographs...')
#    tomo_stack = np.zeros([1, len(tomo_img_range), img_x_bounds[1]-img_x_bounds[0], 
    tomo_stack = np.zeros([len(tomo_img_range), img_x_bounds[1]-img_x_bounds[0], 
            img_y_bounds[1]-img_y_bounds[0]])
    for i in np.arange(len(tomo_img_range)):
#        logging.debug('    loading ' + str(i+1) + '/' + str(len(tomo_img_range)) + ' ' +
#                tomo_data_folder + 'nf_%0.6d.tif'%(tomo_img_range[i]))
        if not (i+1)%40: logging.info('    loading ' + str(i+1) + '/' + str(len(tomo_img_range)) + 
                ' ' + tomo_data_folder + 'nf_%0.6d.tif'%(tomo_img_range[i]))
        # loading data
        tomo_img = img.imread(tomo_data_folder + 'nf_%0.6d.tif'%(tomo_img_range[i]))
        # applying data window, removing negative values and applying intensity correction
        tmp_img = tomopy.misc.corr.remove_neg(
                tomo_img[img_x_bounds[0]:img_x_bounds[1], img_y_bounds[0]:img_y_bounds[1]] - 
                tdf[img_x_bounds[0]:img_x_bounds[1], img_y_bounds[0]:img_y_bounds[1]], 
                val=0.0) * int_corr_range[i]
        # normalize in some way idk RV???
        # RV: should we normalize with (tbf-tdf) instead of tbf alone? see also: tomopy.normalize(proj, flat, dark)
        # RV: this can give inf's, get rid of them here?
#        tomo_stack[0, i, :, :] = tomopy.prep.normalize.minus_log(
        tomo_stack[i, :, :] = tomopy.prep.normalize.minus_log(
                tmp_img / tbf[img_x_bounds[0]:img_x_bounds[1], img_y_bounds[0]:img_y_bounds[1]])
    logging.debug("... done!")
#    return np.swapaxes(tomo_stack, 1, 2)
    return np.swapaxes(tomo_stack, 0, 1)

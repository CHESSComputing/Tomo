#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:54:37 2021

@author: rv43

Based on tomoFunctions2.py of Tue Jul 31 09:08:56 2018 by jk989
""" 
import tomopy # the TomoPy instruction say to import tomopy before numpy
import sys
import numpy as np
import imageio as img
import logging

def genDark(tdf_data_folder, num_rows, num_columns, tdf_img_start, tdf_num_imgs,
        num_img_skip = None):
    if num_img_skip is None: num_img_skip = 0
    tdf_img_range = np.arange(tdf_img_start, tdf_img_start+tdf_num_imgs, num_img_skip+1)

    logging.debug('Loading data to get median dark field...')
    tdf_stack = np.zeros([len(tdf_img_range), num_rows, num_columns])
    for i in range(len(tdf_img_range)):
        logging.debug(f'    loading {i+1}/{len(tdf_img_range)} {tdf_data_folder}' + 
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
        logging.debug(f'    loading {i+1}/{len(tbf_img_range)} {tbf_data_folder}' + 
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
        sys.exit(f'Inconsistent number of images: tomo_num_imgs = {tomo_num_imgs}' + 
                f' and len(int_corr) = {len(int_corr)}')

    tomo_img_range = np.arange(tomo_img_start, tomo_img_start+tomo_num_imgs, num_img_skip+1)
    int_corr_range = int_corr[0:tomo_num_imgs:num_img_skip+1]

    #numbers for intensity corrections values of ic0/median of ic0
    logging.debug('Loading tomography images, applying data window, removing negative values, ' +
            'applying intensity correction, and building radiographs...')
#    tomo_stack = np.zeros([1, len(tomo_img_range), img_x_bounds[1]-img_x_bounds[0], 
    tomo_stack = np.zeros([len(tomo_img_range), img_x_bounds[1]-img_x_bounds[0], 
            img_y_bounds[1]-img_y_bounds[0]])
    for i in np.arange(len(tomo_img_range)):
        if not (i+1)%40: logging.info(f'    loading {i+1}/{len(tomo_img_range)}' + 
                f' {tomo_data_folder}' + 'nf_%0.6d.tif'%(tomo_img_range[i]))
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

def reconstruct(sinograms, centers, image_bounds, layers, theta, sigma = .1, ncore = 4, 
        algorithm = 'gridrec', run_secondary_sirt=False, secondary_iter=100):
    """
    Reconstructs object from projection data.
    Takes in list [elements, rows, angles, cols]
    or [rows, angles, cols],
    and returns ndarray representing a 3D reconstruction
    
    Parameters
    ----------
    sinograms : ndarray
        3D tomographic data.
    centers : scalar, array
        estimated location(s) of rotation axis relative to image_bounds
    image_bounds : len 2 array
        boundary of sample to be evaluated
    layers : scalar, len 2 array
        single layer or bounds of layers
    theta : ndarray
        list of angles used in tomography data (radians)
    sigma : float
        damping param in Fourier space
    ncore : int
        # of cores that will be assigned
    algorithm : {str, function}
        see tomopy.recon.algorithm for list of algs
    
    Returns
    -------
    ndarray
        Reconstructed multi-elemental 3D object.
    """
    #normalizing recon params
    if np.isscalar(layers):
        layers = [layers, layers+1]
    if np.isscalar(centers):
        assert len(layers) >=  2, 'Mismatch between centers layers'
        centers = np.linspace(centers, centers, layers[1]-layers[0])
    elif len(centers) == 2:
        assert len(layers) >=  2, 'Mismatch between centers layers'
        centers = np.linspace(centers[0], centers[1], layers[1]-layers[0])
    else:
        assert len(centers) == layers[1]-layers[0], 'unequal # of centers and layers'
    #begin reconstruction
    if len(sinograms.shape) == 4:
        sinograms = np.swapaxes(sinograms,1,2)
        recons = np.zeros([len(sinograms),layers[1]-layers[0],image_bounds[1]-image_bounds[0],
                image_bounds[1]-image_bounds[0]])
        recon_clean = np.zeros([len(sinograms), layers[1]-layers[0],image_bounds[1]-image_bounds[0],
                image_bounds[1]-image_bounds[0]])
        for el in range(len(sinograms)):
            logging.info(f'Processing element {el+1}/{len(sinograms)}')
            for x in range(layers[1]-layers[0]):
                if not (x+1)%20:
                    logging.info(f'Processing layer {x+1}/{layers[1]-layers[0]}')
                tmp = tomopy.prep.stripe.remove_stripe_fw(
                        sinograms[el,:,layers[0]+x:layers[0]+x+1,image_bounds[0]:image_bounds[1]],
                                sigma=sigma, ncore=ncore)
                tmp_recon = tomopy.recon(tmp, theta, center=(image_bounds[1]-image_bounds[0])/2.0+centers[x], 
                        algorithm=algorithm, sinogram_order=False, ncore=ncore)
                if run_secondary_sirt:
                    options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':secondary_iter}
                    recon = tomopy.recon(tmp, theta, center = (image_bounds[1]-image_bounds[0])/2.0+centers[x],
                            init_recon=tmp_recon, algorithm=tomopy.astra, options=options, sinogram_order=False, ncore=ncore)
                    recons[el][x] += recon[0]
                else:
                    recons[el][x] += tmp_recon
            recon_clean[el,:] = tomopy.misc.corr.remove_ring(recons[el,:], rwidth=17)
    #if there is no element dimension
    elif len(sinograms.shape) == 3:
        sinograms = np.swapaxes(sinograms,0,1)
        recons = np.zeros([1,layers[1]-layers[0],image_bounds[1]-image_bounds[0],
                image_bounds[1]-image_bounds[0]])
        recon_clean = np.zeros([1,layers[1]-layers[0],image_bounds[1]-image_bounds[0],
                image_bounds[1]-image_bounds[0]])
        for x in range(layers[1]-layers[0]):
            if not (x+1)%20:
                logging.info(f'Processing layer {x+1}/{layers[1]-layers[0]}')
            tmp = tomopy.prep.stripe.remove_stripe_fw(
                    sinograms[:,layers[0]+x:layers[0]+x+1,image_bounds[0]:image_bounds[1]],
                            sigma=sigma, ncore=ncore)
            tmp_recon = tomopy.recon(tmp, theta, center=(image_bounds[1]-image_bounds[0])/2.0+centers[x], 
                    algorithm=algorithm,sinogram_order=False,ncore=ncore)
            
            if run_secondary_sirt:
                options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':secondary_iter}
                recon = tomopy.recon(tmp, theta, center=(image_bounds[1]-image_bounds[0])/2.0+centers[x],
                        init_recon=tmp_recon, algorithm=tomopy.astra, options=options, sinogram_order=False, ncore=ncore)
                recons[x] += recon[0]
            else:
                recons[x] += tmp_recon
        recon_clean[0,:] = tomopy.misc.corr.remove_ring(recons[0,:],rwidth=17)

    if layers[1]-layers[0] != 1: logging.info('complete')
    return recon_clean

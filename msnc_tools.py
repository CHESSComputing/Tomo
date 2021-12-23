#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:36:22 2021

@author: rv43
"""

import logging

import os
import sys
import pyinputplus as pyip
import numpy as np
import imageio as img
import matplotlib.pyplot as plt

def loadConfigFile(filepath):
    # Ensure the file exists before opening
    if not os.path.isfile(filepath):
        logging.error(f'file does not exist: {filepath}')
        return
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    config = {}
    for line in lines:
        line = line.split('#')[0]
        if '=' in line:
            key, value = tuple(line.split('='))
            key = key.replace(' ', '')
            config[key] = eval(value)
    return config

def searchConfigFile(config_filepath, search_string):
    with open(config_filepath, 'r') as f:
        lines = f.read()
        if search_string in lines:
            return True
    return False

def appendConfigFile(config_filepath, newlines):
    with open(config_filepath, 'a') as f:
        f.write('\n')
        for line in newlines.splitlines():
            f.write(f'{line}\n')
    # update config in memory
    return loadConfigFile(config_filepath)

def updateConfigFile(config_filepath, keyword, keyvalue):
    with open(config_filepath, 'r') as f:
        lines = f.read().splitlines()
    update = False
    for index in range(len(lines)):
        line = lines[index].split('#')[0]
        if keyword in line:
            key, value = tuple(line.split('='))
            key = key.replace(' ', '')
            lines[index] = f'{key} = {keyvalue}'
            update = True
            break
    if not update:
        lines += ['', f'{keyword} = {keyvalue}']
    with open(config_filepath, 'w') as f: 
        for line in lines:
            f.write(f'{line}\n')
    # update config in memory
    return loadConfigFile(config_filepath)

def addtoConfigFile(config_filepath, search_string, newlines):
    with open(config_filepath, 'r') as f:
        lines = f.read().splitlines()
    update = False
    for index in range(len(lines)):
        line = lines[index]
        if search_string in line:
            lines = lines[:index+1] + newlines.splitlines() + lines[index+1:]
            update = True
            break
    if not update:
        lines += [''] + newlines.splitlines()
    with open(config_filepath, 'w') as f:
        for line in lines:
            f.write(f'{line}\n')
    # update config in memory
    return loadConfigFile(config_filepath)

def getNumFiles(files, file_type, num_angles = None):
    num_files = len(files)
    # RV assume that the order is correct and that the angles match the images
    if num_angles is not None and num_files >= num_angles:
        return num_angles
    if num_files:
        if num_files == 1:
            logging.debug(f'found {num_files} {file_type}')
        else:
            logging.debug(f'found {num_files} {file_type}s')
            num_files = pyip.inputInt('How many would you like to use (enter 0 for all)?: ', 
                    min=0, max=num_files)
            if not num_files:
                num_files = len(files)
    return num_files

def loadImage(filepath, img_x_bounds=None, img_y_bounds=None):
    """Load a single image from file."""
    if not os.path.isfile(filepath):
       logging.error(f'Unable to load {filepath}')
       return None
    img_read = img.imread(filepath)
    if not img_x_bounds:
        img_x_bounds = [0, img_read.shape[0]]
    else:
        if (type(img_x_bounds) != list or len(img_x_bounds) != 2 or 
                img_x_bounds[0] < 0 or img_x_bounds[1] > img_read.shape[0]):
            logging.error(f'inconsistent row dimension in {filepath}')
            return None
    if not img_y_bounds:
        img_y_bounds = [0, img_read.shape[1]]
    else:
        if (type(img_y_bounds) != list or len(img_y_bounds) != 2 or 
                img_y_bounds[0] < 0 or img_y_bounds[1] > img_read.shape[1]):
            logging.error(f'inconsistent column dimension in {filepath}')
            return None
    return img_read[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]

def loadImageStack(img_folder, img_start, num_imgs, num_img_skip=0,
            img_x_bounds=None, img_y_bounds=None):
    """Load a set of images and return them as a stack."""
    img_range = np.arange(img_start, img_start+num_imgs, num_img_skip+1)
    filepath = img_folder + 'nf_%0.6d.tif'%(img_range[0])
    logging.debug(f'    loading 1/{len(img_range)}: {filepath}')
    img_read = loadImage(filepath, img_x_bounds, img_y_bounds)
    img_stack = np.expand_dims(img_read, 0)
    for i in range(1, len(img_range)):
        filepath = img_folder + 'nf_%0.6d.tif'%(img_range[i])
        if not (i+1)%20:
            logging.info(f'    loading {i+1}/{len(img_range)}: {filepath}')
        else:
            logging.debug(f'    loading {i+1}/{len(img_range)}: {filepath}')
        img_stack = np.concatenate((img_stack, np.expand_dims(loadImage(
                filepath, img_x_bounds, img_y_bounds), 0)))
    return img_stack

def quickImshow(a, title=None, save_figname=None, clear=False, **kwargs):
    if clear:
        if title:
            plt.close(fig=title)
        else:
            plt.clf()
    plt.ion()
    plt.figure(title)
    plt.imshow(a, **kwargs)
    if save_figname:
        plt.savefig(save_figname)
    plt.pause(1)

def quickPlot(y, title=None, clear=False):
    if clear:
        if title:
            plt.close(fig=title)
        else:
            plt.clf()
    plt.ion()
    plt.figure(title)
    plt.plot(y)
    plt.pause(1)

def quickXyplot(x, y, title=None, clear=False):
    if clear:
        if title:
            plt.close(fig=title)
        else:
            plt.clf()
    plt.ion()
    plt.figure(title)
    plt.plot(x, y)
    plt.pause(1)

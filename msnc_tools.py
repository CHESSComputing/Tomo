#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:36:22 2021

@author: rv43
"""

import logging

import os
import sys
import re
import pyinputplus as pyip
import numpy as np
import imageio as img
import matplotlib.pyplot as plt
from time import time

def depth_list(L): return isinstance(L, list) and max(map(depth_list, L))+1
def depth_tuple(T): return isinstance(T, tuple) and max(map(depth_tuple, T))+1

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

def selectFiles(files, name=None, num_required=None):
    indexRegex = re.compile(r'\d+')
    num_files = len(files)
    if type(name) != str:
        name = ' '
    else:
        name = f' {name} '
    if len(files) < 1:
        logging.warning('No available' + name + 'files')
        return (0, 0)
    if num_required == None:
        if len(files) == 1:
            first_index = indexRegex.search(files[0]).group()
            if first_index == None:
                logging.error('Unable to find correctly indexed' + name + 'images')
                return (0, 0)
            return (int(first_index), 1)
    else:
        if type(num_required) != int or num_required < 1:
            logging.error(f'Illegal value of num_required ({num_required}, ' +
                    f'{type(num_required)})')
            return (0, 0)
        if num_files < num_required:
            logging.error('Unable to find the required' + name +
                    f'images ({num_files} out of {num_required})')
            return (0, 0)
    files.sort()
    first_index = indexRegex.search(files[0]).group()
    last_index = indexRegex.search(files[-1]).group()
    if first_index == None or last_index == None:
        logging.error('Unable to find correctly indexed' + name + 'images')
        return (0, 0)
    else:
        first_index = int(first_index)
        last_index = int(last_index)
        if len(files) != last_index-first_index+1:
            logging.warning('Non-consecutive set of indices for' + name + 'images')
        if num_required and last_index-first_index+1 < num_required:
            logging.error('Unable to find the required' + name +
                    f'images ({last_index-first_index+1} out of {num_required})')
            return (0, 0)
        print('\nNumber of available' + name + f'images: {len(files)}')
        print('Index range of available' + name + f'images: [{first_index}, ' +
                f'{last_index}]')
        if num_required == None:
            use_all = f'Use all ([{first_index}, {last_index}])'
            pick_offset = 'Pick a starting index offset and a number of images'
            pick_bounds = 'Pick the first and last index'
            menuchoice = pyip.inputMenu([use_all, pick_offset, pick_bounds], numbered=True)
            if menuchoice == use_all:
                return (first_index, len(files))
            elif menuchoice == pick_offset:
                first_index += pyip.inputInt('Enter the starting index offset' +
                        f' [0, {last_index-first_index}]: ', min=0,
                        max=last_index-first_index)
                if first_index == last_index:
                    return (first_index, 1)
                else:
                    return (first_index, pyip.inputInt('Enter the number of images' +
                            f' [1, {last_index-first_index+1}]: ', min=1,
                            max=last_index-first_index+1))
            else:
                first_index = pyip.inputInt('Enter the starting index ' +
                        f'[{first_index}, {last_index}]: ', min=first_index, max=last_index)
                return (first_index, pyip.inputInt('Enter the last index ' +
                        f'[{first_index}, {last_index}]: ', min=first_index,
                        max=last_index)-first_index+1)
        else:
            use_all = f'Use ([{first_index}, {first_index+num_required-1}])'
            pick_offset = 'Pick a starting index offset'
            menuchoice = pyip.inputMenu([use_all, pick_offset], numbered=True)
            if menuchoice == pick_offset:
                first_index += pyip.inputInt('Enter the starting index offset' +
                        f'[0, {last_index-first_index-num_required}]: ',
                        min=0, max=last_index-first_index-num_required)
            return (first_index, num_required)

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
    logging.debug(f'img_start = {img_start}')
    logging.debug(f'num_imgs = {num_imgs}')
    logging.debug(f'num_img_skip = {num_img_skip}')
    img_range = np.arange(img_start, img_start+num_imgs, num_img_skip+1)
    num_read = len(img_range)
    if num_read == 1:
        logging.info(f'Reading {num_read} image ...')
    else:
        logging.info(f'Reading {num_read} images ...')
    t0 = time()
    img_read_stack = []
    for i in range(0, num_read):
        filepath = f'{img_folder}nf_{img_range[i]:06d}.tif'
        if not (i+1)%20:
            logging.info(f'    loading {i+1}/{len(img_range)}: {filepath}')
        else:
            logging.debug(f'    loading {i+1}/{len(img_range)}: {filepath}')
        img_read = loadImage(filepath, img_x_bounds, img_y_bounds)
        img_read_stack.append(np.expand_dims(img_read, 0))
    img_stack = np.concatenate([img_read for img_read in img_read_stack])
    logging.info(f'... done in {time()-t0:.2f} seconds!')
    logging.debug(f'img_stack shape = {np.shape(img_stack)}')
    del img_read_stack, img_read, img_range
    return img_stack

def quickImshow(a, title=None, save_fig=False, save_only=False, clear=True, **kwargs):
    if title != None and type(title) != str:
        logging.error('Illegal entry for title in quickImshow')
        return
    if type(save_fig) != bool:
        logging.error('Illegal entry for save_fig in quickImshow')
        return
    if type(save_only) != bool:
        logging.error('Illegal entry for save_only in quickImshow')
        return
    if type(clear) != bool:
        logging.error('Illegal entry for clear in quickImshow')
        return
    if not title:
        title='quick_imshow'
    else:
        title = re.sub(r"\s+", '_', title)
    if clear:
        plt.close(fig=title)
    if save_only:
        plt.figure(title)
        plt.imshow(a, **kwargs)
        if save_fig:
            plt.savefig(f'{title}.png')
        plt.close(fig=title)
        #if save_fig:
        #    plt.imsave(f'{title}.png', a, **kwargs)
    else:
        plt.ion()
        plt.figure(title)
        plt.imshow(a, **kwargs)
        if save_fig:
            plt.savefig(f'{title}.png')
        plt.pause(1)

def quickPlot(*args, title=None, save_fig=False, save_only=False, clear=True, **kwargs):
    if title != None and type(title) != str:
        logging.error('Illegal entry for title in quickPlot')
        return
    if type(save_fig) != bool:
        logging.error('Illegal entry for save_fig in quickPlot')
        return
    if type(save_only) != bool:
        logging.error('Illegal entry for save_only in quickPlot')
        return
    if type(clear) != bool:
        logging.error('Illegal entry for clear in quickPlot')
        return
    if not title:
        title = 'quick_plot'
    else:
        title = re.sub(r"\s+", '_', title)
    if clear:
        plt.close(fig=title)
    if save_only:
        plt.figure(title)
        if depth_tuple(args) > 1:
           for y in args:
               plt.plot(*y, **kwargs)
        else:
            plt.plot(*args, **kwargs)
        if save_fig:
            plt.savefig(f'{title}.png')
        plt.close(fig=title)
    else:
        plt.ion()
        plt.figure(title)
        if depth_tuple(args) > 1:
           for y in args:
               print(f'y = {y} {type(y)}')
               plt.plot(*y, **kwargs)
        else:
            plt.plot(*args, **kwargs)
        if save_fig:
            plt.savefig(f'{title}.png')
        plt.pause(1)

def selectArrayBounds(a, x_low=None, x_upp=None, num_x_min=None,
        title='select array bounds'):
    """Interactively select the lower and upper data bounds for a numpy array."""
    if type(a) != np.ndarray or a.ndim != 1:
        logging.error('Illegal array type or dimension in selectArrayBounds')
        return None
    if num_x_min == None:
        num_x_min = 1
    else:
        if num_x_min < 2 or num_x_min > a.size:
            logging.warning('Illegal input for num_x_min in selectArrayBounds, input ignored')
            num_x_min = 1
    if x_low == None:
        x_min = 0
        x_max = a.size
        x_low_max = a.size-num_x_min
        while True:
            quickPlot(range(x_min, x_max), a[x_min:x_max], title=title)
            zoom_flag = pyip.inputInt('Set lower data bound (0) or zoom in (1)?: ',
                    min=0, max=1)
            if zoom_flag:
                x_min = pyip.inputInt(f'    Set lower zoom index [0, {x_low_max}]: ',
                        min=0, max=x_low_max)
                x_max = pyip.inputInt(f'    Set upper zoom index [{x_min+1}, {x_low_max+1}]: ',
                        min=x_min+1, max=x_low_max+1)
            else:
                x_low = pyip.inputInt(f'    Set lower data bound [0, {x_low_max}]: ',
                        min=0, max=x_low_max)
                break
    else:
        if type(x_low) != int or x_low < 0 or x_low >= a.size:
            logging.error('Illegal x_low input in selectArrayBounds')
            return None
    if x_upp == None:
        x_min = x_low+num_x_min
        x_max = a.size
        x_upp_min = x_min
        while True:
            quickPlot(range(x_min, x_max), a[x_min:x_max], title=title)
            zoom_flag = pyip.inputInt('Set upper data bound (0) or zoom in (1)?: ',
                    min=0, max=1)
            if zoom_flag:
                x_min = pyip.inputInt(f'    Set upper zoom index [{x_upp_min}, {a.size-1}]: ',
                        min=x_upp_min, max=a.size-1)
                x_max = pyip.inputInt(f'    Set upper zoom index [{x_min+1}, {a.size}]: ',
                        min=x_min+1, max=a.size)
            else:
                x_upp = pyip.inputInt(f'    Set upper data bound [{x_upp_min}, {a.size}]: ',
                        min=x_upp_min, max=a.size)
                break
    else:
        if type(x_upp) != int or x_upp < 1 or x_upp > a.size:
            logging.error('Illegal x_upp input in selectArrayBounds')
            return None
    print(f'lower bound = {x_low} (inclusive)\nupper bound = {x_upp} (exclusive)]')
    bounds = [x_low, x_upp]
    quickPlot(range(bounds[0], bounds[1]), a[bounds[0]:bounds[1]], title=title)
    if pyip.inputYesNo('Accept these bounds (y/n)?: ') == 'no':
        bounds = selectArrayBounds(a, title=title)
    return bounds


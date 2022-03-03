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
import h5py
import pyinputplus as pyip
import numpy as np
import imageio as img
import matplotlib.pyplot as plt
from time import time
from ast import literal_eval
from lmfit.models import StepModel, RectangleModel

def depth_list(L): return isinstance(L, list) and max(map(depth_list, L))+1
def depth_tuple(T): return isinstance(T, tuple) and max(map(depth_tuple, T))+1

def is_int(v, v_min=None, v_max=None):
    """Value is an integer in range v_min <= v <= v_max"""
    if not isinstance(v, int):
        return False
    if (v_min != None and v < v_min) or (v_max != None and v > v_max):
        return False
    return True

def is_num(v, v_min=None, v_max=None):
    """Value is a number in range v_min <= v <= v_max"""
    if not isinstance(v, (int,float)):
        return False
    if (v_min != None and v < v_min) or (v_max != None and v > v_max):
        return False
    return True

def is_index(v, v_min=0, v_max=None):
    """Value is an array index in range v_min <= v < v_max"""
    if not isinstance(v, int):
        return False
    if v < v_min or (v_max != None and v >= v_max):
        return False
    return True

def is_index_range(v, v_min=0, v_max=None):
    """Value is an array index range in range v_min <= v[0] <= v[1] < v_max"""
    if not (isinstance(v, list) and len(v) == 2 and isinstance(v[0], int) and
            isinstance(v[1], int)):
        return False
    if not 0 <= v[0] < v[1] or (v_max != None and v[1] >= v_max):
        return False
    return True

def illegal_value(name, value, location=None, exit_flag=False):
    if not isinstance(location, str):
        location = ''
    else:
        location = f'in {location} '
    if isinstance(name, str):
        logging.error(f'Illegal value for {name} {location}({value}, {type(value)})')
    else:
        logging.error(f'Illegal value {location}({value}, {type(value)})')
    if exit_flag:
        exit(1)

def get_trailing_int(string):
    indexRegex = re.compile(r'\d+$')
    mo = indexRegex.search(string)
    if mo == None:
        return None
    else:
        return int(mo.group())

def loadConfigFile(config_file):
    if not os.path.isfile(config_file):
        logging.error(f'Unable to load {config_file}')
        return {}
    with open(config_file, 'r') as f:
        lines = f.read().splitlines()
    return {item[0].strip():literal_eval(item[1].strip()) for item in
            [line.split('#')[0].split('=') for line in lines if '=' in line.split('#')[0]]}

def searchConfigFile(config_file, search_string):
    if not os.path.isfile(config_file):
        logging.error(f'Unable to load {config_file}')
        return False
    with open(config_file, 'r') as f:
        lines = f.read()
        if search_string in lines:
            return True
    return False

def appendConfigFile(config_file, new_lines):
    with open(config_file, 'a') as f:
        f.write('\n')
        for line in new_lines.splitlines():
            f.write(f'{line}\n')
    # Update config in memory
    return loadConfigFile(config_file)

def updateConfigFile(config_file, key, value, search_string=None, header=None):
    if not os.path.isfile(config_file):
        logging.error(f'Unable to load {config_file}')
        lines = []
    else:
        with open(config_file, 'r') as f:
            lines = f.read().splitlines()
    config = {item[0].strip():literal_eval(item[1].strip()) for item in
            [line.split('#')[0].split('=') for line in lines if '=' in line.split('#')[0]]}
    if not isinstance(key, str):
        logging.error(f'Illegal key input type in updateConfigFile ({type(key)})')
        return config
    if isinstance(value, str):
        newline = f"{key} = '{value}'"
    else:
        newline = f'{key} = {value}'
    if key in config.keys():
        # Update key with value
        for index,line in enumerate(lines):
            if '=' in line:
                item = line.split('#')[0].split('=')
                if item[0].strip() == key:
                    lines[index] = newline
                    break
    else:
        # Insert new key/value pair
        if search_string != None:
            if isinstance(search_string, str):
                search_string = [search_string]
            elif not isinstance(search_string, (tuple, list)):
                logging.error('Illegal search_string input in updateConfigFile'+
                        f'(type{search_string})')
                search_string = None
        update_flag = False
        if search_string != None:
            indices = [[index for index,line in enumerate(lines) if item in line]
                    for item in search_string]
            for i,index in enumerate(indices):
                if index:
                    if len(search_string) > 1 and key < search_string[i]:
                        lines.insert(index[0], newline)
                    else:
                        lines.insert(index[0]+1, newline)
                    update_flag = True
                    break
        if not update_flag:
            if isinstance(header, str):
                lines += ['', header, newline]
            else:
                lines += ['', newline]
    # Write updated config file
    with open(config_file, 'w') as f:
        for line in lines:
            f.write(f'{line}\n')
    # Return updated config
    config['key'] = value
    return config

def selectImageRange(path, filetype, name=None, num_required=None):
    if isinstance(name, str):
        name = f' {name} '
    else:
        name = ' '

    # Find available index range
    if filetype == 'tif':
        if not isinstance(path, str) and not os.path.isdir(path):
            illegal_value('path', path, 'selectImageRange')
            return (0, 0, 0)
        indexRegex = re.compile(r'\d+')
        # At this point only tiffs
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and
                f.endswith('.tif') and indexRegex.search(f)]
        num_files = len(files)
        if num_files < 1:
            logging.warning('No available'+name+'files')
            return (0, 0, 0)
        files.sort()
        first_index = indexRegex.search(files[0]).group()
        last_index = indexRegex.search(files[-1]).group()
        if first_index == None or last_index == None:
            logging.error('Unable to find correctly indexed'+name+'images')
            return (0, 0, 0)
        first_index = int(first_index)
        last_index = int(last_index)
        if num_files != last_index-first_index+1:
            logging.warning('Non-consecutive set of indices for'+name+'images')
    elif filetype == 'h5':
        if not isinstance(path, str) or not os.path.isfile(path):
            illegal_value('path', path, 'selectImageRange')
            return (0, 0, 0)
        # At this point only h5 in alamo2 detector style
        first_index = 0
        with h5py.File(path, 'r') as f:
            num_files = f['entry/instrument/detector/data'].shape[0]
            last_index = num_files-1
    else:
        illegal_value('filetype', filetype, 'selectImageRange')
        return (0, 0, 0)

    # Check range against requirements
    if num_files < 1:
        logging.warning('No available'+name+'files')
        return (0, 0, 0)
    if num_required == None:
        if num_files == 1:
            return (first_index, 0, 1)
    else:
        if not is_int(num_required, 1):
            illegal_value('num_required', num_required, 'selectImageRange')
            return (0, 0, 0)
        if num_files < num_required:
            logging.error('Unable to find the required'+name+
                    f'images ({num_files} out of {num_required})')
            return (0, 0, 0)

    # Select index range
    if num_required and last_index-first_index+1 < num_required:
        logging.error('Unable to find the required'+name+
                f'images ({last_index-first_index+1} out of {num_required})')
        return (0, 0, 0)
    print('\nNumber of available'+name+f'images: {num_files}')
    print('Index range of available'+name+f'images: [{first_index}, '+
            f'{last_index}]')
    if num_required == None:
        use_all = f'Use all ([{first_index}, {last_index}])'
        pick_offset = 'Pick a starting index offset and a number of images'
        pick_bounds = 'Pick the first and last index'
        menuchoice = pyip.inputMenu([use_all, pick_offset, pick_bounds], numbered=True)
        if menuchoice == use_all:
            return (first_index, 0, num_files)
        elif menuchoice == pick_offset:
            offset = pyip.inputInt('Enter the starting index offset'+
                    f' [0, {last_index-first_index}]: ', min=0,
                    max=last_index-first_index)
            first_index += offset
            if first_index == last_index:
                return (first_index, offset, 1)
            else:
                return (first_index, offset, pyip.inputInt('Enter the number of images'+
                        f' [1, {last_index-first_index+1}]: ', min=1,
                        max=last_index-first_index+1))
        else:
            offset = -first_index+pyip.inputInt('Enter the starting index '+
                    f'[{first_index}, {last_index}]: ', min=first_index, max=last_index)
            first_index += offset
            return (first_index, offset, pyip.inputInt('Enter the last index '+
                    f'[{first_index}, {last_index}]: ', min=first_index,
                    max=last_index)-first_index+1)
    else:
        use_all = f'Use ([{first_index}, {first_index+num_required-1}])'
        pick_offset = 'Pick a starting index offset'
        menuchoice = pyip.inputMenu([use_all, pick_offset], numbered=True)
        offset = 0
        if menuchoice == pick_offset:
            offset = pyip.inputInt('Enter the starting index offset'+
                    f'[0, {last_index-first_index-num_required+1}]: ',
                    min=0, max=last_index-first_index-num_required+1)
            first_index += offset
        return (first_index, offset, num_required)

def loadImage(f, img_x_bounds=None, img_y_bounds=None):
    """Load a single image from file."""
    if not os.path.isfile(f):
        logging.error(f'Unable to load {f}')
        return None
    img_read = img.imread(f)
    if not img_x_bounds:
        img_x_bounds = [0, img_read.shape[0]]
    else:
        if (not isinstance(img_x_bounds, list) or len(img_x_bounds) != 2 or 
                not (0 <= img_x_bounds[0] < img_x_bounds[1] <= img_read.shape[0])):
            logging.error(f'inconsistent row dimension in {f}')
            return None
    if not img_y_bounds:
        img_y_bounds = [0, img_read.shape[1]]
    else:
        if (not isinstance(img_y_bounds, list) or len(img_y_bounds) != 2 or 
                not (0 <= img_y_bounds[0] < img_y_bounds[1] <= img_read.shape[0])):
            logging.error(f'inconsistent column dimension in {f}')
            return None
    return img_read[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]

def loadImageStack(path, filetype, img_start, num_imgs, num_img_skip=0,
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
    img_stack = np.array([])
    if filetype == 'tif':
        if not isinstance(path, str) and not os.path.isdir(path):
            illegal_value('path', path, 'loadImageStack')
            return img_stack
        t0 = time()
        img_read_stack = []
        for i in range(0, num_read):
            f = f'{path}/nf_{img_range[i]:06d}.tif'
            if not (i+1)%20:
                logging.info(f'    loading {i+1}/{len(img_range)}: {f}')
            else:
                logging.debug(f'    loading {i+1}/{len(img_range)}: {f}')
            img_read = loadImage(f, img_x_bounds, img_y_bounds)
            img_read_stack.append(img_read)
        img_stack = np.stack([img_read for img_read in img_read_stack])
        logging.info(f'... done in {time()-t0:.2f} seconds!')
        logging.debug(f'img_stack shape = {np.shape(img_stack)}')
        del img_read_stack, img_read
    elif filetype == 'h5':
        if not isinstance(path, str) and not os.path.isfile(path):
            illegal_value('path', path, 'loadImageStack')
            return img_stack
        t0 = time()
        with h5py.File(path, 'r') as f:
            shape = f['entry/instrument/detector/data'].shape
            if len(shape) != 3:
                logging.error(f'inconsistent dimensions in {path}')
            if not img_x_bounds:
                img_x_bounds = [0, shape[1]]
            else:
                if (not isinstance(img_x_bounds, list) or len(img_x_bounds) != 2 or 
                        not (0 <= img_x_bounds[0] < img_x_bounds[1] <= shape[1])):
                    logging.error(f'inconsistent row dimension in {path}')
            if not img_y_bounds:
                img_y_bounds = [0, shape[2]]
            else:
                if (not isinstance(img_y_bounds, list) or len(img_y_bounds) != 2 or 
                        not (0 <= img_y_bounds[0] < img_y_bounds[1] <= shape[2])):
                    logging.error(f'inconsistent column dimension in {path}')
            img_stack = f.get('entry/instrument/detector/data')[
                    img_start:img_start+num_imgs:num_img_skip+1,
                    img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]
        logging.info(f'... done in {time()-t0:.2f} seconds!')
    else:
        illegal_value('filetype', filetype, 'selectImageRange')
    return img_stack

def clearFig(title):
    if not isinstance(title, str):
        illegal_value('title', title, 'clearFig')
        return
    plt.close(fig=re.sub(r"\s+", '_', title))

def quickImshow(a, title=None, path='.', save_fig=False, save_only=False, clear=True, **kwargs):
    if title != None and not isinstance(title, str):
        illegal_value('title', title, 'quickImshow')
        return
    if not isinstance(path, str):
        illegal_value('path', path, 'quickImshow')
        return
    if not isinstance(save_fig, bool):
        illegal_value('save_fig', save_fig, 'quickImshow')
        return
    if not isinstance(save_only, bool):
        illegal_value('save_only', save_only, 'quickImshow')
        return
    if not isinstance(clear, bool):
        illegal_value('clear', clear, 'quickImshow')
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
        plt.savefig(f'{path}/{title}.png')
        plt.close(fig=title)
        #plt.imsave(f'{title}.png', a, **kwargs)
    else:
        plt.ion()
        plt.figure(title)
        plt.imshow(a, **kwargs)
        if save_fig:
            plt.savefig(f'{path}/{title}.png')
        plt.pause(1)

def quickPlot(*args, title=None, path='.', save_fig=False, save_only=False, clear=True, **kwargs):
    if title != None and not isinstance(title, str):
        illegal_value('title', title, 'quickPlot')
        return
    if not isinstance(path, str):
        illegal_value('path', path, 'quickPlot')
        return
    if not isinstance(save_fig, bool):
        illegal_value('save_fig', save_fig, 'quickPlot')
        return
    if not isinstance(save_only, bool):
        illegal_value('save_only', save_only, 'quickPlot')
        return
    if not isinstance(clear, bool):
        illegal_value('clear', clear, 'quickPlot')
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
        plt.savefig(f'{path}/{title}.png')
        plt.close(fig=title)
    else:
        plt.ion()
        plt.figure(title)
        if depth_tuple(args) > 1:
           for y in args:
               plt.plot(*y, **kwargs)
        else:
            plt.plot(*args, **kwargs)
        if save_fig:
            plt.savefig(f'{path}/{title}.png')
        plt.pause(1)

def selectImageBounds(a, axis, low=None, upp=None, num_min=None,
        title='select array bounds'):
    """Interactively select the lower and upper data bounds for a 2D numpy array."""
    if not isinstance(a, np.ndarray) or a.ndim != 2:
        logging.error('Illegal array type or dimension in selectImageBounds')
        return None
    if axis < 0 or axis >= a.ndim:
        illegal_value('axis', axis, 'selectImageBounds')
        return None
    if num_min == None:
        num_min = 1
    else:
        if num_min < 2 or num_min > a.shape[axis]:
            logging.warning('Illegal input for num_min in selectImageBounds, input ignored')
            num_min = 1
    if low == None:
        min_ = 0
        max_ = a.shape[axis]
        low_max = a.shape[axis]-num_min
        while True:
            if axis:
                quickImshow(a[:,min_:max_], title=title, aspect='auto',
                        extent=[min_,max_,a.shape[0],0])
            else:
                quickImshow(a[min_:max_,:], title=title, aspect='auto',
                        extent=[0,a.shape[1], max_,min_])
            zoom_flag = pyip.inputInt('Set lower data bound (0) or zoom in (1)?: ',
                    min=0, max=1)
            if zoom_flag:
                min_ = pyip.inputInt(f'    Set lower zoom index [0, {low_max}]: ',
                        min=0, max=low_max)
                max_ = pyip.inputInt(f'    Set upper zoom index [{min_+1}, {low_max+1}]: ',
                        min=min_+1, max=low_max+1)
            else:
                low = pyip.inputInt(f'    Set lower data bound [0, {low_max}]: ',
                        min=0, max=low_max)
                break
    else:
        if not is_int(low, 0, a.shape[axis]-num_min):
            illegal_value('low', low, 'selectImageBounds')
            return None
    if upp == None:
        min_ = low+num_min
        max_ = a.shape[axis]
        upp_min = min_
        while True:
            if axis:
                quickImshow(a[:,min_:max_], title=title, aspect='auto',
                        extent=[min_,max_,a.shape[0],0])
            else:
                quickImshow(a[min_:max_,:], title=title, aspect='auto',
                        extent=[0,a.shape[1], max_,min_])
            zoom_flag = pyip.inputInt('Set upper data bound (0) or zoom in (1)?: ',
                    min=0, max=1)
            if zoom_flag:
                min_ = pyip.inputInt(f'    Set upper zoom index [{upp_min}, {a.shape[axis]-1}]: ',
                        min=upp_min, max=a.shape[axis]-1)
                max_ = pyip.inputInt(f'    Set upper zoom index [{min_+1}, {a.shape[axis]}]: ',
                        min=min_+1, max=a.shape[axis])
            else:
                upp = pyip.inputInt(f'    Set upper data bound [{upp_min}, {a.shape[axis]}]: ',
                        min=upp_min, max=a.shape[axis])
                break
    else:
        if not is_int(upp, low+num_min, a.shape[axis]):
            illegal_value('upp', upp, 'selectImageBounds')
            return None
    print(f'lower bound = {low} (inclusive)\nupper bound = {upp} (exclusive)')
    bounds = [low, upp]
    a_tmp = a
    if axis:
        a_tmp[:,bounds[0]] = a.max()
        a_tmp[:,bounds[1]] = a.max()
    else:
        a_tmp[bounds[0],:] = a.max()
        a_tmp[bounds[1],:] = a.max()
    quickImshow(a_tmp, title=title)
    if pyip.inputYesNo('Accept these bounds ([y]/n)?: ', blank=True) == 'no':
        bounds = selectImageBounds(a, title=title)
    return bounds

def selectArrayBounds(a, x_low=None, x_upp=None, num_x_min=None,
        title='select array bounds'):
    """Interactively select the lower and upper data bounds for a numpy array."""
    if not isinstance(a, np.ndarray) or a.ndim != 1:
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
        if not is_int(x_low, 0, a.size-num_x_min):
            illegal_value('x_low', x_low, 'selectArrayBounds')
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
        if not is_int(x_upp, x_low+num_x_min, a.size):
            illegal_value('x_upp', x_upp, 'selectArrayBounds')
            return None
    print(f'lower bound = {x_low} (inclusive)\nupper bound = {x_upp} (exclusive)]')
    bounds = [x_low, x_upp]
    #quickPlot(range(bounds[0], bounds[1]), a[bounds[0]:bounds[1]], title=title)
    quickPlot((range(a.size), a), ([bounds[0], bounds[0]], [a.min(), a.max()], 'r-'),
            ([bounds[1], bounds[1]], [a.min(), a.max()], 'r-'), title=title)
    if pyip.inputYesNo('Accept these bounds ([y]/n)?: ', blank=True) == 'no':
        bounds = selectArrayBounds(a, title=title)
    return bounds

def fitStep(x=None, y=None, model='step', form='arctan'):
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        logging.error('Illegal array type or dimension for y in fitStep')
        return
    if isinstance(x, type(None)):
        x = np.array(range(y.size))
    elif not isinstance(x, np.ndarray) or x.ndim != 1 or x.size != y.size:
        logging.error('Illegal array type or dimension for x in fitStep')
        return
    if not isinstance(model, str) or not model in ('step', 'rectangle'):
        illegal_value('model', model, 'fitStepModel')
        return
    if not isinstance(form, str) or not form in ('linear', 'atan', 'arctan', 'erf', 'logistic'):
        illegal_value('form', form, 'fitStepModel')
        return

    if model == 'step':
        mod = StepModel(form=form)
    else:
        mod = RectangleModel(form=form)
    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)
    #print(out.fit_report())
    #quickPlot((x,y),(x,out.best_fit))
    return out.best_values


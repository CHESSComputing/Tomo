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
import yaml
import h5py
try:
    import pyinputplus as pyip
except:
    pass
import numpy as np
import imageio as img
import matplotlib.pyplot as plt
from time import time
from ast import literal_eval
try:
    from lmfit.models import StepModel, RectangleModel
except:
    pass

def depth_list(L): return isinstance(L, list) and max(map(depth_list, L))+1
def depth_tuple(T): return isinstance(T, tuple) and max(map(depth_tuple, T))+1

def is_int(v, v_min=None, v_max=None):
    """Value is an integer in range v_min <= v <= v_max.
    """
    if not isinstance(v, int):
        return False
    if (v_min != None and v < v_min) or (v_max != None and v > v_max):
        return False
    return True

def is_num(v, v_min=None, v_max=None):
    """Value is a number in range v_min <= v <= v_max.
    """
    if not isinstance(v, (int,float)):
        return False
    if (v_min != None and v < v_min) or (v_max != None and v > v_max):
        return False
    return True

def is_index(v, v_min=0, v_max=None):
    """Value is an array index in range v_min <= v < v_max.
    """
    if not isinstance(v, int):
        return False
    if v < v_min or (v_max != None and v >= v_max):
        return False
    return True

def is_index_range(v, v_min=0, v_max=None):
    """Value is an array index range in range v_min <= v[0] <= v[1] < v_max.
    """
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

def findImageFiles(path, filetype, name=None):
    if isinstance(name, str):
        name = f' {name} '
    else:
        name = ' '
    # Find available index range
    if filetype == 'tif':
        if not isinstance(path, str) and not os.path.isdir(path):
            illegal_value('path', path, 'findImageRange')
            return -1, 0, []
        indexRegex = re.compile(r'\d+')
        # At this point only tiffs
        files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and
                f.endswith('.tif') and indexRegex.search(f)])
        num_imgs = len(files)
        if num_imgs < 1:
            logging.warning('No available'+name+'files')
            return -1, 0, []
        first_index = indexRegex.search(files[0]).group()
        last_index = indexRegex.search(files[-1]).group()
        if first_index == None or last_index == None:
            logging.error('Unable to find correctly indexed'+name+'images')
            return -1, 0, []
        first_index = int(first_index)
        last_index = int(last_index)
        if num_imgs != last_index-first_index+1:
            logging.error('Non-consecutive set of indices for'+name+'images')
            return -1, 0, []
        paths = [os.path.join(path, f) for f in files]
    elif filetype == 'h5':
        if not isinstance(path, str) or not os.path.isfile(path):
            illegal_value('path', path, 'findImageRange')
            return -1, 0, []
        # At this point only h5 in alamo2 detector style
        first_index = 0
        with h5py.File(path, 'r') as f:
            num_imgs = f['entry/instrument/detector/data'].shape[0]
            last_index = num_imgs-1
        paths = [path]
    else:
        illegal_value('filetype', filetype, 'findImageRange')
        return -1, 0, []
    logging.debug('\nNumber of available'+name+f'images: {num_imgs}')
    logging.debug('Index range of available'+name+f'images: [{first_index}, '+
            f'{last_index}]')

    return first_index, num_imgs, paths

def selectImageRange(first_index, offset, num_imgs, name=None, num_required=None):
    if isinstance(name, str):
        name = f' {name} '
    else:
        name = ' '
    # Check existing values
    use_input = 'no'
    if (is_int(first_index, 0) and is_int(offset, 0) and is_int(num_imgs, 1)):
        if offset < 0:
            use_input = pyip.inputYesNo('\nCurrent'+name+f'first index = {first_index}, '+
                    'use this value ([y]/n)? ', blank=True)
        else:
            use_input = pyip.inputYesNo('\nCurrent'+name+'first index/offset = '+
                    f'{first_index}/{offset}, use these values ([y]/n)? ',
                    blank=True)
        if use_input != 'no':
            use_input = pyip.inputYesNo('Current number of'+name+'images = '+
                    f'{num_imgs}, use this value ([y]/n)? ',
                    blank=True)
    if use_input == 'yes':
        return first_index, offset, num_imgs

    # Check range against requirements
    if num_imgs < 1:
        logging.warning('No available'+name+'images')
        return -1, -1, 0
    if num_required == None:
        if num_imgs == 1:
            return first_index, 0, 1
    else:
        if not is_int(num_required, 1):
            illegal_value('num_required', num_required, 'selectImageRange')
            return -1, -1, 0
        if num_imgs < num_required:
            logging.error('Unable to find the required'+name+
                    f'images ({num_imgs} out of {num_required})')
            return -1, -1, 0

    # Select index range
    if num_required == None:
        last_index = first_index+num_imgs
        use_all = f'Use all ([{first_index}, {last_index}])'
        pick_offset = 'Pick a first index offset and a number of images'
        pick_bounds = 'Pick the first and last index'
        menuchoice = pyip.inputMenu([use_all, pick_offset, pick_bounds], numbered=True)
        if menuchoice == use_all:
            offset = 0
        elif menuchoice == pick_offset:
            offset = pyip.inputInt('Enter the first index offset'+
                    f' [0, {last_index-first_index}]: ', min=0, max=last_index-first_index)
            first_index += offset
            if first_index == last_index:
                num_imgs = 1
            else:
                num_imgs = pyip.inputInt(f'Enter the number of images [1, {num_imgs-offset}]: ',
                        min=1, max=num_imgs-offset)
        else:
            offset = pyip.inputInt(f'Enter the first index [{first_index}, {last_index}]: ',
                    min=first_index, max=last_index)-first_index
            first_index += offset
            num_imgs = pyip.inputInt(f'Enter the last index [{first_index}, {last_index}]: ',
                    min=first_index, max=last_index)-first_index+1
    else:
        use_all = f'Use ([{first_index}, {first_index+num_required-1}])'
        pick_offset = 'Pick the first index offset'
        menuchoice = pyip.inputMenu([use_all, pick_offset], numbered=True)
        offset = 0
        if menuchoice == pick_offset:
            offset = pyip.inputInt('Enter the first index offset'+
                    f'[0, {num_imgs-num_required}]: ', min=0, max=num_imgs-num_required)
            first_index += offset
        num_imgs = num_required

    return first_index, offset, num_imgs

def loadImage(f, img_x_bounds=None, img_y_bounds=None):
    """Load a single image from file.
    """
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

def loadImageStack(files, filetype, img_offset, num_imgs, num_img_skip=0,
        img_x_bounds=None, img_y_bounds=None):
    """Load a set of images and return them as a stack.
    """
    logging.debug(f'img_offset = {img_offset}')
    logging.debug(f'num_imgs = {num_imgs}')
    logging.debug(f'num_img_skip = {num_img_skip}')
    logging.debug(f'\nfiles:\n{files}\n')
    img_stack = np.array([])
    if filetype == 'tif':
        img_read_stack = []
        i = 1
        t0 = time()
        for f in files[img_offset:img_offset+num_imgs:num_img_skip+1]:
            if not i%20:
                logging.info(f'    loading {i}/{num_imgs}: {f}')
            else:
                logging.debug(f'    loading {i}/{num_imgs}: {f}')
            img_read = loadImage(f, img_x_bounds, img_y_bounds)
            img_read_stack.append(img_read)
            i += num_img_skip+1
        img_stack = np.stack([img_read for img_read in img_read_stack])
        logging.info(f'... done in {time()-t0:.2f} seconds!')
        logging.debug(f'img_stack shape = {np.shape(img_stack)}')
        del img_read_stack, img_read
    elif filetype == 'h5':
        if not isinstance(files[0], str) and not os.path.isfile(files[0]):
            illegal_value('files[0]', files[0], 'loadImageStack')
            return img_stack
        t0 = time()
        with h5py.File(files[0], 'r') as f:
            shape = f['entry/instrument/detector/data'].shape
            if len(shape) != 3:
                logging.error(f'inconsistent dimensions in {files[0]}')
            if not img_x_bounds:
                img_x_bounds = [0, shape[1]]
            else:
                if (not isinstance(img_x_bounds, list) or len(img_x_bounds) != 2 or 
                        not (0 <= img_x_bounds[0] < img_x_bounds[1] <= shape[1])):
                    logging.error(f'inconsistent row dimension in {files[0]}')
            if not img_y_bounds:
                img_y_bounds = [0, shape[2]]
            else:
                if (not isinstance(img_y_bounds, list) or len(img_y_bounds) != 2 or 
                        not (0 <= img_y_bounds[0] < img_y_bounds[1] <= shape[2])):
                    logging.error(f'inconsistent column dimension in {files[0]}')
            img_stack = f.get('entry/instrument/detector/data')[
                    img_offset:img_offset+num_imgs:num_img_skip+1,
                    img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]
        logging.info(f'... done in {time()-t0:.2f} seconds!')
    else:
        illegal_value('filetype', filetype, 'findImageRange')
    return img_stack

def clearFig(title):
    if not isinstance(title, str):
        illegal_value('title', title, 'clearFig')
        return
    plt.close(fig=re.sub(r"\s+", '_', title))

def quickImshow(a, title=None, path=None, name=None, save_fig=False, save_only=False,
            clear=True, **kwargs):
    if title != None and not isinstance(title, str):
        illegal_value('title', title, 'quickImshow')
        return
    if path is not None and not isinstance(path, str):
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
    if name is None:
        if path is None:
            path = f'{title}.png'
        else:
            path = f'{path}/{title}.png'
    else:
        if path is None:
            path = name
        else:
            path = f'{path}/{name}'
    if clear:
        plt.close(fig=title)
    if save_only:
        plt.figure(title)
        plt.imshow(a, **kwargs)
        plt.savefig(path)
        plt.close(fig=title)
        #plt.imsave(f'{title}.png', a, **kwargs)
    else:
        plt.ion()
        plt.figure(title)
        plt.imshow(a, **kwargs)
        if save_fig:
            plt.savefig(path)
        plt.pause(1)

def quickPlot(*args, title=None, path=None, name=None, save_fig=False, save_only=False,
        clear=True, **kwargs):
    if title != None and not isinstance(title, str):
        illegal_value('title', title, 'quickPlot')
        return
    if path is not None and not isinstance(path, str):
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
    if name is None:
        if path is None:
            path = f'{title}.png'
        else:
            path = f'{path}/{title}.png'
    else:
        if path is None:
            path = name
        else:
            path = f'{path}/{name}'
    if clear:
        plt.close(fig=title)
    if save_only:
        plt.figure(title)
        if depth_tuple(args) > 1:
           for y in args:
               plt.plot(*y, **kwargs)
        else:
            plt.plot(*args, **kwargs)
        plt.savefig(path)
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
            plt.savefig(path)
        plt.pause(1)

def selectArrayBounds(a, x_low=None, x_upp=None, num_x_min=None,
        title='select array bounds'):
    """Interactively select the lower and upper data bounds for a numpy array.
    """
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
    bounds = [x_low, x_upp]
    print(f'lower bound = {x_low} (inclusive)\nupper bound = {x_upp} (exclusive)]')
    #quickPlot(range(bounds[0], bounds[1]), a[bounds[0]:bounds[1]], title=title)
    quickPlot((range(a.size), a), ([bounds[0], bounds[0]], [a.min(), a.max()], 'r-'),
            ([bounds[1], bounds[1]], [a.min(), a.max()], 'r-'), title=title)
    if pyip.inputYesNo('Accept these bounds ([y]/n)?: ', blank=True) == 'no':
        bounds = selectArrayBounds(a, title=title)
    return bounds

def selectImageBounds(a, axis, low=None, upp=None, num_min=None,
        title='select array bounds'):
    """Interactively select the lower and upper data bounds for a 2D numpy array.
    """
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
    bounds = [low, upp]
    a_tmp = a
    if axis:
        a_tmp[:,bounds[0]] = a.max()
        a_tmp[:,bounds[1]] = a.max()
    else:
        a_tmp[bounds[0],:] = a.max()
        a_tmp[bounds[1],:] = a.max()
    print(f'lower bound = {low} (inclusive)\nupper bound = {upp} (exclusive)')
    quickImshow(a_tmp, title=title)
    if pyip.inputYesNo('Accept these bounds ([y]/n)?: ', blank=True) == 'no':
        bounds = selectImageBounds(a, title=title)
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

class Config:
    """Base class for processing a config file or dictionary.
    """
    def __init__(self, config_file=None, config_dict=None):
        self.config = {}
        self.load_flag = False
        self.suffix = None

        # Load config file 
        if config_file is not None and config_dict is not None:
            logging.warning('Ignoring config_dict (both config_file and config_dict are specified)')
        if config_file is not None:
           self.loadFile(config_file)
        elif config_dict is not None:
           self.loadDict(config_dict)

    def loadFile(self, config_file):
        """Load a config file.
        """
        if self.load_flag:
            logging.warning('Overwriting any previously loaded config file')
        self.config = {}

        # Ensure config file exists
        if not os.path.isfile(config_file):
            logging.error(f'Unable to load {config_file}')
            return

        # Load config file
        self.suffix = os.path.splitext(config_file)[1]
        if self.suffix == '.yml' or self.suffix == '.yaml':
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        elif self.suffix == '.txt':
            with open(config_file, 'r') as f:
                lines = f.read().splitlines()
            self.config = {item[0].strip():literal_eval(item[1].strip()) for item in
                    [line.split('#')[0].split('=') for line in lines if '=' in line.split('#')[0]]}
        else:
            logging.error(f'Illegal config file extension: {self.suffix}')

        # Make sure config file was correctly loaded
        if isinstance(self.config, dict):
            self.load_flag = True
        else:
            logging.error(f'Unable to load dictionary from config file: {config_file}')
            self.config = {}

    def loadDict(self, config_dict):
        """Takes a dictionary and places it into self.config.
        """
        exit('loadDict not tested yet, what format do we follow: txt or yaml?')
        if self.load_flag:
            logging.warning('Overwriting the previously loaded config file')

        if isinstance(config_dict, dict):
            self.config = config_dict
            self.load_flag = True
        else:
            logging.error(f'Illegal dictionary config object: {config_dict}')
            self.config = {}

    def saveFile(self, config_file):
        """Save the config file (as a yaml file only right now).
        """
        suffix = os.path.splitext(config_file)[1]
        if suffix != '.yml' and suffix != '.yaml':
            logging.error(f'Illegal config file extension: {suffix}')

        # Check if config file exists
        if os.path.isfile(config_file):
            logging.info(f'Updating {config_file}')
        else:
            logging.info(f'Saving {config_file}')

        # Save config file
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f)

    def validate(self, pars_required, pars_missing=None):
        """Returns False if any required first level keys are missing.
        """
        if not self.load_flag:
            logging.error('Load a config file prior to calling Config.validate')
        pars = [p for p in pars_required if p not in self.config]
        if isinstance(pars_missing, list):
            pars_missing.extend(pars)
        elif pars_missing is not None:
            illegal_value('pars_missing', pars_missing, 'Config.validate')
        if len(pars) > 0:
            return False
        return True

#RV FIX this is for a txt file, obsolete?
#    def update_txt(self, config_file, key, value, search_string=None, header=None):
#        if not self.load_flag:
#            logging.error('Load a config file prior to calling Config.update')
#
#        if not os.path.isfile(config_file):
#            logging.error(f'Unable to load {config_file}')
#            lines = []
#        else:
#            with open(config_file, 'r') as f:
#                lines = f.read().splitlines()
#        config = {item[0].strip():literal_eval(item[1].strip()) for item in
#                [line.split('#')[0].split('=') for line in lines if '=' in line.split('#')[0]]}
#        if not isinstance(key, str):
#            illegal_value('key', key, 'Config.update')
#            return config
#        if isinstance(value, str):
#            newline = f"{key} = '{value}'"
#        else:
#            newline = f'{key} = {value}'
#        if key in config.keys():
#            # Update key with value
#            for index,line in enumerate(lines):
#                if '=' in line:
#                    item = line.split('#')[0].split('=')
#                    if item[0].strip() == key:
#                        lines[index] = newline
#                        break
#        else:
#            # Insert new key/value pair
#            if search_string != None:
#                if isinstance(search_string, str):
#                    search_string = [search_string]
#                elif not isinstance(search_string, (tuple, list)):
#                    illegal_value('search_string', search_string, 'Config.update')
#                    search_string = None
#            update_flag = False
#            if search_string != None:
#                indices = [[index for index,line in enumerate(lines) if item in line]
#                        for item in search_string]
#                for i,index in enumerate(indices):
#                    if index:
#                        if len(search_string) > 1 and key < search_string[i]:
#                            lines.insert(index[0], newline)
#                        else:
#                            lines.insert(index[0]+1, newline)
#                        update_flag = True
#                        break
#            if not update_flag:
#                if isinstance(header, str):
#                    lines += ['', header, newline]
#                else:
#                    lines += ['', newline]
#        # Write updated config file
#        with open(config_file, 'w') as f:
#            for line in lines:
#                f.write(f'{line}\n')
#        # Update loaded config
#        config['key'] = value
#    
#RV update and bring into Config if needed again
#def search(config_file, search_string):
#    if not os.path.isfile(config_file):
#        logging.error(f'Unable to load {config_file}')
#        return False
#    with open(config_file, 'r') as f:
#        lines = f.read()
#        if search_string in lines:
#            return True
#    return False

class Detector:
    """Class for processing a detector info file or dictionary.
    """
    def __init__(self, detector_id):
        self.detector = {}
        self.load_flag = False
        self.validate_flag = False

        # Load detector file 
        self.loadFile(detector_id)

    def loadFile(self, detector_id):
        """Load a detector file.
        """
        if self.load_flag:
            logging.warning('Overwriting the previously loaded detector file')
        self.detector = {}

        # Ensure detector file exists
        if not isinstance(detector_id, str):
            illegal_value('detector_id', detector_id, 'Detector.loadFile')
            return
        detector_file = f'{detector_id}.yaml'
        if not os.path.isfile(detector_file):
            detector_file = self.config['detector_id']+'.yaml'
            if not os.path.isfile(detector_file):
                logging.error(f'Unable to load detector info file for {detector_id}')
                return

        # Load detector file
        with open(detector_file, 'r') as f:
            self.detector = yaml.safe_load(f)

        # Make sure detector file was correctly loaded
        if isinstance(self.detector, dict):
            self.load_flag = True
        else:
            logging.error(f'Unable to load dictionary from detector file: {detector_file}')
            self.detector = {}

    def validate(self):
        """Returns False if any config parameters is illegal or missing.
        """
        if not self.load_flag:
            logging.error('Load a detector file prior to calling Detector.validate')

        # Check for required first-level keys
        pars_required = ['detector', 'lens_magnification']
        pars_missing = [p for p in pars_required if p not in self.detector]
        if len(pars_missing) > 0:
            logging.error(f'Missing item(s) in detector file: {", ".join(pars_missing)}')
            return False

        is_valid = True

        # Check detector pixel config keys
        pixels = self.detector['detector'].get('pixels')
        if not pixels:
            pars_missing.append('detector:pixels')
        else:
            rows = pixels.get('rows')
            if not rows:
                pars_missing.append('detector:pixels:rows')
            columns = pixels.get('columns')
            if not columns:
                pars_missing.append('detector:pixels:columns')
            size = pixels.get('size')
            if not size:
                pars_missing.append('detector:pixels:size')

        if not len(pars_missing):
            self.validate_flag = True
        else:
            is_valid = False

        return is_valid

    def getPixelSize(self):
        """Returns the detector pixel size.
        """
        if not self.validate_flag:
            logging.error('Validate detector file info prior to calling Detector.getPixelSize')

        lens_magnification = self.detector.get('lens_magnification')
        if not isinstance(lens_magnification, (int,float)) or lens_magnification <= 0.:
            illegal_value('lens_magnification', lens_magnification, 'detector file')
            return 0
        pixel_size = self.detector['detector'].get('pixels').get('size')
        if isinstance(pixel_size, (int,float)):
            if pixel_size <= 0.:
                illegal_value('pixel_size', pixel_size, 'detector file')
                return 0
            pixel_size /= lens_magnification
        elif isinstance(pixel_size, list):
            if ((len(pixel_size) > 2) or
                    (len(pixel_size) == 2 and pixel_size[0] != pixel_size[1])):
                illegal_value('pixel size', pixel_size, 'detector file')
                return 0
            elif not is_num(pixel_size[0], 0.):
                illegal_value('pixel size', pixel_size, 'detector file')
                return 0
            else:
                pixel_size = pixel_size[0]/lens_magnification
        else:
            illegal_value('pixel size', pixel_size, 'detector file')
            return 0

        return pixel_size

    def getDimensions(self):
        """Returns the detector pixel dimensions.
        """
        if not self.validate_flag:
            logging.error('Validate detector file info prior to calling Detector.getDimensions')

        pixels = self.detector['detector'].get('pixels')
        num_rows = pixels.get('rows')
        if not is_int(num_rows, 1):
            illegal_value('rows', num_rows, 'detector file')
            return (0, 0)
        num_columns = pixels.get('columns')
        if not is_int(num_columns, 1):
            illegal_value('columns', num_columns, 'detector file')
            return (0, 0)

        return num_rows, num_columns

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
try:
    import h5py
except:
    pass
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass
try:
    import pyinputplus as pyip
except:
    pass

from time import time
from ast import literal_eval
from copy import deepcopy
try:
    #from lmfit import Minimizer
    from lmfit import Model, Parameters, minimize, fit_report
except:
    pass
try:
    from lmfit.models import ConstantModel, LinearModel, QuadraticModel, PolynomialModel,\
            StepModel, RectangleModel, GaussianModel, LorentzianModel
except:
    pass

def depth_list(L): return isinstance(L, list) and max(map(depth_list, L))+1
def depth_tuple(T): return isinstance(T, tuple) and max(map(depth_tuple, T))+1
def unwrap_tuple(T):
    if depth_tuple(T) > 1 and len(T) == 1:
        T = unwrap_tuple(*T)
    return T
   
def illegal_value(value, name, location=None, exit_flag=False):
    if not isinstance(location, str):
        location = ''
    else:
        location = f'in {location} '
    if isinstance(name, str):
        logging.error(f'Illegal value for {name} {location}({value}, {type(value)})')
    else:
        logging.error(f'Illegal value {location}({value}, {type(value)})')
    if exit_flag:
        raise ValueError

def is_int(v, v_min=None, v_max=None):
    """Value is an integer in range v_min <= v <= v_max.
    """
    if not isinstance(v, int):
        return False
    if v_min is not None and not isinstance(v_min, int):
        illegal_value(v_min, 'v_min', 'is_int') 
        return False
    if v_max is not None and not isinstance(v_max, int):
        illegal_value(v_max, 'v_max', 'is_int') 
        return False
    if v_min is not None and v_max is not None and v_min > v_max:
        logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
        return False
    if (v_min is not None and v < v_min) or (v_max is not None and v > v_max):
        return False
    return True

def is_int_pair(v, v_min=None, v_max=None):
    """Value is an integer pair, each in range v_min <= v[i] <= v_max or 
           v_min[i] <= v[i] <= v_max[i].
    """
    if not (isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[0], int) and
            isinstance(v[1], int)):
        return False
    if v_min is not None or v_max is not None:
        if (v_min is None or isinstance(v_min, int)) and (v_max is None or isinstance(v_max, int)):
            if True in [True if not is_int(vi, v_min=v_min, v_max=v_max) else False for vi in v]:
                return False
        elif is_int_pair(v_min) and is_int_pair(v_max):
            if True in [True if v_min[i] > v_max[i] else False for i in range(2)]:
                logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
                return False
            if True in [True if not is_int(v[i], v_min[i], v_max[i]) else False for i in range(2)]:
                return False
        elif is_int_pair(v_min) and (v_max is None or isinstance(v_max, int)):
            if True in [True if not is_int(v[i], v_min=v_min[i], v_max=v_max) else False
                    for i in range(2)]:
                return False
        elif (v_min is None or isinstance(v_min, int)) and is_int_pair(v_max):
            if True in [True if not is_int(v[i], v_min=v_min, v_max=v_max[i]) else False
                    for i in range(2)]:
                return False
        else:
            logging.error(f'Illegal v_min or v_max input ({v_min} {type(v_min)} and '+
                    f'{v_max} {type(v_max)})')
            return False
    return True

def is_int_series(l, v_min=None, v_max=None):
    """Value is a tuple or list of integers, each in range v_min <= l[i] <= v_max.
    """
    if v_min is not None and not isinstance(v_min, int):
        illegal_value(v_min, 'v_min', 'is_int_series') 
        return False
    if v_max is not None and not isinstance(v_max, int):
        illegal_value(v_max, 'v_max', 'is_int_series') 
        return False
    if not isinstance(l, (tuple, list)):
        return False
    if True in [True if not is_int(v, v_min=v_min, v_max=v_max) else False for v in l]:
        return False
    return True

def is_num(v, v_min=None, v_max=None):
    """Value is a number in range v_min <= v <= v_max.
    """
    if not isinstance(v, (int, float)):
        return False
    if v_min is not None and not isinstance(v_min, (int, float)):
        illegal_value(v_min, 'v_min', 'is_num') 
        return False
    if v_max is not None and not isinstance(v_max, (int, float)):
        illegal_value(v_max, 'v_max', 'is_num') 
        return False
    if v_min is not None and v_max is not None and v_min > v_max:
        logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
        return False
    if (v_min is not None and v < v_min) or (v_max is not None and v > v_max):
        return False
    return True

def is_num_pair(v, v_min=None, v_max=None):
    """Value is a number pair, each in range v_min <= v[i] <= v_max or 
           v_min[i] <= v[i] <= v_max[i].
    """
    if not (isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[0], (int, float)) and
            isinstance(v[1], (int, float))):
        return False
    if v_min is not None or v_max is not None:
        if ((v_min is None or isinstance(v_min, (int, float))) and
                (v_max is None or isinstance(v_max, (int, float)))):
            if True in [True if not is_num(vi, v_min=v_min, v_max=v_max) else False for vi in v]:
                return False
        elif is_num_pair(v_min) and is_num_pair(v_max):
            if True in [True if v_min[i] > v_max[i] else False for i in range(2)]:
                logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
                return False
            if True in [True if not is_num(v[i], v_min[i], v_max[i]) else False for i in range(2)]:
                return False
        elif is_num_pair(v_min) and (v_max is None or isinstance(v_max, (int, float))):
            if True in [True if not is_num(v[i], v_min=v_min[i], v_max=v_max) else False
                    for i in range(2)]:
                return False
        elif (v_min is None or isinstance(v_min, (int, float))) and is_num_pair(v_max):
            if True in [True if not is_num(v[i], v_min=v_min, v_max=v_max[i]) else False
                    for i in range(2)]:
                return False
        else:
            logging.error(f'Illegal v_min or v_max input ({v_min} {type(v_min)} and '+
                    f'{v_max} {type(v_max)})')
            return False
    return True

def is_num_series(l, v_min=None, v_max=None):
    """Value is a tuple or list of numbers, each in range v_min <= l[i] <= v_max.
    """
    if v_min is not None and not isinstance(v_min, (int, float)):
        illegal_value(v_min, 'v_min', 'is_num_series') 
        return False
    if v_max is not None and not isinstance(v_max, (int, float)):
        illegal_value(v_max, 'v_max', 'is_num_series') 
        return False
    if not isinstance(l, (tuple, list)):
        return False
    if True in [True if not is_num(v, v_min=v_min, v_max=v_max) else False for v in l]:
        return False
    return True

def is_index(v, v_min=0, v_max=None):
    """Value is an array index in range v_min <= v < v_max.
       NOTE v_max IS NOT included!
    """
    if isinstance(v_max, int):
        if v_max <= v_min:
            logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
            return False
        v_max -= 1
    return is_int(v, v_min, v_max)

def is_index_range(v, v_min=0, v_max=None):
    """Value is an array index range in range v_min <= v[0] <= v[1] <= v_max.
       NOTE v_max IS included!
    """
    if not is_int_pair(v):
        return False
    if not isinstance(v_min, int):
        illegal_value(v_min, 'v_min', 'is_index_range') 
        return False
    if v_max is not None:
        if not isinstance(v_max, int):
            illegal_value(v_max, 'v_max', 'is_index_range') 
            return False
        if v_max < v_min:
            logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
            return False
    if not v_min <= v[0] <= v[1] or (v_max is not None and v[1] > v_max):
        return False
    return True

def index_nearest(a, value):
    return (int)(np.argmin(np.abs(np.asarray(a)-value)))

def round_to_n(x, n=1):
    if x == 0.0:
        return 0
    else:
        return round(x, n-1-int(np.floor(np.log10(abs(x)))))

def round_up_to_n(x, n=1):
    xr = round_to_n(x, n)
    if abs(x/xr) > 1.0:
        xr += np.sign(x)*10**(np.floor(np.log10(abs(x)))+1-n)
    return xr

def trunc_to_n(x, n=1):
    xr = round_to_n(x, n)
    if abs(xr/x) > 1.0:
        xr -= np.sign(x)*10**(np.floor(np.log10(abs(x)))+1-n)
    return xr

def string_to_list(s):
    """Return a list of numbers by splitting/expanding a string on any combination of
       dashes, commas, and/or whitespaces
       e.g: '1, 3, 5-8,12 ' -> [1, 3, 5, 6, 7, 8, 12]
    """
    if not isinstance(s, str):
        illegal_value(s, location='string_to_list') 
        return None
    if not len(s):
        return []
    try:
        list1 = [x for x in re.split('\s+,\s+|\s+,|,\s+|\s+|,', s.strip())]
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    try:
        l = []
        for l1 in list1:
            l2 = [literal_eval(x) for x in re.split('\s+-\s+|\s+-|-\s+|\s+|-', l1)]
            if len(l2) == 1:
                l += l2
            elif len(l2) == 2 and l2[1] > l2[0]:
                l += [i for i in range(l2[0], l2[1]+1)]
            else:
                raise ValueError
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    return sorted(set(l))

def get_trailing_int(string):
    indexRegex = re.compile(r'\d+$')
    mo = indexRegex.search(string)
    if mo is None:
        return None
    else:
        return int(mo.group())

def input_int(s=None, v_min=None, v_max=None, default=None):
    if default is not None:
        if not isinstance(default, int):
            illegal_value(default, 'default', 'input_int') 
            return None
        default_string = f' [{default}]'
    else:
        default_string = ''
    if v_min is not None:
        if not isinstance(v_min, int):
            illegal_value(vmin, 'vmin', 'input_int') 
            return None
        if default is not None and default < v_min:
            logging.error('Illegal v_min, default combination ({v_min}, {default})')
            return None
    if v_max is not None:
        if not isinstance(v_max, int):
            illegal_value(vmax, 'vmax', 'input_int') 
            return None
        if v_min is not None and v_min > v_max:
            logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
            return None
        if default is not None and default > v_max:
            logging.error('Illegal default, v_max combination ({default}, {v_max})')
            return None
    if v_min is not None and v_max is not None:
        v_range = f' (in range [{v_min}, {v_max}])'
    elif v_min is not None:
        v_range = f' (>= {v_min})'
    elif v_max is not None:
        v_range = f' (<= {v_max})'
    else:
        v_range = ''
    if s is None:
        print(f'Enter an integer{v_range}{default_string}: ')
    else:
        print(f'{s}{v_range}{default_string}: ')
    try:
        i = input()
        if isinstance(i, str) and not len(i):
            v = default
        else:
            v = literal_eval(i)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        v = None
    except:
        print('Unexpected error')
        raise
    if not is_int(v, v_min, v_max):
        print('Illegal input, enter a valid integer')
        v = input_int(s, v_min, v_max, default)
    return v

def input_num(s=None, v_min=None, v_max=None, default=None):
    if default is not None:
        if not isinstance(default, (int, float)):
            illegal_value(default, 'default', 'input_num') 
            return None
        default_string = f' [{default}]'
    else:
        default_string = ''
    if v_min is not None:
        if not isinstance(v_min, (int, float)):
            illegal_value(vmin, 'vmin', 'input_num') 
            return None
        if default is not None and default < v_min:
            logging.error('Illegal v_min, default combination ({v_min}, {default})')
            return None
    if v_max is not None:
        if not isinstance(v_max, (int, float)):
            illegal_value(vmax, 'vmax', 'input_num') 
            return None
        if v_min is not None and v_max < v_min:
            logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
            return None
        if default is not None and default > v_max:
            logging.error('Illegal default, v_max combination ({default}, {v_max})')
            return None
    if v_min is not None and v_max is not None:
        v_range = f' (in range [{v_min}, {v_max}])'
    elif v_min is not None:
        v_range = f' (>= {v_min})'
    elif v_max is not None:
        v_range = f' (<= {v_max})'
    else:
        v_range = ''
    if s is None:
        print(f'Enter a number{v_range}{default_string}: ')
    else:
        print(f'{s}{v_range}{default_string}: ')
    try:
        i = input()
        if isinstance(i, str) and not len(i):
            v = default
        else:
            v = literal_eval(i)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        v = None
    except:
        print('Unexpected error')
        raise
    if not is_num(v, v_min, v_max):
        print('Illegal input, enter a valid number')
        v = input_num(s, v_min, v_max, default)
    return v

def input_int_list(s=None, v_min=None, v_max=None):
    if v_min is not None and not isinstance(v_min, int):
        illegal_value(vmin, 'vmin', 'input_int_list') 
        return None
    if v_max is not None:
        if not isinstance(v_max, int):
            illegal_value(vmax, 'vmax', 'input_int_list') 
            return None
        if v_max < v_min:
            logging.error(f'Illegal v_min, v_max combination ({v_min}, {v_max})')
            return None
    if v_min is not None and v_max is not None:
        v_range = f' (each value in range [{v_min}, {v_max}])'
    elif v_min is not None:
        v_range = f' (each value >= {v_min})'
    elif v_max is not None:
        v_range = f' (each value <= {v_max})'
    else:
        v_range = ''
    if s is None:
        print(f'Enter a series of integers{v_range}: ')
    else:
        print(f'{s}{v_range}: ')
    try:
        l = string_to_list(input())
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        l = None
    except:
        print('Unexpected error')
        raise
    if (not isinstance(l, list) or
            True in [True if not is_int(v, v_min, v_max) else False for v in l]):
        print('Illegal input: enter a valid set of dash/comma/whitespace separated integers '+
                'e.g. 2,3,5-8,10')
        l = input_int_list(s, v_min, v_max)
    return l

def input_yesno(s=None, default=None):
    if default is not None:
        if not isinstance(default, str):
            illegal_value(default, 'default', 'input_yesno') 
            return None
        if default.lower() in 'yes':
            default = 'y'
        elif default.lower() in 'no':
            default = 'n'
        else:
            illegal_value(default, 'default', 'input_yesno') 
            return None
        default_string = f' [{default}]'
    else:
        default_string = ''
    if s is None:
        print(f'Enter yes or no{default_string}: ')
    else:
        print(f'{s}{default_string}: ')
    i = input()
    if isinstance(i, str) and not len(i):
        i = default
    if i.lower() in 'yes':
        v = True
    elif i.lower() in 'no':
        v = False
    else:
        print('Illegal input, enter yes or no')
        v = input_yesno(s, default)
    return v

def create_mask(x, bounds=None, reverse_mask=False, current_mask=None):
    # bounds is a pair of number in the same units a x
    if not isinstance(x, (tuple, list, np.ndarray)) or not len(x):
        logging.warning(f'Illegal input array ({x}, {type(x)})')
        return None
    if bounds is not None and not is_num_pair(bounds):
        logging.warning(f'Illegal bounds parameter ({bounds} {type(bounds)}, input ignored')
        bounds = None
    if bounds is not None:
        if not reverse_mask:
            mask = np.logical_and(x > min(bounds), x < max(bounds))
        else:
            mask = np.logical_or(x < min(bounds), x > max(bounds))
    else:
        mask = np.ones(len(x), dtype=bool)
    if current_mask is not None:
        if not isinstance(current_mask, (tuple, list, np.ndarray)) or len(current_mask) != len(x):
            logging.warning(f'Illegal current_mask ({current_mask}, {type(current_mask)}), '+
                    'input ignored')
        else:
            mask = np.logical_and(mask, current_mask)
    if not True in mask:
        logging.warning('Entire data array is masked')
    return mask

def findImageFiles(path, filetype, name=None):
    if isinstance(name, str):
        name = f' {name} '
    else:
        name = ' '
    # Find available index range
    if filetype == 'tif':
        if not isinstance(path, str) or not os.path.isdir(path):
            illegal_value(path, 'path', 'findImageRange')
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
        if first_index is None or last_index is None:
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
            illegal_value(path, 'path', 'findImageRange')
            return -1, 0, []
        # At this point only h5 in alamo2 detector style
        first_index = 0
        with h5py.File(path, 'r') as f:
            num_imgs = f['entry/instrument/detector/data'].shape[0]
            last_index = num_imgs-1
        paths = [path]
    else:
        illegal_value(filetype, 'filetype', 'findImageRange')
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
        if num_required is None:
            if use_input != 'no':
                use_input = pyip.inputYesNo('Current number of'+name+'images = '+
                        f'{num_imgs}, use this value ([y]/n)? ',
                        blank=True)
    if use_input != 'no':
        return first_index, offset, num_imgs

    # Check range against requirements
    if num_imgs < 1:
        logging.warning('No available'+name+'images')
        return -1, -1, 0
    if num_required is None:
        if num_imgs == 1:
            return first_index, 0, 1
    else:
        if not is_int(num_required, 1):
            illegal_value(num_required, 'num_required', 'selectImageRange')
            return -1, -1, 0
        if num_imgs < num_required:
            logging.error('Unable to find the required'+name+
                    f'images ({num_imgs} out of {num_required})')
            return -1, -1, 0

    # Select index range
    print('\nThe number of available'+name+f'images is {num_imgs}')
    if num_required is None:
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
    img_read = plt.imread(f)
    if not img_x_bounds:
        img_x_bounds = (0, img_read.shape[0])
    else:
        if (not isinstance(img_x_bounds, (tuple, list)) or len(img_x_bounds) != 2 or 
                not (0 <= img_x_bounds[0] < img_x_bounds[1] <= img_read.shape[0])):
            logging.error(f'inconsistent row dimension in {f}')
            return None
    if not img_y_bounds:
        img_y_bounds = (0, img_read.shape[1])
    else:
        if (not isinstance(img_y_bounds, list) or len(img_y_bounds) != 2 or 
                not (0 <= img_y_bounds[0] < img_y_bounds[1] <= img_read.shape[1])):
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
            illegal_value(files[0], 'files[0]', 'loadImageStack')
            return img_stack
        t0 = time()
        logging.info(f'Loading {files[0]}')
        with h5py.File(files[0], 'r') as f:
            shape = f['entry/instrument/detector/data'].shape
            if len(shape) != 3:
                logging.error(f'inconsistent dimensions in {files[0]}')
            if not img_x_bounds:
                img_x_bounds = (0, shape[1])
            else:
                if (not isinstance(img_x_bounds, (tuple, list)) or len(img_x_bounds) != 2 or 
                        not (0 <= img_x_bounds[0] < img_x_bounds[1] <= shape[1])):
                    logging.error(f'inconsistent row dimension in {files[0]} {img_x_bounds} '+
                            f'{shape[1]}')
            if not img_y_bounds:
                img_y_bounds = (0, shape[2])
            else:
                if (not isinstance(img_y_bounds, list) or len(img_y_bounds) != 2 or 
                        not (0 <= img_y_bounds[0] < img_y_bounds[1] <= shape[2])):
                    logging.error(f'inconsistent column dimension in {files[0]}')
            img_stack = f.get('entry/instrument/detector/data')[
                    img_offset:img_offset+num_imgs:num_img_skip+1,
                    img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]
        logging.info(f'... done in {time()-t0:.2f} seconds!')
    else:
        illegal_value(filetype, 'filetype', 'findImageRange')
    return img_stack

def combine_tiffs_in_h5(files, num_imgs, h5_filename):
    img_stack = loadImageStack(files, 'tif', 0, num_imgs)
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('entry/instrument/detector/data', data=img_stack)
    del img_stack
    return [h5_filename]

def clearImshow(title=None):
    plt.ioff()
    if title is None:
        title = 'quick imshow'
    elif not isinstance(title, str):
        illegal_value(title, 'title', 'clearImshow')
        return
    plt.close(fig=re.sub(r"\s+", '_', title))

def clearPlot(title=None):
    plt.ioff()
    if title is None:
        title = 'quick plot'
    elif not isinstance(title, str):
        illegal_value(title, 'title', 'clearPlot')
        return
    plt.close(fig=re.sub(r"\s+", '_', title))

def quickImshow(a, title=None, path=None, name=None, save_fig=False, save_only=False,
            clear=True, extent=None, show_grid=False, grid_color='w', grid_linewidth=1, **kwargs):
    if title is not None and not isinstance(title, str):
        illegal_value(title, 'title', 'quickImshow')
        return
    if path is not None and not isinstance(path, str):
        illegal_value(path, 'path', 'quickImshow')
        return
    if not isinstance(save_fig, bool):
        illegal_value(save_fig, 'save_fig', 'quickImshow')
        return
    if not isinstance(save_only, bool):
        illegal_value(save_only, 'save_only', 'quickImshow')
        return
    if not isinstance(clear, bool):
        illegal_value(clear, 'clear', 'quickImshow')
        return
    if not title:
        title='quick imshow'
#    else:
#        title = re.sub(r"\s+", '_', title)
    if name is None:
        ttitle = re.sub(r"\s+", '_', title)
        if path is None:
            path = f'{ttitle}.png'
        else:
            path = f'{path}/{ttitle}.png'
    else:
        if path is None:
            path = name
        else:
            path = f'{path}/{name}'
    if extent is None:
        extent = (0, a.shape[1], a.shape[0], 0)
    if clear:
        plt.close(fig=title)
    if not save_only:
        plt.ion()
    plt.figure(title)
    plt.imshow(a, extent=extent, **kwargs)
    if show_grid:
        ax = plt.gca()
        ax.grid(color=grid_color, linewidth=grid_linewidth)
#    if title != 'quick imshow':
#        plt.title = title
    if save_only:
        plt.savefig(path)
        plt.close(fig=title)
    else:
        if save_fig:
            plt.savefig(path)

def quickPlot(*args, xerr=None, yerr=None, vlines=None, title=None, xlim=None, ylim=None,
        xlabel=None, ylabel=None, legend=None, path=None, name=None, show_grid=False, 
        save_fig=False, save_only=False, clear=True, block=False, **kwargs):
    if title is not None and not isinstance(title, str):
        illegal_value(title, 'title', 'quickPlot')
        title = None
    if xlim is not None and not isinstance(xlim, (tuple, list)) and len(xlim) != 2:
        illegal_value(xlim, 'xlim', 'quickPlot')
        xlim = None
    if ylim is not None and not isinstance(ylim, (tuple, list)) and len(ylim) != 2:
        illegal_value(ylim, 'ylim', 'quickPlot')
        ylim = None
    if xlabel is not None and not isinstance(xlabel, str):
        illegal_value(xlabel, 'xlabel', 'quickPlot')
        xlabel = None
    if ylabel is not None and not isinstance(ylabel, str):
        illegal_value(ylabel, 'ylabel', 'quickPlot')
        ylabel = None
    if legend is not None and not isinstance(legend, (tuple, list)):
        illegal_value(legend, 'legend', 'quickPlot')
        legend = None
    if path is not None and not isinstance(path, str):
        illegal_value(path, 'path', 'quickPlot')
        return
    if not isinstance(show_grid, bool):
        illegal_value(show_grid, 'show_grid', 'quickPlot')
        return
    if not isinstance(save_fig, bool):
        illegal_value(save_fig, 'save_fig', 'quickPlot')
        return
    if not isinstance(save_only, bool):
        illegal_value(save_only, 'save_only', 'quickPlot')
        return
    if not isinstance(clear, bool):
        illegal_value(clear, 'clear', 'quickPlot')
        return
    if not isinstance(block, bool):
        illegal_value(block, 'block', 'quickPlot')
        return
    if title is None:
        title = 'quick plot'
#    else:
#        title = re.sub(r"\s+", '_', title)
    if name is None:
        ttitle = re.sub(r"\s+", '_', title)
        if path is None:
            path = f'{ttitle}.png'
        else:
            path = f'{path}/{ttitle}.png'
    else:
        if path is None:
            path = name
        else:
            path = f'{path}/{name}'
    if clear:
        plt.close(fig=title)
    args = unwrap_tuple(args)
    if depth_tuple(args) > 1 and (xerr is not None or yerr is not None):
        logging.warning('Error bars ignored form multiple curves')
    if not save_only:
        if block:
            plt.ioff()
        else:
            plt.ion()
    plt.figure(title)
    if depth_tuple(args) > 1:
       for y in args:
           plt.plot(*y, **kwargs)
    else:
        if xerr is None and yerr is None:
            plt.plot(*args, **kwargs)
        else:
            plt.errorbar(*args, xerr=xerr, yerr=yerr, **kwargs)
    if vlines is not None:
        for v in vlines:
            plt.axvline(v, color='r', linestyle='--', **kwargs)
#    if vlines is not None:
#        for s in tuple(([x, x], list(plt.gca().get_ylim())) for x in vlines):
#            plt.plot(*s, color='red', **kwargs)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if show_grid:
        ax = plt.gca()
        ax.grid(color='k')#, linewidth=1)
    if legend is not None:
        plt.legend(legend)
    if save_only:
        plt.savefig(path)
        plt.close(fig=title)
    else:
        if save_fig:
            plt.savefig(path)
        if block:
            plt.show(block=block)

def selectArrayBounds(a, x_low=None, x_upp=None, num_x_min=None, ask_bounds=False,
        title='select array bounds'):
    """Interactively select the lower and upper data bounds for a numpy array.
    """
    if isinstance(a, (tuple, list)):
        a = np.array(a)
    if not isinstance(a, np.ndarray) or a.ndim != 1:
        illegal_value(a.ndim, 'array type or dimension', 'selectArrayBounds')
        return None
    len_a = len(a)
    if num_x_min is None:
        num_x_min = 1
    else:
        if num_x_min < 2 or num_x_min > len_a:
            logging.warning('Illegal value for num_x_min in selectArrayBounds, input ignored')
            num_x_min = 1

    # Ask to use current bounds
    if ask_bounds and (x_low is not None or x_upp is not None):
        if x_low is None:
            x_low = 0
        if not is_int(x_low, 0, len_a-num_x_min):
            illegal_value(x_low, 'x_low', 'selectArrayBounds')
            return None
        if x_upp is None:
            x_upp = len_a
        if not is_int(x_upp, x_low+num_x_min, len_a):
            illegal_value(x_upp, 'x_upp', 'selectArrayBounds')
            return None
        quickPlot((range(len_a), a), vlines=(x_low,x_upp), title=title)
        if pyip.inputYesNo(f'\nCurrent array bounds: [{x_low}, {x_upp}], '+
                'use these values ([y]/n)? ', blank=True) == 'no':
            x_low = None
            x_upp = None
        else:
            clearPlot(title)
            return x_low, x_upp

    if x_low is None:
        x_min = 0
        x_max = len_a
        x_low_max = len_a-num_x_min
        while True:
            quickPlot(range(x_min, x_max), a[x_min:x_max], title=title)
            zoom_flag = pyip.inputInt('Set lower data bound ([0]) or zoom in (1)?: ',
                    min=0, max=1, blank=True)
            if zoom_flag == 1:
                x_min = pyip.inputInt(f'    Set lower zoom index [0, {x_low_max}]: ',
                        min=0, max=x_low_max)
                x_max = pyip.inputInt(f'    Set upper zoom index [{x_min+1}, {x_low_max+1}]: ',
                        min=x_min+1, max=x_low_max+1)
            else:
                x_low = pyip.inputInt(f'    Set lower data bound [0, {x_low_max}]: ',
                        min=0, max=x_low_max)
                break
    else:
        if not is_int(x_low, 0, len_a-num_x_min):
            illegal_value(x_low, 'x_low', 'selectArrayBounds')
            return None
    if x_upp is None:
        x_min = x_low+num_x_min
        x_max = len_a
        x_upp_min = x_min
        while True:
            quickPlot(range(x_min, x_max), a[x_min:x_max], title=title)
            zoom_flag = pyip.inputInt('Set upper data bound ([0]) or zoom in (1)?: ',
                    min=0, max=1, blank=True)
            if zoom_flag == 1:
                x_min = pyip.inputInt(f'    Set upper zoom index [{x_upp_min}, {len_a-1}]: ',
                        min=x_upp_min, max=len_a-1)
                x_max = pyip.inputInt(f'    Set upper zoom index [{x_min+1}, {len_a}]: ',
                        min=x_min+1, max=len_a)
            else:
                x_upp = pyip.inputInt(f'    Set upper data bound [{x_upp_min}, {len_a}]: ',
                        min=x_upp_min, max=len_a)
                break
    else:
        if not is_int(x_upp, x_low+num_x_min, len_a):
            illegal_value(x_upp, 'x_upp', 'selectArrayBounds')
            return None
    print(f'lower bound = {x_low} (inclusive)\nupper bound = {x_upp} (exclusive)]')
    quickPlot((range(len_a), a), vlines=(x_low,x_upp), title=title)
    if pyip.inputYesNo('Accept these bounds ([y]/n)?: ', blank=True) == 'no':
        x_low, x_upp = selectArrayBounds(a, None, None, num_x_min, title=title)
    clearPlot(title)
    return x_low, x_upp

def selectImageBounds(a, axis, low=None, upp=None, num_min=None,
        title='select array bounds'):
    """Interactively select the lower and upper data bounds for a 2D numpy array.
    """
    if isinstance(a, np.ndarray):
        if a.ndim != 2:
            illegal_value(a.ndim, 'array dimension', 'selectImageBounds')
            return None
    elif isinstance(a, (tuple, list)):
        if len(a) != 2:
            illegal_value(len(a), 'array dimension', 'selectImageBounds')
            return None
        if len(a[0]) != len(a[1]) or not (isinstance(a[0], (tuple, list, np.ndarray)) and
                isinstance(a[1], (tuple, list, np.ndarray))):
            logging.error(f'Illegal array type in selectImageBounds ({type(a[0])} {type(a[1])})')
            return None
        a = np.array(a)
    else:
        illegal_value(a, 'array type', 'selectImageBounds')
        return None
    if axis < 0 or axis >= a.ndim:
        illegal_value(axis, 'axis', 'selectImageBounds')
        return None
    low_save = low
    upp_save = upp
    num_min_save = num_min
    if num_min is None:
        num_min = 1
    else:
        if num_min < 2 or num_min > a.shape[axis]:
            logging.warning('Illegal input for num_min in selectImageBounds, input ignored')
            num_min = 1
    if low is None:
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
            illegal_value(low, 'low', 'selectImageBounds')
            return None
    if upp is None:
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
            illegal_value(upp, 'upp', 'selectImageBounds')
            return None
    bounds = (low, upp)
    a_tmp = np.copy(a)
    a_tmp_max = a.max()
    if axis:
        a_tmp[:,bounds[0]] = a_tmp_max
        a_tmp[:,bounds[1]-1] = a_tmp_max
    else:
        a_tmp[bounds[0],:] = a_tmp_max
        a_tmp[bounds[1]-1,:] = a_tmp_max
    print(f'lower bound = {low} (inclusive)\nupper bound = {upp} (exclusive)')
    quickImshow(a_tmp, title=title)
    del a_tmp
    if pyip.inputYesNo('Accept these bounds ([y]/n)?: ', blank=True) == 'no':
        bounds = selectImageBounds(a, axis, low=low_save, upp=upp_save, num_min=num_min_save,
            title=title)
    return bounds


class Fit:
    """Wrapper class for lmfit
    """
    def __init__(self, x, y, models=None, **kwargs):
        self._x = x
        self._y = y
        self._model = None
        self._parameters = Parameters()
        self._result = None
        if models is not None:
            if callable(models) or isinstance(models, str):
                kwargs = self.addModel(models, **kwargs)
            elif isinstance(models, (tuple, list)):
                for model in models:
                    kwargs = self.addModel(model, **kwargs)
            self.fit(**kwargs)

    @classmethod
    def fitData(cls, x, y, models, **kwargs):
        return cls(x, y, models, **kwargs)

    @property
    def init_values(self):
        if self._result is None:
            return None
        return self._result.init_values

    @property
    def best_values(self):
        if self._result is None:
            return None
        return self._result.params.valuesdict()

    @property
    def best_errors(self):
        if self._result is None:
            return None
        errors = {}
        names = sorted(self._result.params)
        for name in names:
            par = self._result.params[name]
            errors[name] = par.stderr
        return errors

    @property
    def best_fit(self):
        if self._result is None:
            return None
        return self._result.best_fit

    @property
    def best_parameters(self):
        if self._result is None:
            return None
        parameters = []
        names = sorted(self._result.params)
        for name in names:
            par = self._result.params[name]
            parameters.append({'name' : par.name, 'value' : par.value, 'error' : par.stderr,
                    'init_value' : par.init_value, 'min' : par.min, 'max' : par.max,
                    'vary' : par.vary, 'expr' : par.expr})
        return parameters

    @property
    def var_names(self):
        """Intended to be used with cover
        """
        if self._result is None:
            return None
        return self._result.var_names

    @property
    def covar(self):
        return self._result.covar

    @property
    def chisqr(self):
        return self._result.chisqr

    @property
    def redchi(self):
        return self._result.redchi

    @property
    def residual(self):
        return self._result.residual

    @property
    def success(self):
        if not self._result.success:
#            print(f'ier = {self._result.ier}')
#            print(f'lmdif_message = {self._result.lmdif_message}')
#            print(f'message = {self._result.message}')
#            print(f'nfev = {self._result.nfev}')
#            print(f'redchi = {self._result.redchi}')
#            print(f'success = {self._result.success}')
            if self._result.ier == 0 or self._result.ier == 5:
                logging.warning(f'ier = {self._result.ier}: {self._result.message}')
            else:
                logging.warning(f'ier = {self._result.ier}: {self._result.message}')
                return True
#            self.printFitReport()
#            self.plot()
        return self._result.success

    @property
    def num_func_eval(self):
        return self._result.nfev

    def printFitReport(self, show_correl=False):
        if self._result is not None:
            print(self._result.fit_report(show_correl=show_correl))

    def addParameter(self, **parameter):
        if not isinstance(parameter, dict):
            illegal_value(parameter, 'parameter', 'addParameter')
            return
        self._parameters.add(**parameter)

    def addModel(self, model, prefix=None, parameters=None, **kwargs):
        # Create the new model
#        print('\nAt start adding model:')
#        self._parameters.pretty_print()
        if prefix is not None and not isinstance(prefix, str):
            logging.warning('Ignoring illegal prefix: {model} {type(model)}')
            prefix = None
        if callable(model):
            newmodel = Model(model, prefix=prefix)
        elif isinstance(model, str):
            if model == 'constant':
                newmodel = ConstantModel(prefix=prefix)
            elif model == 'linear':
                newmodel = LinearModel(prefix=prefix)
            elif model == 'quadratic':
                newmodel = QuadraticModel(prefix=prefix)
            elif model == 'gaussian':
                newmodel = GaussianModel(prefix=prefix)
            elif model == 'step':
                form = kwargs.get('form')
                if form is not None:
                    del kwargs['form']
                if form is None or form not in ('linear', 'atan', 'arctan', 'erf', 'logistic'):
                    logging.error(f'Illegal form parameter for build-in step model ({form})')
                    return kwargs
                newmodel = StepModel(prefix=prefix, form=form)
            elif model == 'rectangle':
                form = kwargs.get('form')
                if form is not None:
                    del kwargs['form']
                if form is None or form not in ('linear', 'atan', 'arctan', 'erf', 'logistic'):
                    logging.error(f'Illegal form parameter for build-in rectangle model ({form})')
                    return kwargs
                newmodel = RectangleModel(prefix=prefix, form=form)
            else:
                logging.error('Unknown build-in fit model')
                return kwargs
        else:
            illegal_value(model, 'model', 'addModel')
            return kwargs

        # Add the new model to the current one
        if self._model is None:
            self._model = newmodel
        else:
            self._model += newmodel
        if self._parameters is None:
            self._parameters = newmodel.make_params()
        else:
            self._parameters += newmodel.make_params()
#        print('\nAfter adding model:')
#        self._parameters.pretty_print()

        # Initialize the model parameters
        if prefix is None:
            prefix = ""
        if parameters is not None:
            if not isinstance(parameters, (tuple, list)):
                illegal_value(parameters, 'parameters', 'addModel')
                return kwargs
            for parameter in parameters:
                if not isinstance(parameter, dict):
                    illegal_value(parameter, 'parameter in parameters', 'addModel')
                    return kwargs
                parameter['name']  = prefix+parameter['name']
                self._parameters.add(**parameter)
        for name, value in kwargs.items():
            if isinstance(value, (int, float)):
                self._parameters.add(prefix+name, value=value)
#        print('\nAt end addModel:')
#        self._parameters.pretty_print()

        return kwargs

    def fit(self, interactive=False, guess=False, **kwargs):
        if self._model is None:
            logging.error('Undefined fit model')
            return
#        print(f'kwargs = {kwargs}')
#        self._parameters.pretty_print()
        # Current parameter values
        pars = self._parameters.valuesdict()
        # Apply parameter updates through keyword arguments
        for par in set(pars) & set(kwargs):
            pars[par] = kwargs.pop(par)
            self._parameters[par].set(value=pars[par])
        # Check for uninitialized parameters
        for par, value in pars.items():
            if value is None or np.isinf(value) or np.isnan(value):
                if interactive:
                    self._parameters[par].set(value=
                            pyip.inputNum(f'Enter an initial value for {par}: '))
                else:
                    self._parameters[par].set(value=1.0)
#        print('\nAt start actual fit:')
#        print(f'kwargs = {kwargs}')
#        self._parameters.pretty_print()
#        print(f'parameters:\n{self._parameters}')
#        print(f'x = {self._x}')
#        print(f'len(x) = {len(self._x)}')
#        print(f'y = {self._y}')
#        print(f'len(y) = {len(self._y)}')
        if guess:
            self._parameters = self._model.guess(self._y, x=self._x)
        self._result = self._model.fit(self._y, self._parameters, x=self._x, **kwargs)
#        print('\nAt end actual fit:')
#        print(f'var_names:\n{self._result.var_names}')
#        print(f'stderr:\n{np.sqrt(np.diagonal(self._result.covar))}')
#        self._parameters.pretty_print()
#        print(f'parameters:\n{self._parameters}')
#        print(f'values:\n{self._result.best_values}')

    def plot(self):
        if self._result is None:
            return
        components = self._result.eval_components()
        plots = ((self._x, self._y, '.'), (self._x, self._result.best_fit, 'k-'),
                (self._x, self._result.init_fit, 'g-'))
        legend = ['data', 'best fit', 'init']
        if len(components) > 1:
            for modelname, y_fit in components.items():
                if isinstance(y_fit, (int, float)):
                    y_fit *= np.ones(self._x.size)
                plots += ((self._x, y_fit, '--'),)
#                if modelname[-1] == '_':
#                    legend.append(modelname[:-1])
#                else:
#                    legend.append(modelname)
        quickPlot(plots, legend=legend, block=True)

    @staticmethod
    def guess_init_peak(x, y, *args, center_guess=None, use_max_for_center=True):
        """ Return a guess for the initial height, center and fwhm for a peak
        """
        center_guesses = None
        if len(x) != len(y):
            logging.error(f'Illegal x and y lengths ({len(x)}, {len(y)}), skip initial guess')
            return None, None, None
        if isinstance(center_guess, (int, float)):
            if len(args):
                logging.warning('Ignoring additional arguments for single center_guess value')
        elif isinstance(center_guess, (tuple, list, np.ndarray)):
            if len(center_guess) == 1:
                logging.warning('Ignoring additional arguments for single center_guess value')
                if not isinstance(center_guess[0], (int, float)):
                    raise ValueError(f'Illegal center_guess type ({type(center_guess[0])})')
                center_guess = center_guess[0]
            else:
                if len(args) != 1:
                    raise ValueError(f'Illegal number of arguments ({len(args)})')
                n = args[0]
                if not is_index(n, 0, len(center_guess)):
                    raise ValueError('Illegal argument')
                center_guesses = center_guess
                center_guess = center_guesses[n]
        elif center_guess is not None:
            raise ValueError(f'Illegal center_guess type ({type(center_guess)})')

        # Sort the inputs
        index = np.argsort(x)
        x = x[index]
        y = y[index]
        miny = y.min()
#        print(f'miny = {miny}')
#        print(f'x_range = {x[0]} {x[-1]} {len(x)}')
#        print(f'y_range = {y[0]} {y[-1]} {len(y)}')

        xx = x
        yy = y
        # Set range for current peak
#        print(f'center_guesses = {center_guesses}')
        if center_guesses is not None:
            if n == 0:
               low = 0
               upp = index_nearest(x, (center_guesses[0]+center_guesses[1])/2)
            elif n == len(center_guesses)-1:
               low = index_nearest(x, (center_guesses[n-1]+center_guesses[n])/2)
               upp = len(x)
            else:
               low = index_nearest(x, (center_guesses[n-1]+center_guesses[n])/2)
               upp = index_nearest(x, (center_guesses[n]+center_guesses[n+1])/2)
#            print(f'low = {low}')
#            print(f'upp = {upp}')
            x = x[low:upp]
            y = y[low:upp]
#            quickPlot(x, y, vlines=(x[0], center_guess, x[-1]), block=True)

        # Estimate FHHM
        maxy = y.max()
#        print(f'x_range = {x[0]} {x[-1]} {len(x)}')
#        print(f'y_range = {y[0]} {y[-1]} {len(y)} {miny} {maxy}')
#        print(f'center_guess = {center_guess}')
        if center_guess is None:
            center_index = np.argmax(y)
            center = x[center_index]
            height = maxy-miny
        else:
            if use_max_for_center:
                center_index = np.argmax(y)
                center = x[center_index]
                if center_index < 0.1*len(x) or center_index > 0.9*len(x):
                    center_index = index_nearest(x, center_guess)
                    center = center_guess
            else:
                center_index = index_nearest(x, center_guess)
                center = center_guess
            height = y[center_index]-miny
#        print(f'center_index = {center_index}')
#        print(f'center = {center}')
#        print(f'height = {height}')
        half_height = miny+0.5*height
#        print(f'half_height = {half_height}')
        fwhm_index1 = 0
        for i in range(center_index, fwhm_index1, -1):
            if y[i] < half_height:
                fwhm_index1 = i
                break
#        print(f'fwhm_index1 = {fwhm_index1} {x[fwhm_index1]}')
        fwhm_index2 = len(x)-1
        for i in range(center_index, fwhm_index2):
            if y[i] < half_height:
                fwhm_index2 = i
                break
#        print(f'fwhm_index2 = {fwhm_index2} {x[fwhm_index2]}')
#        quickPlot((x,y,'o'), vlines=(x[fwhm_index1], center, x[fwhm_index2]), block=True)
        if fwhm_index1 == 0 and fwhm_index2 < len(x)-1:
            fwhm = 2*(x[fwhm_index2]-center)
        elif fwhm_index1 > 0 and fwhm_index2 == len(x)-1:
            fwhm = 2*(center-x[fwhm_index1])
        else:
            fwhm = x[fwhm_index2]-x[fwhm_index1]
#        print(f'fwhm_index1 = {fwhm_index1} {x[fwhm_index1]}')
#        print(f'fwhm_index2 = {fwhm_index2} {x[fwhm_index2]}')
#        print(f'fwhm = {fwhm}')

        # Return height, center and FWHM
#        quickPlot((x,y,'o'), (xx,yy), vlines=(x[fwhm_index1], center, x[fwhm_index2]), block=True)
        return height, center, fwhm


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

        # Load config file (for now for Galaxy, allow .dat extension)
        self.suffix = os.path.splitext(config_file)[1]
        if self.suffix == '.yml' or self.suffix == '.yaml' or self.suffix == '.dat':
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        elif self.suffix == '.txt':
            with open(config_file, 'r') as f:
                lines = f.read().splitlines()
            self.config = {item[0].strip():literal_eval(item[1].strip()) for item in
                    [line.split('#')[0].split('=') for line in lines if '=' in line.split('#')[0]]}
        else:
            illegal_value(self.suffix, 'config file extension', 'loadFile')

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
            illegal_value(config_dict, 'dictionary config object', 'loadDict')
            self.config = {}

    def saveFile(self, config_file):
        """Save the config file (as a yaml file only right now).
        """
        suffix = os.path.splitext(config_file)[1]
        if suffix != '.yml' and suffix != '.yaml':
            illegal_value(suffix, 'config file extension', 'saveFile')

        # Check if config file exists
        if os.path.isfile(config_file):
            logging.info(f'Updating {config_file}')
        else:
            logging.info(f'Saving {config_file}')

        # Save config file
        with open(config_file, 'w') as f:
            yaml.safe_dump(self.config, f)

    def validate(self, pars_required, pars_missing=None):
        """Returns False if any required first level keys are missing.
        """
        if not self.load_flag:
            logging.error('Load a config file prior to calling Config.validate')
        pars = [p for p in pars_required if p not in self.config]
        if isinstance(pars_missing, list):
            pars_missing.extend(pars)
        elif pars_missing is not None:
            illegal_value(pars_missing, 'pars_missing', 'Config.validate')
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
#            illegal_value(key, 'key', 'Config.update')
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
#            if search_string is not None:
#                if isinstance(search_string, str):
#                    search_string = [search_string]
#                elif not isinstance(search_string, (tuple, list)):
#                    illegal_value(search_string, 'search_string', 'Config.update')
#                    search_string = None
#            update_flag = False
#            if search_string is not None:
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
            illegal_value(detector_id, 'detector_id', 'Detector.loadFile')
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
        if not isinstance(lens_magnification, (int, float)) or lens_magnification <= 0.:
            illegal_value(lens_magnification, 'lens_magnification', 'detector file')
            return 0
        pixel_size = self.detector['detector'].get('pixels').get('size')
        if isinstance(pixel_size, (int, float)):
            if pixel_size <= 0.:
                illegal_value(pixel_size, 'pixel_size', 'detector file')
                return 0
            pixel_size /= lens_magnification
        elif isinstance(pixel_size, list):
            if ((len(pixel_size) > 2) or
                    (len(pixel_size) == 2 and pixel_size[0] != pixel_size[1])):
                illegal_value(pixel_size, 'pixel size', 'detector file')
                return 0
            elif not is_num(pixel_size[0], 0.):
                illegal_value(pixel_size, 'pixel size', 'detector file')
                return 0
            else:
                pixel_size = pixel_size[0]/lens_magnification
        else:
            illegal_value(pixel_size, 'pixel size', 'detector file')
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
            illegal_value(num_rows, 'rows', 'detector file')
            return 0, 0
        num_columns = pixels.get('columns')
        if not is_int(num_columns, 1):
            illegal_value(num_columns, 'columns', 'detector file')
            return 0, 0

        return num_rows, num_columns

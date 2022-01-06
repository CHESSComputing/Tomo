#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:54:37 2021

@author: rv43
"""

import logging
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s %(message)s')

import os
import sys
import yaml
import re
from pathlib import Path
import pyinputplus as pyip
import numpy as np
import numexpr as ne
import multiprocessing as mp
import scipy.ndimage as spi
import tomopy
from time import time
from skimage.transform import iradon
from skimage.restoration import denoise_tv_chambolle

import msnc_tools as msnc

class set_numexpr_threads:

    def __init__(self, nthreads):
        cpu_count = mp.cpu_count()
        if nthreads is None or nthreads > cpu_count:
            self.n = cpu_count
        else:
            self.n = nthreads

    def __enter__(self):
        self.oldn = ne.set_num_threads(self.n)

    def __exit__(self, exc_type, exc_value, traceback):
        ne.set_num_threads(self.oldn)

class Tomo:
    """Processing tomography data with misalignment."""
    
    def __init__(self, filepath=None, config_dict=None):
        """Initialize with optional config input filepath or dictionary"""
        self.ncore = mp.cpu_count()
        self.config = {}
        self.detector = {}
        self.is_valid = False
        self.start_theta = 0.
        self.end_theta = 180.
        self.num_thetas = None
        self.num_tomo_data_sets = 1
        self.num_rows = None
        self.num_columns = None
        self.pixel_size = None
        self.tdf = np.array([])
        self.tbf = np.array([])
        self.tomo_sets = np.array([])
        self.loaded_tomo_sets = np.array([])
        self.plot_or_save = 0

        # Load config file 
        if not filepath is None:
           self.loadConfigFile(filepath)
        elif not config_dict is None:
           self.loadConfigDict(config_dict)
        else:
           pass

        # Check config info
        pars_missing = []
        self.is_valid = self.validateConfig(pars_missing)
        if len(pars_missing) > 0:
            logging.error(f'missing item(s) in config file: {", ".join(pars_missing)}')

        # Load detector file
        self.loadDetectorFile()

        # Check detector info
        pars_missing = []
        is_valid = self.validateDetector(pars_missing)
        if len(pars_missing) > 0:
            logging.error(f'missing item(s) in detector file: {", ".join(pars_missing)}')
        self.is_valid = is_valid and self.is_valid

    def loadConfigFile(self, filepath):
        '''Takes the full/relative path to a text or yml file and loads it into 
        the dictionary self.config'''

        # Ensure config file exists before opening
        if not os.path.isfile(filepath):
            logging.error(f'file does not exist: {filepath}')
            return

        # Load config file
        suffix = Path(filepath).suffix
        if suffix == '.yml' or suffix == '.yaml':
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
        elif suffix == '.txt':
            config = msnc.loadConfigFile(filepath)
        else:
            logging.error(f'illegal config file extension: {suffix}')

        # Make sure yaml could load config file as a dictionary
        if type(config) == dict:
            self.config = config
        else:
            logging.error(f'could not load dictionary from config file: {filepath}')

    def loadConfigDict(self, config_dict):
        '''Takes a dictionary and places it into self.config.'''

        # Ensure a dictionary was actually supplied
        if type(config_dict) == dict:
            self.config = config_dict
        else:
            logging.error(f'could not pass dictionary config object: {config_dict}')

    def validateConfig(self, pars_missing):
        '''Returns False if config parameters are missing or other errors are
        present.'''

        is_valid = True

        # Check fo required first-level keys
        pars_needed = ['tdf_data_folder', 'tbf_data_folder', 'detector_id']
        pars_missing.extend([p for p in pars_needed if p not in self.config])
        if len(pars_missing) > 0:
            is_valid = False

        # Check tomo angle (theta) range keys
        start_theta = self.config.get('start_theta', 0.)
        if type(start_theta) == int or type(start_theta) == float:
            self.start_theta = start_theta
        else:
            logging.error('illegal value for start_theta')
            is_valid = False
        end_theta = self.config.get('end_theta', 180.)
        if type(end_theta) == int or type(end_theta) == float:
            self.end_theta = end_theta
        else:
            logging.error('illegal value for end_theta')
            is_valid = False
        if 'num_thetas' in self.config:
            num_thetas = self.config['num_thetas']
            if type(num_thetas) != int:
                num_thetas = None
                logging.error('illegal value for num_thetas')
                is_valid = False
        else:
            num_thetas = pyip.inputInt('\nEnter the number of thetas (>0): ', greaterThan=0)
            # Update config file
            self.config = msnc.addtoConfigFile('config.txt', 'Scan parameters', f'num_thetas = {num_thetas}')
        self.num_thetas = num_thetas
        logging.debug(f'start_theta = {self.start_theta}')
        logging.debug(f'end_theta = {self.end_theta}')
        logging.debug(f'num_thetas = {self.num_thetas}')

        # Check number of tomography image stacks
        num_tomo_data_sets = self.config.get('num_tomo_data_sets', 1)
        if type(num_tomo_data_sets) != int or num_tomo_data_sets < 1:
            num_tomo_data_sets = None
            logging.error('illegal value for num_tomo_data_sets')
            is_valid = False
        self.num_tomo_data_sets = num_tomo_data_sets
        logging.info(f'num_tomo_data_sets = {self.num_tomo_data_sets}')

        return is_valid

    def loadDetectorFile(self):
        '''looks for a detector yml file and loads it into the 
        dictionary self.detector'''

        if not self.config.get('detector_id'):
            return

        # Ensure config file exists before opening
        filepath = self.config['detector_id'] + '.yml'
        if not os.path.isfile(filepath):
            filepath = self.config['detector_id'] + '.yaml'
            if not os.path.isfile(filepath):
                logging.error(f'file does not exist: {filepath}')
                return

        # Load detector yml file
        with open(filepath, 'r') as f:
            detector = yaml.safe_load(f)

        # Make sure yaml could load detector file as a dictionary
        if type(detector) == dict:
            self.detector = detector
        else:
            logging.error(f'could not load dictionary from file: {filepath}')

    def validateDetector(self, pars_missing):
        '''Returns False if detector parameters are missing or other errors are
        present.'''

        # Check fo required first-level keys
        pars_needed = ['detector']
        pars_missing.extend([p for p in pars_needed if p not in self.detector])
        if len(pars_missing) > 0:
            return False

        is_valid = True

        # Check detector pixel config keys
        lens_magnification = self.config.get('lens_magnification', 1.)
        if ((type(lens_magnification) != int and type(lens_magnification) != float)
                or lens_magnification <= 0.):
            lens_magnification = 1.
            logging.error('illegal lens_magnification in config info')
            is_valid = False
        pixels = self.detector['detector'].get('pixels')
        if not pixels:
            pars_missing.append('detector:pixels')
        else:
            num_rows = pixels.get('rows')
            if not num_rows:
                pars_missing.append('detector:pixels:rows')
            else:
                if type(num_rows) == int:
                    self.num_rows = num_rows
                else:
                    logging.error('illegal number of rows in detector info')
                    is_valid = False
            num_columns = pixels.get('columns')
            if not num_columns:
                pars_missing.append('detector:pixels:columns')
            else:
                if type(num_columns) == int:
                    self.num_columns = num_columns
                else:
                    logging.error('illegal number of columns in detector info')
                    is_valid = False
            pixel_size = pixels.get('size')
            if not pixel_size:
                pars_missing.append('detector:pixels:size')
            else:
                if type(pixel_size) == int or type(pixel_size) == float:
                    self.pixel_size = pixel_size/lens_magnification
                elif type(pixel_size) == list:
                    if ((len(pixel_size) > 2) or 
                            (len(pixel_size) == 2 and pixel_size[0] != pixel_size[1])):
                        logging.error('illegal pixel size dimension in detector info 1')
                        is_valid = False
                    elif type(pixel_size[0]) != int and type(pixel_size[0]) != float:
                        logging.error('illegal pixel size dimension in detector info 2')
                        is_valid = False
                    else:
                        self.pixel_size = pixel_size[0]/lens_magnification
                else:
                    logging.error('illegal pixel size dimension in detector info 3')
                    is_valid = False
        logging.debug(f'num_rows = {self.num_rows}')
        logging.debug(f'num_columns = {self.num_columns}')
        logging.debug(f'pixel_size = {self.pixel_size}')

        if len(pars_missing) > 0:
            is_valid = False

        return is_valid

    def validateInputData(self, pars_missing):
        '''Returns False if any of the required data for the scan are missing
        (3A AND ID1A3 SAVING SCHEMA).'''

        indexRegex = re.compile(r'\d+')
        is_valid = True
        
        # Find tomography dark field images
        self.tdf_num_imgs = 0
        self.tdf_img_start = None
        self.tdf_data_folder = self.config.get('tdf_data_folder')
        if self.tdf_data_folder == None:
            pars_missing.append('tdf_data_folder')
        else:
            tdf_files = [f for f in os.listdir(self.tdf_data_folder)
                if os.path.isfile(os.path.join(self.tdf_data_folder, f)) and 
                        f.endswith('.tif') and indexRegex.search(f)]
            if len(tdf_files):
                tdf_files.sort()
                # RV give it a default of up to 20 right now
                self.tdf_num_imgs = msnc.getNumFiles(tdf_files, 'dark field image', 20)
                if self.tdf_num_imgs:
                    tdf_img_start = indexRegex.search(tdf_files[0]).group()
                    if tdf_img_start != None:
                        self.tdf_img_start = int(tdf_img_start)
            if not self.tdf_num_imgs or not self.tdf_img_start:
                logging.error('unable to find suitable dark field images')
                is_valid = False
        logging.info(f'tdf_data_folder = {self.tdf_data_folder}')
        logging.debug(f'tdf_num_imgs = {self.tdf_num_imgs}')
        logging.debug(f'tdf_img_start = {self.tdf_img_start}')
        
        # Find tomography bright field images
        self.tbf_num_imgs = 0
        self.tbf_img_start = None
        self.tbf_data_folder = self.config.get('tbf_data_folder')
        if self.tbf_data_folder == None:
            pars_missing.append('tbf_data_folder')
        else:
            tbf_files = [f for f in os.listdir(self.tbf_data_folder)
                if os.path.isfile(os.path.join(self.tbf_data_folder, f)) and 
                        f.endswith('.tif') and indexRegex.search(f)]
            if len(tbf_files):
                tbf_files.sort()
                # RV give it a default of up to 20 right now
                self.tbf_num_imgs = msnc.getNumFiles(tbf_files, 'bright field image', 20)
                if self.tbf_num_imgs:
                    tbf_img_start = indexRegex.search(tbf_files[0]).group()
                    if tbf_img_start != None:
                        self.tbf_img_start = int(tbf_img_start)
            if not self.tbf_num_imgs or not self.tbf_img_start:
                logging.error('unable to find suitable bright field images')
                is_valid = False
        logging.info(f'tbf_data_folder = {self.tbf_data_folder}')
        logging.debug(f'tbf_num_imgs = {self.tbf_num_imgs}')
        logging.debug(f'tbf_img_start = {self.tbf_img_start}')
        
        # Find tomography images and tomography stack parameters
        self.tomo_data_folders = []
        self.tomo_img_starts = []
        self.tomo_ref_heights = []
        if not self.num_tomo_data_sets:
            is_valid = False
        else:
            for i in range(self.num_tomo_data_sets):
                if self.num_tomo_data_sets == 1:
                    if 'tomo_data_folder' in self.config:
                        tomo_data_folder = self.config['tomo_data_folder']
                    else:
                        tomo_data_folder = self.config.get('tomo_data_folder_1')
                    if tomo_data_folder == None:
                        pars_missing.append('tomo_data_folder or tomo_data_folder_1')
                    if 'z_pos' in self.config:
                        tomo_ref_height = self.config['z_pos']
                    else:
                        tomo_ref_height = self.config.get('z_pos_1')
                    if tomo_ref_height == None:
                        pars_missing.append('z_pos or z_pos_1')
                else:
                    tomo_data_folder = self.config.get(f'tomo_data_folder_{i+1}')
                    if tomo_data_folder == None:
                        pars_missing.append(f'tomo_data_folder_{i+1}')
                    tomo_ref_height = self.config.get(f'z_pos_{i+1}')
                    if tomo_ref_height == None:
                        pars_missing.append(f'z_pos_{i+1}')
                if tomo_data_folder == None:
                    is_valid = False
                elif type(tomo_data_folder) != str:
                    logging.error('illegal value for tomo_data_folder')
                    tomo_data_folder = None
                    is_valid = False
                if tomo_ref_height == None:
                    is_valid = False
                elif type(tomo_ref_height) != int and type(tomo_ref_height) != float:
                    logging.error('illegal value for tomo_ref_height')
                    tomo_ref_height = None
                    is_valid = False
                # Set reference heights relative to first set
                logging.info(f'tomo_data_folder = {tomo_data_folder}')
                logging.info(f'tomo_ref_height = {tomo_ref_height}')
                if i and tomo_ref_height and len(self.tomo_ref_heights) and self.tomo_ref_heights[0]:
                    tomo_ref_height -= self.tomo_ref_heights[0]
                tomo_num_imgs = 0
                tomo_img_start = None
                if tomo_data_folder:
                    tomo_files = [f for f in os.listdir(tomo_data_folder)
                        if os.path.isfile(os.path.join(tomo_data_folder, f)) and 
                                f.endswith('.tif') and indexRegex.search(f)]
                    if len(tomo_files):
                        tomo_files.sort()
                        tomo_num_imgs = msnc.getNumFiles(tomo_files, 'tomography image', self.num_thetas)
                        if self.num_thetas != tomo_num_imgs:
                            logging.error(f'Inconsistent number of thetas: num_thetas = {self.num_thetas}' + 
                                    f' and tomo_num_imgs = {tomo_num_imgs}')
                            is_valid = False
                        else:
                            tomo_img_start = indexRegex.search(tomo_files[0]).group()
                        if tomo_img_start != None:
                            tomo_img_start = int(tomo_img_start)
                        else:
                            logging.error('unable to find suitable bright field images')
                            is_valid = False
                logging.debug(f'tomo_num_imgs = {tomo_num_imgs}')
                logging.debug(f'tomo_img_start = {tomo_img_start}')
                self.tomo_data_folders.append(tomo_data_folder)
                self.tomo_ref_heights.append(tomo_ref_height)
                self.tomo_img_starts.append(tomo_img_start)

            # Set origin for reference height at first set
            if self.tomo_ref_heights[0] != None:
                self.tomo_ref_heights[0] = 0.

        # Update config file
        if not msnc.searchConfigFile('config.txt', 'Reduced stack parameters'):
            self.config = msnc.appendConfigFile('config.txt', '# Reduced stack parameters')
        if 'z_ref_1' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'z_ref_1', self.tomo_ref_heights[0])
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'Reduced stack parameters', 
                    f'z_ref_1 = {self.tomo_ref_heights[0]}')
        for i in range(1, self.num_tomo_data_sets):
            if f'z_ref_{i+1}' in self.config:
                self.config = msnc.updateConfigFile('config.txt', f'z_ref_{i+1}', 
                        self.tomo_ref_heights[i])
            else:
                self.config = msnc.addtoConfigFile('config.txt', f'z_ref_{i}', 
                        f'z_ref_{i+1} = {self.tomo_ref_heights[i]}')

        if len(pars_missing) > 0:
            is_valid = False
        
        return is_valid

    def genDark(self):
        """Generate dark field."""
        if not self.tdf_data_folder:
            sys.exit('Invalid dark field image path.')
        if not self.tdf_num_imgs:
            sys.exit('Invalid number of dark field images.')
        if self.tdf_img_start == None:
            sys.exit('Invalid starting index for dark field images.')
        if not self.num_rows or not self.num_columns:
            sys.exit('Invalid detector dimensions.')

        # Load a stack of dark field images
        logging.debug('Loading data to get median dark field...')
        tdf_stack = msnc.loadImageStack(self.tdf_data_folder, self.tdf_img_start,
                self.tdf_num_imgs, 0, [0, self.num_rows], [0, self.num_columns])

        # Take median
        self.tdf = np.median(tdf_stack, axis=0)
        logging.debug('... done!')
        del tdf_stack

        # RV make input of some kind (not always needed)
        tdf_cutoff = 21
        self.tdf[self.tdf > tdf_cutoff] = np.nan
        tdf_mean = np.nanmean(self.tdf)
        logging.debug(f'tdf_cutoff = {tdf_cutoff}')
        logging.debug(f'tdf_mean = {tdf_mean}')
        np.nan_to_num(self.tdf, copy=False, nan=tdf_mean, posinf=tdf_mean, neginf=0.)
        if self.plot_or_save:
            msnc.quickImshow(self.tdf, title='dark field')
        else:
            msnc.quickImsave('dark_field.png', self.tdf)

    def genBright(self):
        """Generate bright field."""
        if not self.tbf_data_folder:
            sys.exit('Invalid bright field image path')
        if not self.tbf_num_imgs:
            sys.exit('Invalid number of bright field images')
        if self.tbf_img_start == None:
            sys.exit('Invalid starting index for bright field images.')
        if not self.num_rows or not self.num_columns:
            sys.exit('Invalid detector dimensions.')

        # Load a stack of bright field images
        logging.debug('Loading data to get median bright field...')
        tbf_stack = msnc.loadImageStack(self.tbf_data_folder, self.tbf_img_start,
                self.tbf_num_imgs, 0, [0, self.num_rows], [0, self.num_columns])

        # Take median
        """Median or mean: It may be best to try the median because of some image 
           artifacts that arise due to crinkles in the upstream kapton tape windows 
           causing some phase contrast images to appear on the detector.
           One thing that also may be useful in a future implementation is to do a 
           brightfield adjustment on EACH frame of the tomo based on a ROI in the 
           corner of the frame where there is no sample but there is the direct X-ray 
           beam because there is frame to frame fluctuations from the incoming beam. 
           We don’t typically account for them but potentially could.
        """
        self.tbf = np.median(tbf_stack, axis=0)
        del tbf_stack

        # Subtract dark field
        if not self.tdf.size:
            sys.exit('Dark field unavailable')
        self.tbf -= self.tdf
        if self.plot_or_save:
            msnc.quickImshow(self.tbf, title='bright field', save_figname='bright_field.png')
        else:
            msnc.quickImsave('bright_field.png', self.tbf)

    def setDectectorBounds(self):
        """Set vertical detector bounds for image stack."""
        if 'x_low' not in self.config or 'x_upp' not in self.config:
            print('\nSelect image bounds from bright field')
        if not self.tbf.size:
            sys.exit('Bright field unavailable')
        tbf_x_sum = np.sum(self.tbf, 1)
        x_low = self.config.get('x_low')
        if x_low == None:
            x_min = 0
            x_max = self.tbf.shape[0]
            while True:
                msnc.quickXyplot(range(x_min, x_max), tbf_x_sum[x_min:x_max], 
                        'sum over theta and y', True)
                zoom_flag = pyip.inputInt('Set lower image bound (0) or zoom in (1)?: ', min=0, max=1)
                if zoom_flag:
                    x_min = pyip.inputInt(f'    Set lower zoom index [{x_min}, {x_max-1}]: ', 
                            min=x_min, max=x_max-1)
                    x_max = pyip.inputInt(f'    Set upper zoom index [{x_min+1}, {x_max}]: ', 
                            min=x_min+1, max=x_max)
                else:
                    x_low = pyip.inputInt(f'    Set lower image bound [{x_min}, {x_max}]: ', 
                            min=x_min, max=x_max)
                    break
        x_upp = self.config.get('x_upp')
        if x_upp == None:
            x_min = x_low+1
            x_max = self.tbf.shape[0]
            while True:
                msnc.quickXyplot(range(x_min, x_max), tbf_x_sum[x_min:x_max], 
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

        self.img_x_bounds = [x_low, x_upp]
        self.img_y_bounds = [0, self.num_columns]
        logging.info(f'img_x_bounds: {self.img_x_bounds}')
        logging.info(f'img_y_bounds: {self.img_y_bounds}')
#        msnc.quickXyplot(range(x_low, x_upp), tbf_x_sum[x_low:x_upp], 'sum over theta and y', True)

        # Update config file
        if not msnc.searchConfigFile('config.txt', 'Reduced stack parameters'):
            self.config = msnc.appendConfigFile('config.txt', '# Reduced stack parameters')
        if 'x_low' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'x_low', x_low)
        else:
            self.config = msnc.addtoConfigFile(
                    'config.txt', 'Reduced stack parameters', f'x_low = {x_low}')
        if 'x_upp' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'x_upp', x_upp)
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'x_low', f'x_upp = {x_upp}')

    def setZoomOrSkip(self):
        """Set zoom and/or theta skip to reduce memory the requirement for the analysis."""
        if 'zoom_perc' in self.config:
            zoom_perc = self.config['zoom_perc']
            if type(zoom_perc) == int or type(zoom_perc) == float:
                zoom_perc = int(zoom_perc)
                if zoom_perc < 1 or zoom_perc > 100:
                    logging.error('skip zoom: illegal value for zoom_perc')
                    zoom_perc = 100
            else:
                logging.error('skip zoom: illegal value for zoom_perc')
                zoom_perc = 100
        else:
            if pyip.inputYesNo('\nDo you want to zoom in to reduce memory requirement (y/n)? ') == 'yes':
                zoom_perc = pyip.inputInt('    Enter zoom percentage [1, 100]: ', min=1, max=100)
            else:
                zoom_perc = 100
        if 'num_theta_skip' in self.config:
            num_theta_skip = int(self.config['num_theta_skip'])
            if type(num_theta_skip) == int or type(num_theta_skip) == float:
                num_theta_skip = int(num_theta_skip)
                if num_theta_skip < 0 or num_theta_skip > self.num_thetas-1:
                    logging.error('skip theta skip: illegal value for num_theta_skip')
                    num_theta_skip = 0
            else:
                logging.error('skip theta skip: illegal value for num_theta_skip')
                num_theta_skip = 0
        else:
            if pyip.inputYesNo('Do you want to skip thetas to reduce memory requirement (y/n)? ') == 'yes':
                num_theta_skip = pyip.inputInt('    Enter the number skip theta interval' + 
                        f' [0, {self.num_thetas-1}]: ', min=0, max=self.num_thetas-1)
            else:
                num_theta_skip = 0
        logging.info(f'zoom_perc = {zoom_perc}')
        logging.info(f'num_theta_skip = {num_theta_skip}')

        # Update config file
        if not msnc.searchConfigFile('config.txt', 'Reduced stack parameters'):
            self.config = msnc.appendConfigFile('config.txt', '# Reduced stack parameters')
        if 'zoom_perc' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'zoom_perc', zoom_perc)
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'Reduced stack parameters', 
                    f'zoom_perc = {zoom_perc}')
        if 'num_theta_skip' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'num_theta_skip', num_theta_skip)
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'zoom_perc', 
                    f'num_theta_skip = {num_theta_skip}')

    def saveTomo(self, basename, stack, i=None):
        """Save tomography fields."""
        zoom_perc = self.config.get('zoom_perc')
        if zoom_perc == None:
            zoom_perc = 100
        if zoom_perc == 100:
            if i or self.num_tomo_data_sets > 1:
                filepath = f'{basename}_fullres_{i+1}.npy'
            else:
                filepath = f'{basename}_fullres.npy'
        else:
            if i or self.num_tomo_data_sets > 1:
                filepath = f'{basename}_{zoom_perc}p_{i+1}.npy'
            else:
                filepath = f'{basename}_{zoom_perc}p.npy'
        logging.info(f'saving {filepath} ...')
        np.save(filepath, stack)
        logging.info('... done!')

    def genTomo(self, save_flag = False):
        """Generate tomography fields."""
        zoom_perc = self.config.get('zoom_perc')
        if zoom_perc == None:
            zoom_perc = 100
        num_theta_skip = self.config.get('num_theta_skip')
        if num_theta_skip == None:
            num_theta_skip = 0
        if not self.tdf.size:
            sys.exit('Dark field unavailable')
        tdf = self.tdf[self.img_x_bounds[0]:self.img_x_bounds[1],
                self.img_y_bounds[0]:self.img_y_bounds[1]]
        if not self.tbf.size:
            sys.exit('Bright field unavailable')
        tbf = self.tbf[self.img_x_bounds[0]:self.img_x_bounds[1],
                self.img_y_bounds[0]:self.img_y_bounds[1]]
        if not self.loaded_tomo_sets.size:
            self.loaded_tomo_sets = np.zeros(self.num_tomo_data_sets, dtype=np.int8)
        for i in range(self.num_tomo_data_sets):
            # Check if stack is already loaded
            if self.loaded_tomo_sets[i]:
                continue

            # Load a stack of tomography images
            t0 = time()
            tomo_stack = msnc.loadImageStack(self.tomo_data_folders[i], self.tomo_img_starts[i],
                    self.num_thetas, num_theta_skip, self.img_x_bounds, self.img_y_bounds)
            tomo_stack = tomo_stack.astype('float64')
            logging.debug(f'loading took {time()-t0:.2f} seconds!')

            # Subtract dark field
            t0 = time()
            with set_numexpr_threads(self.ncore):
                ne.evaluate('tomo_stack-tdf', out=tomo_stack)
            logging.debug(f'subtracting dark field took {time()-t0:.2f} seconds!')

            # Normalize
            t0 = time()
            with set_numexpr_threads(self.ncore):
                ne.evaluate('tomo_stack/tbf', out=tomo_stack, truediv=True)
            logging.debug(f'normalizing took {time()-t0:.2f} seconds!')

            # Remove non-positive values and linearize data
            t0 = time()
            cutoff = 1.e-6
            with set_numexpr_threads(self.ncore):
                ne.evaluate('where(tomo_stack<cutoff, cutoff, tomo_stack)', out=tomo_stack)
            with set_numexpr_threads(self.ncore):
                ne.evaluate('-log(tomo_stack)', out=tomo_stack)
            logging.debug(f'removing non-positive values and linearizing data took {time()-t0:.2f} seconds!')

            # Get rid of nans/infs that may be introduced by normalization
            t0 = time()
#            np.nan_to_num(tomo_stack, copy=False, nan=0., posinf=0., neginf=0.)
            np.where(np.isfinite(tomo_stack), tomo_stack, 0.)
            logging.debug(f'remove nans/infs took {time()-t0:.2f} seconds!')

            # Downsize tomo stack to smaller size
            tomo_stack = tomo_stack.astype('float32')
            if self.plot_or_save:
                msnc.quickImshow(tomo_stack[0,:,:], title=f'red_stack_fullres_{i+1}',
                        save_figname=f'red_stack_fullres_{i+1}.png')
            else:
                msnc.quickImsave(f'red_stack_fullres_{i+1}.png', tomo_stack[0,:,:])
            if zoom_perc != 100:
                t0 = time()
                logging.info(f'zooming in ...')
                tomo_zoom_list = []
                for j in range(tomo_stack.shape[0]):
                    tomo_zoom = spi.zoom(tomo_stack[j,:,:], 0.01*zoom_perc)
                    tomo_zoom_list.append(np.expand_dims(tomo_zoom, 0))
                tomo_stack = np.concatenate([tomo_zoom for tomo_zoom in tomo_zoom_list])
                del tomo_zoom_list
                logging.info('... done!')
                if self.plot_or_save:
                    msnc.quickImshow(tomo_stack[0,:,:], 
                            title=f'red_stack_{zoom_perc}p_{i+1}',
                            save_figname=f'red_stack_{zoom_perc}p_{i+1}.png')
                else:
                    msnc.quickImsave(f'red_stack_{zoom_perc}p_{i+1}.png', tomo_stack[0,:,:])
                logging.debug(f'zooming took {time()-t0:.2f} seconds!')
    
            # Convert tomo_stack from theta,row,column to row,theta,column
            t0 = time()
            tomo_stack = np.swapaxes(tomo_stack, 0, 1)
            logging.debug(f'swap axis took {time()-t0:.2f} seconds!')

            # Save tomo stack to file
            t0 = time()
            if save_flag:
                self.saveTomo('red_stack', tomo_stack, i)
            logging.debug(f'saving stack took {time()-t0:.2f} seconds!')
                
            # Combine stacks
            t0 = time()
            if not self.tomo_sets.size:
                self.tomo_sets = np.zeros((self.num_tomo_data_sets,
                        tomo_stack.shape[0],tomo_stack.shape[1],tomo_stack.shape[2]))
            self.tomo_sets[i,:] = tomo_stack
            self.loaded_tomo_sets[i] = 1
            logging.debug(f'combining stack took {time()-t0:.2f} seconds!')
            del tomo_stack

            # Adjust sample reference height and update config file
            self.tomo_ref_heights[i] += self.img_x_bounds[0]*self.pixel_size
            if f'z_ref_{i+1}' in self.config:
                self.config = msnc.updateConfigFile('config.txt', f'z_ref_{i+1}', 
                        self.tomo_ref_heights[i])
            else:
                sys.exit(f'Unable to update z_ref_{i+1} in config.txt')

        # Update config file
        if not msnc.searchConfigFile('config.txt', 'Analysis progress'):
            self.config = msnc.appendConfigFile('config.txt', '# Analysis progress')
        if 'pre_processor' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'pre_processor', True)
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'Analysis progress', 
                    'pre_processor = True')
        del tdf
        del tbf

    def loadTomo(self, basename, i=None):
        """Load tomography fields."""
        zoom_perc = self.config.get('zoom_perc')
        if zoom_perc == None:
            zoom_perc = 100
        if zoom_perc == 100:
            if i or self.num_tomo_data_sets > 1:
                filepath = f'{basename}_fullres_{i+1}.npy'
            else:
                filepath = f'{basename}_fullres.npy'
        else:
            if i or self.num_tomo_data_sets > 1:
                filepath = f'{basename}_{zoom_perc}p_{i+1}.npy'
            else:
                filepath = f'{basename}_{zoom_perc}p.npy'
        load_flag = False
        if os.path.isfile(filepath):
            load_flag = pyip.inputYesNo(f'\nDo you want to load {filepath} (y/n)? ')
        stack = np.array([])
        if load_flag == 'yes':
            logging.info(f'loading {filepath} ...')
            try:
                stack = np.load(filepath)
            except IOError or ValueError:
                stack = np.array([])
                logging.error(f'error loading {filepath}')
            logging.info('... done!')
        return stack

    def loadTomoSets(self):
        """Load a set of tomography fields."""
        zoom_perc = self.config.get('zoom_perc')
        if zoom_perc == None:
            zoom_perc == 100
        if not self.loaded_tomo_sets.size:
            self.loaded_tomo_sets = np.zeros(self.num_tomo_data_sets, dtype=np.int8)
        for i in range(self.num_tomo_data_sets):
            # Try to load a tomo stack from file
            tomo_stack = self.loadTomo('red_stack', i)
            if tomo_stack.size:
                if zoom_perc == 100:
                    if self.plot_or_save:
                        msnc.quickImshow(tomo_stack[:,0,:], title=f'red_stack_fullres_{i+1}',
                                save_figname=f'red_stack_fullres_{i+1}.png')
                    else:
                        msnc.quickImsave(f'red_stack_fullres_{i+1}.png', tomo_stack[:,0,:])
                else:
                    if self.plot_or_save:
                        msnc.quickImshow(tomo_stack[:,0,:], 
                                title=f'red_stack_{zoom_perc}p_{i+1}',
                                save_figname=f'red_stack_{zoom_perc}p_{i+1}.png')
                    else:
                        msnc.quickImsave(f'red_stack_{zoom_perc}p_{i+1}.png', tomo_stack[:,0,:])
                # Combine stacks
                if not self.tomo_sets.size:
                    self.tomo_sets = np.zeros((self.num_tomo_data_sets,
                            tomo_stack.shape[0],tomo_stack.shape[1],tomo_stack.shape[2]))
                self.tomo_sets[i,:] = tomo_stack
                self.loaded_tomo_sets[i] = 1
                del tomo_stack

    def reconstructOnePlane(self, tomo_plane_T, center, plot_sinogram=True):
        """Invert the sinogram for a single tomo plane."""
        # tomo_plane_T index order: column,theta
        assert(center >= 0 and center < tomo_plane_T.shape[0])
        center_offset = center-0.5*tomo_plane_T.shape[0]
        two_offset = 2*int(np.round(center_offset))
        two_offset_abs = np.abs(two_offset)
        max_rad = int(0.5*(self.cross_sectional_dim/self.eff_pixel_size)*1.1) # 10% slack to avoid edge effects
        dist_from_edge = max(1, int(np.floor((tomo_plane_T.shape[0]-two_offset_abs)/2.) - max_rad))
        if two_offset >= 0:
            logging.debug(f'sinogram range = [{two_offset+dist_from_edge}, {-dist_from_edge}]')
            sinogram = tomo_plane_T[two_offset+dist_from_edge:-dist_from_edge,:]
        else:
            logging.debug(f'sinogram range = [{dist_from_edge}, {two_offset-dist_from_edge}]')
            sinogram = tomo_plane_T[dist_from_edge:two_offset-dist_from_edge,:]
        if plot_sinogram:
            if self.plot_or_save:
                msnc.quickImshow(sinogram.T, f'sinogram_center_offset{center_offset:.2f}', aspect='auto')
            else:
                msnc.quickImsave(f'sinogram_center_offset{center_offset:.2f}', sinogram.T, aspect='auto')

        # Inverting sinogram
#        t0 = time()
        recon_sinogram = iradon(sinogram, theta=self.thetas_deg, circle=True)
#        logging.debug(f'inverting sinogram took {time()-t0:.2f} seconds!')
        del sinogram

        # Removing ring artifacts
#        t0 = time()
#        recon_sinogram = filters.gaussian(recon_sinogram, 3.0)
        recon_sinogram = spi.gaussian_filter(recon_sinogram, 0.5)
        recon_clean = np.expand_dims(recon_sinogram, axis=0)
        del recon_sinogram
        recon_clean = tomopy.misc.corr.remove_ring(recon_clean, rwidth=17)
#        logging.debug(f'filtering and removing ring artifact took {time()-t0:.2f} seconds!')
        return recon_clean

    def plotEdgesOnePlane(self, recon_plane, basename, weight=0.001):
        edges = denoise_tv_chambolle(recon_plane, weight = weight)
        figname=f'{basename}.png'
        if self.plot_or_save:
            msnc.quickImshow(edges[0,:,:], f'{basename} (coolwarm scale)',
                    save_figname=figname, cmap='coolwarm')
        else:
            msnc.quickImsave(figname, edges[0,:,:], cmap='coolwarm')
        vmax = np.max(edges[0,:,:])
        vmin = -vmax
        figname=f'{basename}_gray.png'
        if self.plot_or_save:
            msnc.quickImshow(edges[0,:,:], f'{basename} (gray scale)',
                    save_figname=figname, cmap='gray', vmin=vmin, vmax=vmax)
        else:
            msnc.quickImsave(figname, edges[0,:,:], cmap='gray', vmin=vmin, vmax=vmax)
        del edges

    def findCenterOnePlane(self, sinogram, row, tol=0.1):
        """Find center for a single tomo plane."""
        # sinogram index order: theta,column
        # need index order column,theta for iradon, so take transpose
        sinogram_T = sinogram.T
        center = 0.5*sinogram.shape[1]

        # try automatic center finding routines for initial value
        tomo_center = tomopy.find_center_vo(sinogram)
        center_offset = tomo_center-center
        logging.info(f'center at row {row} using Nghia Vo’s method = {center_offset:.2f}')
        recon_plane = self.reconstructOnePlane(sinogram_T, tomo_center, False)
        basename=f'edges_row{row}_center_offset{center_offset:.2f}'
        self.plotEdgesOnePlane(recon_plane, basename)
        tomo_center = tomopy.find_center_pc(sinogram, sinogram, tol=0.1, rotc_guess=tomo_center)
        error = 1.
        while error > tol:
            prev = tomo_center
            tomo_center = tomopy.find_center_pc(sinogram, sinogram, tol=tol, rotc_guess=tomo_center)
            error = np.abs(tomo_center-prev)
        center_offset = tomo_center-center
        logging.info(f'center at row {row} using phase correlation = {center_offset:.2f}')
        recon_plane = self.reconstructOnePlane(sinogram_T, tomo_center, False)
        basename=f'edges_row{row}_center_offset{center_offset:.2f}'
        self.plotEdgesOnePlane(recon_plane, basename)
        if pyip.inputYesNo('\nAccept either one of these centers (y/n)?: ') == 'yes':
            del recon_plane
            return pyip.inputNum('    Enter chosen center offset [0, sinogram.shape[1]): ')

        while True:
            center_offset_low = pyip.inputInt('\nEnter lower bound for center offset ' + 
                    f'[{-int(center)}, {int(center)}]: ', min=-int(center), max=int(center))
            center_offset_upp = pyip.inputInt('Enter upper bound for center offset ' + 
                    f'[{center_offset_low}, {int(center)}]: ', min=center_offset_low, max=int(center))
            if center_offset_upp == center_offset_low:
                center_offset_step = 1
            else:
                center_offset_step = pyip.inputInt('Enter step size for center offset search ' +
                        f'[1, {center_offset_upp-center_offset_low}]: ',
                        min=1, max=center_offset_upp-center_offset_low)
            for center_offset in range(center_offset_low, center_offset_upp+center_offset_step, 
                        center_offset_step):
                logging.info(f'center_offset = {center_offset}')
                recon_plane = self.reconstructOnePlane(sinogram_T, center_offset+center, False)
                basename=f'edges_row{row}_center_offset{center_offset}'
                self.plotEdgesOnePlane(recon_plane, basename)
            if pyip.inputInt('\nContinue (0) or end the search (1): ', min=0, max=1):
                break

        del sinogram_T
        del recon_plane
        return pyip.inputNum('    Enter chosen center offset ' +
                f'[{-int(center)}, {int(center)}]: ', min=-int(center), max=int(center))

    def findCenters(self):
        """Find centers for the tomo stacks."""
        logging.debug('Find centers for tomo stacks')
        if not self.loaded_tomo_sets.size or not np.sum(self.loaded_tomo_sets) or not self.tomo_sets.size:
            sys.exit('Unable to load any tomo sets')
        assert(self.num_tomo_data_sets == self.tomo_sets.shape[0])
        center_set_index = 0
        while self.num_tomo_data_sets > 1:
            center_set_index = pyip.inputInt('\nEnter tomo set index to get center ' +
                    f'[1,{self.num_tomo_data_sets}]: ', min=0, max=self.num_tomo_data_sets)
            center_set_index -= 1
            if not self.loaded_tomo_sets[center_set_index]:
                print('    set not loaded, pick another set')
            else:
                break

        # Get non-overlapping sample row boundaries
        zoom_perc = self.config.get('zoom_perc')
        if zoom_perc == None:
            zoom_perc = 100
        self.eff_pixel_size = 100.*self.pixel_size/zoom_perc
        logging.debug(f'eff_pixel_size = {self.eff_pixel_size}')
        self.tomo_ref_heights = []
        for i in range(self.num_tomo_data_sets):
            if f'z_ref_{i+1}' in self.config:
                self.tomo_ref_heights.append(self.config[f'z_ref_{i+1}'])
            else:
                sys.exit(f'Unable to read z_ref_{i+1} from config.txt')
        if self.num_tomo_data_sets == 1:
            n1 = 0
            n2 = self.tomo_sets.shape[1]
            center_stack = self.tomo_sets[center_set_index,:]
        else:
            n1 = int(0.5*(1. + (self.tomo_ref_heights[0]+
                self.tomo_sets.shape[1]*self.eff_pixel_size-
                self.tomo_ref_heights[1])/self.eff_pixel_size))
            n2 = self.tomo_sets.shape[1]-n1
            center_stack = self.tomo_sets[center_set_index,:]
        logging.info(f'n1 = {n1}, n2 = {n2} (n2-n1) = {(n2-n1)*self.eff_pixel_size:.3f} mm')
        if not center_stack.size:
            sys.exit('Center stack not loaded')
        if self.plot_or_save:
            msnc.quickImshow(center_stack[:,0,:], title=f'center stack at theta = {self.start_theta}')
        else:
            msnc.quickImsave(f'center_stack_theta{self.start_theta:.2f}.png', center_stack[:,0,:])

        # Set thetas (in degrees and radians)
        num_theta_skip = self.config.get('num_theta_skip')
        if num_theta_skip == None:
            num_theta_skip = 0
        self.thetas_deg = np.linspace(self.start_theta, self.end_theta,
                int(self.num_thetas/(num_theta_skip+1)), endpoint=False)
        self.thetas = np.radians(self.thetas_deg)

        # Get cross sectional diameter in mm
        self.cross_sectional_dim = self.config.get('cross_sectional_dim')
        if self.cross_sectional_dim == None:
            self.cross_sectional_dim = center_stack.shape[2]*self.eff_pixel_size
        logging.debug(f'cross_sectional_dim = {self.cross_sectional_dim}')

        # Determine center offset at sample row boundaries
        logging.info('Determine center offset at sample row boundaries')

        # Lower row center
        use_row = False
        use_center = False
        if self.config.get('lower_row'):
            row = int(self.config['lower_row'])
            use_row = pyip.inputYesNo('\nCurrent row index for lower center = '
                    f'{row}, use this value (y/n)? ')
            if use_row:
                if self.config.get('lower_center_offset'):
                    center_offset = int(self.config['lower_center_offset'])
                    use_center = pyip.inputYesNo('Current lower center offset = ' +
                            f'{center_offset}, use this value (y/n)? ')
        if not use_center:
            if not use_row:
                row = pyip.inputInt('\nEnter row index to find lower center ' +
                        f'[{n1}, {n2-2}]: ', min=n1, max=n2-2)
            # center_stack order: row,theta,column
            center_offset = self.findCenterOnePlane(center_stack[row,:,:], row)
        lower_row = row
        lower_center_offset = center_offset
        logging.info(f'lower center offset = {lower_center_offset}')

        # Update config file
        if not ('center_set_index' in self.config or 'lower_row' in self.config or 'upper_row' in self.config or
                'lower_center_offset' in self.config or 'upper_center_offset' in self.config):
            self.config = msnc.appendConfigFile('config.txt', '# Calibration center offset info')
        if 'center_set_index' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'center_set_index', f'{center_set_index+1}')
        else:
            self.config = msnc.addtoConfigFile(
                    'config.txt', 'Calibration center offset info', f'center_set_index = {center_set_index+1}')
        if 'lower_row' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'lower_row', f'{lower_row}')
        else:
            self.config = msnc.addtoConfigFile(
                    'config.txt', 'center_set_index', f'lower_row = {lower_row}')
        if 'lower_center_offset' in self.config:
            self.config = msnc.updateConfigFile(
                    'config.txt', 'lower_center_offset', f'{lower_center_offset}')
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'lower_row',
                    f'lower_center_offset = {lower_center_offset}')

        # Upper row center
        use_row = False
        use_center = False
        if self.config.get('upper_row'):
            row = int(self.config['upper_row'])
            use_row = pyip.inputYesNo('\nCurrent row index for upper center = '
                    f'{row}, use this value (y/n)? ')
            if use_row:
                if self.config.get('upper_center_offset'):
                    center_offset = int(self.config['upper_center_offset'])
                    use_center = pyip.inputYesNo('Current upper center offset = ' +
                            f'{center_offset}, use this value (y/n)? ')
        if not use_center:
            if not use_row:
                row = pyip.inputInt('\nEnter row index to find upper center ' +
                        f'[{lower_row+1}, {n2-1}]: ', min=lower_row+1, max=n2-1)
            # center_stack order: row,theta,column
            center_offset = self.findCenterOnePlane(center_stack[row,:,:], row)
        upper_row = row
        upper_center_offset = center_offset
        logging.info(f'upper center offset = {upper_center_offset}')

        # Update config file
        if 'upper_row' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'upper_row', f'{upper_row}')
        else:
            self.config = msnc.addtoConfigFile(
                    'config.txt', 'lower_center_offset', f'upper_row = {upper_row}')
        if 'upper_center_offset' in self.config:
            self.config = msnc.updateConfigFile(
                    'config.txt', 'upper_center_offset', f'{upper_center_offset}')
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'upper_row',
                    f'upper_center_offset = {upper_center_offset}')
        if 'find_centers' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'find_centers', True)
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'pre_processor', 
                    'find_centers = True')


#%%============================================================================
if __name__ == '__main__':
    # Must specify a config file if running from command line.
    if len(sys.argv) < 2:
#        sys.exit('Please specify a config file to configure tomography analyses.')
        filepath = 'config.txt'
    else:
        filepath = sys.argv[1]

    tomo = Tomo(filepath=filepath)
    if not tomo.is_valid:
        sys.exit('Invalid config and/or detector file provided.')

    pre_processor_flag = tomo.config.get('pre_processor', False)

    if pre_processor_flag:
        tomo.loadTomoSets()

    if not pre_processor_flag:

#%%============================================================================
#% Check required info for the image directories
#==============================================================================
        pars_missing = []
        is_valid = tomo.validateInputData(pars_missing)
        if len(pars_missing) > 0:
            logging.error(f'missing input item(s): {", ".join(pars_missing)}')
        if not tomo.is_valid:
            sys.exit('Invalid image directory config.')

#%%============================================================================
#% Generate dark field
#==============================================================================
        tomo.genDark()

#%%============================================================================
#% Generate bright field
#==============================================================================
        tomo.genBright()

#%%============================================================================
#% Set vertical detector bounds for image stack
#==============================================================================
        tomo.setDectectorBounds()

#%%============================================================================
#% Set zoom and/or theta skip to reduce memory the requirement
#==============================================================================
        tomo.setZoomOrSkip()

#%%============================================================================
#% Generate tomography fields
#==============================================================================
        tomo.genTomo(save_flag = True)

#%%============================================================================
#% Find centers
#==============================================================================
    if not tomo.config.get('find_centers', False):
        tomo.findCenters()




#%%============================================================================
    input('Press any key to continue')
#%%============================================================================

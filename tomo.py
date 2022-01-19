# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:54:37 2021

@author: rv43
"""

import logging
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s %(message)s')
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s %(message)s')

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
        self.pixel_size = None
        self.img_x_bounds = None
        self.img_y_bounds = None
        self.tdf_data_folder = None
        self.tdf_img_start = None
        self.tdf_num_imgs = 0
        self.tdf = np.array([])
        self.tbf_data_folder = None
        self.tbf_img_start = None
        self.tbf_num_imgs = 0
        self.tbf = np.array([])
        self.tomo_data_folders = []
        self.tomo_data_indices = []
        self.tomo_ref_heights = []
        self.tomo_img_starts = []
        self.tomo_sets = np.array([])
        self.loaded_tomo_sets = np.array([])
        self.save_plots = True
        self.save_plots_only = True

        # Load config file 
        if not filepath is None:
           self._loadConfigFile(filepath)
        elif not config_dict is None:
           self._loadConfigDict(config_dict)
        else:
           pass

        # Check config info
        pars_missing = []
        self.is_valid = self._validateConfig(pars_missing)
        if len(pars_missing) > 0:
            logging.error(f'missing item(s) in config file: {", ".join(pars_missing)}')

        # Load detector file
        self._loadDetectorFile()

        # Check detector info
        pars_missing = []
        is_valid = self._validateDetector(pars_missing)
        if len(pars_missing) > 0:
            logging.error(f'missing item(s) in detector file: {", ".join(pars_missing)}')
        self.is_valid = is_valid and self.is_valid

        # Check required info for the image directories
        pars_missing = []
        is_valid = self._validateInputData(pars_missing)
        if len(pars_missing) > 0:
            logging.error(f'missing input item(s): {", ".join(pars_missing)}')
        self.is_valid = is_valid and self.is_valid

    def _loadConfigFile(self, filepath):
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

    def _loadConfigDict(self, config_dict):
        '''Takes a dictionary and places it into self.config.'''

        # Ensure a dictionary was actually supplied
        if type(config_dict) == dict:
            self.config = config_dict
        else:
            logging.error(f'could not pass dictionary config object: {config_dict}')

    def _validateConfig(self, pars_missing):
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
            self.config = msnc.addtoConfigFile('config.txt', 'Scan parameters',
                f'num_thetas = {num_thetas}')
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

    def _loadDetectorFile(self):
        '''looks for a detector yml file and loads it into the dictionary self.detector'''

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

    def _validateDetector(self, pars_missing):
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
                if type(num_rows) != int:
                    logging.error('illegal number of rows in detector info')
                    is_valid = False
            num_columns = pixels.get('columns')
            if not num_columns:
                pars_missing.append('detector:pixels:columns')
            else:
                if type(num_columns) == int:
                    num_columns = num_columns
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
        self.img_x_bounds = [0, num_rows]
        self.img_y_bounds = [0, num_columns]
        logging.debug(f'pixel_size = {self.pixel_size}')
        logging.debug(f'img_x_bounds: {self.img_x_bounds}')
        logging.debug(f'img_y_bounds: {self.img_y_bounds}')

        if len(pars_missing) > 0:
            is_valid = False

        return is_valid

    def _validateInputData(self, pars_missing):
        '''Returns False if any of the required data for the scan are missing
        (3A AND ID1A3 SAVING SCHEMA).'''

        is_valid = True
        
        # Find tomography dark field images folder
        self.tdf_data_folder = self.config.get('tdf_data_folder')
        if self.tdf_data_folder == None:
            pars_missing.append('tdf_data_folder')
        logging.info(f'tdf_data_folder = {self.tdf_data_folder}')
 
        # Find tomography bright field images folder
        self.tbf_data_folder = self.config.get('tbf_data_folder')
        if self.tbf_data_folder == None:
            pars_missing.append('tbf_data_folder')
        logging.info(f'tbf_data_folder = {self.tbf_data_folder}')
        
        # Find tomography images folders and stack parameters
        tomo_data_folders = sorted({key:value for key,value in self.config.items()
                if 'tomo_data_folder' in key}.items())
        if len(tomo_data_folders) != self.num_tomo_data_sets:
            logging.error(f'Incorrect number of tomography data folder names')
            return False
        tomo_ref_heights = sorted({key:value for key,value in self.config.items()
                if 'z_pos' in key}.items())
        if self.num_tomo_data_sets > 1 and len(tomo_ref_heights) != self.num_tomo_data_sets:
            logging.error(f'Incorrect number of tomography reference heights')
            return False
        self.tomo_data_indices = [msnc.tail_int(tomo_data_folders[i][0])-1
                if msnc.tail_int(tomo_data_folders[i][0]) else None
                for i in range(self.num_tomo_data_sets)]
        if len(tomo_ref_heights):
            ref_height_indices = [msnc.tail_int(tomo_ref_heights[i][0])-1
                    if msnc.tail_int(tomo_ref_heights[i][0]) else None
                    for i in range(self.num_tomo_data_sets)]
        else:
            ref_height_indices = [None]*self.num_tomo_data_sets
        if self.tomo_data_indices != ref_height_indices:
            logging.error(f'Incompatible tomography data folder name indices ' +
                    'and reference heights indices')
            return False
        if self.num_tomo_data_sets > 1 and None in self.tomo_data_indices:
                logging.error('Illegal tomography data folder name indices')
                return False
        self.tomo_data_folders = [tomo_data_folders[i][1] for i in range(self.num_tomo_data_sets)]
        if len(tomo_ref_heights):
            self.tomo_ref_heights = [tomo_ref_heights[i][1] for i in range(self.num_tomo_data_sets)]
        for i in range(self.num_tomo_data_sets):
            if type(self.tomo_data_folders[i]) != str:
                logging.error('Illegal tomography data folder name ' +
                        f'{self.tomo_data_folders[i]} {type(self.tomo_data_folders[i])}')
                return False
            if len(self.tomo_ref_heights) and not (type(self.tomo_ref_heights[i]) == int or 
                    type(self.tomo_ref_heights[i]) == float):
                logging.error('Illegal tomography reference height ' +
                        f'{self.tomo_ref_heights[i]} {type(self.tomo_ref_heights[i])}')
                return False
            # Set reference heights relative to first set
            if i:
                self.tomo_ref_heights[i] -= self.tomo_ref_heights[0]
        if len(self.tomo_ref_heights):
            self.tomo_ref_heights[0] = 0.
        else:
            self.tomo_ref_heights = [0.]
        logging.debug(f'tomography data folders: {self.tomo_data_folders}')
        logging.debug(f'tomography data folders indices: {self.tomo_data_indices}')
        logging.debug(f'tomography reference heights: {self.tomo_ref_heights}')

        # Update config file
        if not msnc.searchConfigFile('config.txt', 'Reduced stack parameters'):
            self.config = msnc.appendConfigFile('config.txt', '# Reduced stack parameters')
        key = 'z_ref'
        if self.tomo_data_indices[0] != None:
            key += f'_{1+self.tomo_data_indices[0]}'
        if key in self.config:
            self.config = msnc.updateConfigFile('config.txt', key,
                self.tomo_ref_heights[0])
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'Reduced stack parameters', 
                    f'{key} = {self.tomo_ref_heights[0]}')
        for i in range(1, self.num_tomo_data_sets):
            key = f'z_ref_{1+self.tomo_data_indices[i]}'
            if key in self.config:
                self.config = msnc.updateConfigFile('config.txt', key, 
                        self.tomo_ref_heights[i])
            else:
                self.config = msnc.addtoConfigFile('config.txt', key, 
                        f'{key} = {self.tomo_ref_heights[i]}')

        if len(pars_missing) > 0:
            is_valid = False
        
        return is_valid

    def _getImageFiles(self):
        """Find and check all required image files."""
        is_valid = True

        # Find tomography dark field images
        (self.tdf_img_start, self.tdf_num_imgs) = msnc.selectFiles(
                self.tdf_data_folder, 'dark field')
        if not self.tdf_img_start or not self.tdf_num_imgs:
            logging.error('Unable to find suitable dark field images')
            is_valid = False
        logging.debug(f'tdf_num_imgs = {self.tdf_num_imgs}')
        logging.debug(f'tdf_img_start = {self.tdf_img_start}')

        # Find tomography bright field images
        (self.tbf_img_start, self.tbf_num_imgs) = msnc.selectFiles(
                self.tbf_data_folder, 'bright field')
        if not self.tbf_img_start or not self.tbf_num_imgs:
            logging.error('Unable to find suitable bright field images')
            is_valid = False
        logging.debug(f'tbf_num_imgs = {self.tbf_num_imgs}')
        logging.debug(f'tbf_img_start = {self.tbf_img_start}')

        # Find tomography images
        for i in range(self.num_tomo_data_sets):
            (tomo_img_start, tomo_num_imgs) = msnc.selectFiles(
                    self.tomo_data_folders[i], 'tomography', self.num_thetas)
            if not tomo_img_start or not tomo_num_imgs:
                logging.error('Unable to find suitable tomography images')
                is_valid = False
            logging.debug(f'tomo_num_imgs = {tomo_num_imgs}')
            logging.debug(f'tomo_img_start = {tomo_img_start}')
            self.tomo_img_starts.append(tomo_img_start)
        logging.debug(f'tomography image start indices: {self.tomo_img_starts}')

        return is_valid


    def _genDark(self):
        """Generate dark field."""
        if not self.tdf_data_folder:
            sys.exit('Invalid dark field image path.')
        if not self.tdf_num_imgs:
            sys.exit('Invalid number of dark field images.')
        if self.tdf_img_start == None:
            sys.exit('Invalid starting index for dark field images.')
        if not self.img_x_bounds or not self.img_y_bounds:
            sys.exit('Invalid detector dimensions.')

        # Load a stack of dark field images
        logging.debug('Loading data to get median dark field...')
        tdf_stack = msnc.loadImageStack(self.tdf_data_folder, self.tdf_img_start,
                self.tdf_num_imgs, 0, self.img_x_bounds, self.img_y_bounds)

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
        msnc.quickImshow(self.tdf, title='dark field', save_fig=self.save_plots,
                save_only=self.save_plots_only)

    def _genBright(self):
        """Generate bright field."""
        if not self.tbf_data_folder:
            sys.exit('Invalid bright field image path')
        if not self.tbf_num_imgs:
            sys.exit('Invalid number of bright field images')
        if self.tbf_img_start == None:
            sys.exit('Invalid starting index for bright field images.')
        if not self.img_x_bounds or not self.img_y_bounds:
            sys.exit('Invalid detector dimensions.')

        # Load a stack of bright field images
        logging.debug('Loading data to get median bright field...')
        tbf_stack = msnc.loadImageStack(self.tbf_data_folder, self.tbf_img_start,
                self.tbf_num_imgs, 0, self.img_x_bounds, self.img_y_bounds)

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
        msnc.quickImshow(self.tbf, title='bright field', save_fig=self.save_plots,
                save_only=self.save_plots_only)

    def _setDetectorBounds(self):
        """Set vertical detector bounds for image stack."""
        # Check reference heights
        if self.pixel_size == None:
            sys.exit('pixel_size unavailable')
        if 'img_x_bounds' not in self.config:
            print('\nSelect image bounds from bright field')
        if not self.tbf.size:
            sys.exit('Bright field unavailable')
        num_x_min = None
        if self.num_tomo_data_sets > 1:
            delta_z = self.tomo_ref_heights[1]-self.tomo_ref_heights[0]
            for i in range(2,self.num_tomo_data_sets):
                delta_z = min(delta_z, self.tomo_ref_heights[i]-self.tomo_ref_heights[i-1])
            logging.debug(f'delta_z = {delta_z}')
            num_x_min = int(delta_z/self.pixel_size)+1
            logging.debug(f'num_x_min = {num_x_min}')
            if num_x_min > self.tbf.shape[0]:
                logging.warning('Image bounds and pixel size prevent seamless stacking')
                num_x_min = self.tbf.shape[0]
        tbf_x_sum = np.sum(self.tbf, 1)
        img_x_bounds = self.config.get('img_x_bounds', [None, None])
        msnc.quickImshow(self.tbf, title='bright field')
        img_x_bounds = msnc.selectArrayBounds(tbf_x_sum, img_x_bounds[0], img_x_bounds[1],
                num_x_min, 'sum over theta and y')
        if img_x_bounds != None:
            self.img_x_bounds = img_x_bounds
        if num_x_min != None and self.img_x_bounds[1]-self.img_x_bounds[0]+1 < num_x_min:
            logging.warning('Image bounds and pixel size prevent seamless stacking')
        logging.debug(f'img_x_bounds: {self.img_x_bounds}')
        msnc.quickPlot(range(img_x_bounds[0], img_x_bounds[1]),
                tbf_x_sum[img_x_bounds[0]:img_x_bounds[1]],
                title='sum over theta and y', save_fig=self.save_plots, save_only=True)

        # Update config file
        if not msnc.searchConfigFile('config.txt', 'Reduced stack parameters'):
            self.config = msnc.appendConfigFile('config.txt', '# Reduced stack parameters')
        if 'img_x_bounds' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'img_x_bounds', img_x_bounds)
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'Reduced stack parameters',
                    f'img_x_bounds = {img_x_bounds}')

    def _setZoomOrSkip(self):
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
            if pyip.inputYesNo(
                    '\nDo you want to zoom in to reduce memory requirement (y/n)? ') == 'yes':
                zoom_perc = pyip.inputInt('    Enter zoom percentage [1, 100]: ',
                        min=1, max=100)
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
            if pyip.inputYesNo(
                    'Do you want to skip thetas to reduce memory requirement (y/n)? ') == 'yes':
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

    def _saveTomo(self, basename, stack, i=None):
        """Save a tomography field."""
        assert(i<self.num_tomo_data_sets)
        zoom_perc = self.config.get('zoom_perc')
        if zoom_perc == None or zoom_perc == 100:
            filepath = f'{basename}_fullres'
        else:
            filepath = f'{basename}_{zoom_perc}p'
        if (i != None and type(i) == int and 0 <= i < len(self.tomo_data_indices)
                and self.tomo_data_indices[i] != None):
            filepath += f'_{1+self.tomo_data_indices[i]}'
        filepath += '.npy'
        logging.info(f'saving {filepath} ...')
        np.save(filepath, stack)
        logging.info('... done!')

    def _loadTomo(self, basename, i=None):
        """Load a tomography field."""
        zoom_perc = self.config.get('zoom_perc')
        if zoom_perc == None or zoom_perc == 100:
            filepath = f'{basename}_fullres'
        else:
            filepath = f'{basename}_{zoom_perc}p'
        if i != None and self.tomo_data_indices[i] != None:
            filepath += f'_{1+self.tomo_data_indices[i]}'
        filepath += '.npy'
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

    def _genTomo(self):
        """Generate tomography fields."""
        if not self.img_x_bounds or not self.img_y_bounds:
            sys.exit('Invalid image dimensions.')
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
            tomo_stack = msnc.loadImageStack(self.tomo_data_folders[i],
                    self.tomo_img_starts[i], self.num_thetas, num_theta_skip,
                    self.img_x_bounds, self.img_y_bounds)
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
            np.where(np.isfinite(tomo_stack), tomo_stack, 0.)
            logging.debug(f'remove nans/infs took {time()-t0:.2f} seconds!')

            # Downsize tomo stack to smaller size
            tomo_stack = tomo_stack.astype('float32')
            if self.tomo_data_indices[i] == None:
                title = 'red stack fullres'
            else:
                title = f'red stack fullres {1+self.tomo_data_indices[i]}'
            msnc.quickImshow(tomo_stack[0,:,:], title=title, save_fig=self.save_plots,
                    save_only=self.save_plots_only)
            if zoom_perc != 100:
                t0 = time()
                logging.info(f'zooming in ...')
                tomo_zoom_list = []
                for j in range(tomo_stack.shape[0]):
                    tomo_zoom = spi.zoom(tomo_stack[j,:,:], 0.01*zoom_perc)
                    tomo_zoom_list.append(np.expand_dims(tomo_zoom, 0))
                    del tomo_zoom
                tomo_stack = np.concatenate([tomo_zoom for tomo_zoom in tomo_zoom_list])
                del tomo_zoom_list
                logging.info('... done!')
                title = f'red stack {zoom_perc}p'
                if self.tomo_data_indices[i] != None:
                    title += f' {1+self.tomo_data_indices[i]}'
                msnc.quickImshow(tomo_stack[0,:,:], title=title, save_fig=self.save_plots,
                    save_only=self.save_plots_only)
                logging.debug(f'zooming took {time()-t0:.2f} seconds!')
    
            # Convert tomo_stack from theta,row,column to row,theta,column
            tomo_stack = np.swapaxes(tomo_stack, 0, 1)

            # Save tomo stack to file
            t0 = time()
            self._saveTomo('red_stack', tomo_stack, i)
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

        # Update config file
        if not msnc.searchConfigFile('config.txt', 'Analysis progress'):
            self.config = msnc.appendConfigFile('config.txt', '# Analysis progress')
        if 'pre_processor' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'pre_processor',
                f'{self.num_tomo_data_sets}')
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'Analysis progress', 
                    f'pre_processor = {self.num_tomo_data_sets}')
        del tdf
        del tbf

    def _reconstructOnePlane(self, tomo_plane_T, center, plot_sinogram=True):
        """Invert the sinogram for a single tomo plane."""
        # tomo_plane_T index order: column,theta
        assert(center >= 0 and center < tomo_plane_T.shape[0])
        center_offset = center-tomo_plane_T.shape[0]/2
        two_offset = 2*int(np.round(center_offset))
        two_offset_abs = np.abs(two_offset)
        max_rad = int(0.5*(self.cross_sectional_dim/self.eff_pixel_size)*1.1) # 10% slack to avoid edge effects
        dist_from_edge = max(1, int(np.floor((tomo_plane_T.shape[0]-two_offset_abs)/2.)-max_rad))
        if two_offset >= 0:
            logging.debug(f'sinogram range = [{two_offset+dist_from_edge}, {-dist_from_edge}]')
            sinogram = tomo_plane_T[two_offset+dist_from_edge:-dist_from_edge,:]
        else:
            logging.debug(f'sinogram range = [{dist_from_edge}, {two_offset-dist_from_edge}]')
            sinogram = tomo_plane_T[dist_from_edge:two_offset-dist_from_edge,:]
        if plot_sinogram:
            msnc.quickImshow(sinogram.T, f'sinogram center offset{center_offset:.2f}',
                    save_fig=self.save_plots, save_only=self.save_plots_only, aspect='auto')

        # Inverting sinogram
        t0 = time()
        recon_sinogram = iradon(sinogram, theta=self.thetas_deg, circle=True)
        logging.debug(f'inverting sinogram took {time()-t0:.2f} seconds!')
        del sinogram

        # Removing ring artifacts
        # RV parameters for the denoise, gaussian, and ring removal will be different for different feature sizes
        t0 = time()
#        recon_sinogram = filters.gaussian(recon_sinogram, 3.0)
        recon_sinogram = spi.gaussian_filter(recon_sinogram, 0.5)
        recon_clean = np.expand_dims(recon_sinogram, axis=0)
        del recon_sinogram
        recon_clean = tomopy.misc.corr.remove_ring(recon_clean, rwidth=17)
        logging.debug(f'filtering and removing ring artifact took {time()-t0:.2f} seconds!')
        return recon_clean

    def _plotEdgesOnePlane(self, recon_plane, basename, weight=0.001):
        # RV parameters for the denoise, gaussian, and ring removal will be different for different feature sizes
        edges = denoise_tv_chambolle(recon_plane, weight = weight)
        vmax = np.max(edges[0,:,:])
        vmin = -vmax
        msnc.quickImshow(edges[0,:,:], f'{basename} coolwarm', save_fig=self.save_plots,
                save_only=self.save_plots_only, cmap='coolwarm')
                #save_only=self.save_plots_only, cmap='coolwarm', vmin=vmin, vmax=vmax)
        msnc.quickImshow(edges[0,:,:], f'{basename} gray', save_fig=self.save_plots,
                save_only=self.save_plots_only, cmap='gray', vmin=vmin, vmax=vmax)
        del edges

    def _findCenterOnePlane(self, sinogram, row, tol=0.1):
        """Find center for a single tomo plane."""
        # sinogram index order: theta,column
        # need index order column,theta for iradon, so take transpose
        sinogram_T = sinogram.T
        center = sinogram.shape[1]/2

        # try automatic center finding routines for initial value
        tomo_center = tomopy.find_center_vo(sinogram)
        center_offset = tomo_center-center
        logging.info(f'center at row {row} using Nghia Vo’s method = {center_offset:.2f}')
        recon_plane = self._reconstructOnePlane(sinogram_T, tomo_center, False)
        basename=f'edges_row{row}_center_offset{center_offset:.2f}'
        self._plotEdgesOnePlane(recon_plane, basename)
        if pyip.inputYesNo('Try finding center using phase correlation (y/n)?: ') == 'yes':
            tomo_center = tomopy.find_center_pc(sinogram, sinogram, tol=0.1,
                    rotc_guess=tomo_center)
            error = 1.
            while error > tol:
                prev = tomo_center
                tomo_center = tomopy.find_center_pc(sinogram, sinogram, tol=tol,
                        rotc_guess=tomo_center)
                error = np.abs(tomo_center-prev)
            center_offset = tomo_center-center
            logging.info(f'center at row {row} using phase correlation = {center_offset:.2f}')
            recon_plane = self._reconstructOnePlane(sinogram_T, tomo_center, False)
            basename=f'edges_row{row}_center_offset{center_offset:.2f}'
            self._plotEdgesOnePlane(recon_plane, basename)
        if pyip.inputYesNo('Accept a center location or continue search (y/n)?: ') == 'yes':
            del sinogram_T
            del recon_plane
            return pyip.inputNum(
                f'    Enter chosen center offset [{-int(center)}, {int(center)}): ')

        while True:
            center_offset_low = pyip.inputInt('\nEnter lower bound for center offset ' + 
                    f'[{-int(center)}, {int(center)}]: ', min=-int(center), max=int(center))
            center_offset_upp = pyip.inputInt('Enter upper bound for center offset ' + 
                    f'[{center_offset_low}, {int(center)}]: ',
                    min=center_offset_low, max=int(center))
            if center_offset_upp == center_offset_low:
                center_offset_step = 1
            else:
                center_offset_step = pyip.inputInt('Enter step size for center offset search ' +
                        f'[1, {center_offset_upp-center_offset_low}]: ',
                        min=1, max=center_offset_upp-center_offset_low)
            for center_offset in range(center_offset_low, center_offset_upp+center_offset_step, 
                        center_offset_step):
                logging.info(f'center_offset = {center_offset}')
                recon_plane = self._reconstructOnePlane(sinogram_T, center_offset+center, False)
                basename=f'edges_row{row}_center_offset{center_offset}'
                self._plotEdgesOnePlane(recon_plane, basename)
            if pyip.inputInt('\nContinue (0) or end the search (1): ', min=0, max=1):
                break

        del sinogram_T
        del recon_plane
        return pyip.inputNum(f'    Enter chosen center offset ' +
                f'[{-int(center)}, {int(center)}]: ', min=-int(center), max=int(center))

    def _reconstructOneTomoSet(self, tomo_stack, thetas, center_offsets, sigma=0.1, ncore=1,
            algorithm='gridrec', run_secondary_sirt=False, secondary_iter=100):
        """reconstruct a single tomo stack."""
        # stack order: row,theta,column
        # thetas must be in radians
        # centers must be an absolute index in the column dimension
        # RV should we remove stripes?
        # https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html
        # RV should we remove rings?
        # https://tomopy.readthedocs.io/en/latest/api/tomopy.misc.corr.html
        if center_offsets.size != tomo_stack.shape[0]:
            sys.exit('tomo_stack row dimension mismatch in reconstructOneTomoSet')
        if thetas.size != tomo_stack.shape[1]:
            sys.exit('tomo_stack row dimension mismatch in reconstructOneTomoSet')
        #tmp = tomopy.prep.stripe.remove_stripe_fw(tomo_stack, sigma=sigma, ncore=ncore)
        centers = center_offsets+tomo_stack.shape[2]/2
        tomo_recon = tomopy.recon(tomo_stack, thetas, centers, sinogram_order=True, 
                algorithm=algorithm, ncore=ncore)
        #if run_secondary_sirt:
        #    options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':secondary_iter}
        #    recon = tomopy.recon(tmp, theta, centers, init_recon=tmp_recon, algorithm=tomopy.astra,
        #            options=options,sinogram_order=False,ncore=ncore)
        #    recons[x] += recon[0]
        #else:
        #    recons[x] += tmp_recon
        #recon_clean[0,:] = tomopy.misc.corr.remove_ring(recons[0,:],rwidth=17)
        return tomo_recon

    def loadTomoSets(self, basename):
        """Load a set of tomography fields."""
        zoom_perc = self.config.get('zoom_perc')
        if not self.loaded_tomo_sets.size:
            self.loaded_tomo_sets = np.zeros(self.num_tomo_data_sets, dtype=np.int8)
        assert(len(self.tomo_data_indices) == self.num_tomo_data_sets)
        print(f'self.tomo_data_indices {self.tomo_data_indices}')
        for i in range(self.num_tomo_data_sets):
            print(f'self.loaded_tomo_sets[{i}] {self.loaded_tomo_sets[i]}')
            if self.loaded_tomo_sets[i]:
                continue
            # Try to load a tomo stack from file
            tomo_stack = self._loadTomo(basename, i)
            print(f'tomo_stack.shape {tomo_stack.shape}')
            if tomo_stack.size:
                if zoom_perc == None or zoom_perc == 100:
                    title = 'red stack fullres'
                else:
                    title = f'red stack {zoom_perc}p'
                if self.tomo_data_indices[i] != None:
                    title += f' {1+self.tomo_data_indices[i]}'
                msnc.quickImshow(tomo_stack[:,0,:], title=title, save_fig=self.save_plots,
                    save_only=self.save_plots_only)
                # Combine stacks
                if not self.tomo_sets.size:
                    self.tomo_sets = np.zeros((self.num_tomo_data_sets,
                            tomo_stack.shape[0],tomo_stack.shape[1],tomo_stack.shape[2]))
                self.tomo_sets[i,:] = tomo_stack
                self.loaded_tomo_sets[i] = 1
                del tomo_stack

    def genTomoSets(self):
        """Generate a set of tomography fields."""

        # Find and check required image files
        self.is_valid = self._getImageFiles()
        if self.is_valid == False:
            return

        # Generate dark field
        self._genDark()

        # Generate bright field
        self._genBright()

        # Set vertical detector bounds for image stack
        self._setDetectorBounds()

        # Set zoom and/or theta skip to reduce memory the requirement
        self._setZoomOrSkip()

        # Generate tomography fields
        self._genTomo()

        # Adjust sample reference height and update config file
        for i in range(self.num_tomo_data_sets):
            self.tomo_ref_heights[i] += self.img_x_bounds[0]*self.pixel_size
            key = 'z_ref'
            if self.tomo_data_indices[i] != None:
                key += f'_{1+self.tomo_data_indices[i]}'
            if key in self.config:
                self.config = msnc.updateConfigFile('config.txt', key, 
                        self.tomo_ref_heights[i])
            else:
                sys.exit(f'Unable to update {key} in config.txt')

    def findCenters(self):
        """Find rotation axis centers for the tomo stacks."""
        logging.debug('Find centers for tomo stacks')
        if (not self.loaded_tomo_sets.size or not np.sum(self.loaded_tomo_sets) or 
                not self.tomo_sets.size or self.tomo_sets.shape[0] != self.num_tomo_data_sets):
            sys.exit('Unable to load any tomo sets')
        center_set_index = 0
        while self.num_tomo_data_sets > 1:
            center_set_index = pyip.inputInt(
                    '\nEnter tomo set index to get rotation axis centers ' +
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
            key = 'z_ref'
            if self.tomo_data_indices[i] != None:
                key += f'_{1+self.tomo_data_indices[i]}'
            if key in self.config:
                self.tomo_ref_heights.append(self.config[key])
            else:
                sys.exit(f'Unable to read {key} from config.txt')
        if self.num_tomo_data_sets == 1:
            n1 = 0
            n2 = self.tomo_sets.shape[1]
            center_stack = self.tomo_sets[center_set_index,:]
        else:
            n1 = int((1. + (self.tomo_ref_heights[0]+
                self.tomo_sets.shape[1]*self.eff_pixel_size-
                self.tomo_ref_heights[1])/self.eff_pixel_size)/2)
            n2 = self.tomo_sets.shape[1]-n1
            center_stack = self.tomo_sets[center_set_index,:]
        logging.info(f'n1 = {n1}, n2 = {n2} (n2-n1) = {(n2-n1)*self.eff_pixel_size:.3f} mm')
        if not center_stack.size:
            sys.exit('Center stack not loaded')
        msnc.quickImshow(center_stack[:,0,:], title=f'center stack theta={self.start_theta}',
                save_fig=self.save_plots, save_only=self.save_plots_only)

        # Set thetas (in degrees)
        num_theta_skip = self.config.get('num_theta_skip')
        if num_theta_skip == None:
            num_theta_skip = 0
        self.thetas_deg = np.linspace(self.start_theta, self.end_theta,
                int(self.num_thetas/(num_theta_skip+1)), endpoint=False)

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
                    center_offset = self.config['lower_center_offset']
                    if type(center_offset) == int or type(center_offset) == float:
                        use_center = pyip.inputYesNo('Current lower center offset = ' +
                                f'{center_offset}, use this value (y/n)? ')
        if not use_center:
            if not use_row:
                row = pyip.inputInt('\nEnter row index to find lower center ' +
                        f'[{n1}, {n2-2}]: ', min=n1, max=n2-2)
            # center_stack order: row,theta,column
            center_offset = self._findCenterOnePlane(center_stack[row,:,:], row)
        lower_row = row
        lower_center_offset = center_offset
        logging.info(f'lower center offset = {lower_center_offset}')

        # Update config file
        if not ('center_set_index' in self.config or 'lower_row' in self.config or
                'upper_row' in self.config or 'lower_center_offset' in self.config or
                'upper_center_offset' in self.config):
            self.config = msnc.appendConfigFile('config.txt', '# Calibration center offset info')
        if 'center_set_index' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'center_set_index',
                    f'{center_set_index+1}')
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'Calibration center offset info',
                    f'center_set_index = {center_set_index+1}')
        if 'row_bounds' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'row_bounds', f'[{n1}, {n2}]')
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'center_set_index',
                    f'row_bounds = [{n1}, {n2}]')
        if 'lower_row' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'lower_row', f'{lower_row}')
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'row_bounds',
                    f'lower_row = {lower_row}')
        if 'lower_center_offset' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'lower_center_offset',
                    f'{lower_center_offset}')
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
                    if type(center_offset) == int or type(center_offset) == float:
                        use_center = pyip.inputYesNo('Current upper center offset = ' +
                                f'{center_offset}, use this value (y/n)? ')
        if not use_center:
            if not use_row:
                row = pyip.inputInt('\nEnter row index to find upper center ' +
                        f'[{lower_row+1}, {n2-1}]: ', min=lower_row+1, max=n2-1)
            # center_stack order: row,theta,column
            center_offset = self._findCenterOnePlane(center_stack[row,:,:], row)
        upper_row = row
        upper_center_offset = center_offset
        logging.info(f'upper center offset = {upper_center_offset}')
        del center_stack

        # Update config file
        if 'upper_row' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'upper_row', f'{upper_row}')
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'lower_center_offset',
                    f'upper_row = {upper_row}')
        if 'upper_center_offset' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'upper_center_offset',
                    f'{upper_center_offset}')
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'upper_row',
                    f'upper_center_offset = {upper_center_offset}')
        if 'find_centers' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'find_centers', True)
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'pre_processor', 
                    'find_centers = True')

    def checkCenters(self):
        """Check centers for the tomo stacks."""
        logging.debug('Check centers for tomo stacks')
        # (Re)load set of tomography fields
        if (not self.loaded_tomo_sets.size or not np.sum(self.loaded_tomo_sets) or
                not self.tomo_sets.size or self.tomo_sets.shape[0] != self.num_tomo_data_sets):
            sys.exit('Unable to load any tomo sets')

        center_set_index = self.config.get('center_set_index')
        if center_set_index == None:
            sys.exit('Unable to read center_set_index from config')
        center_set_index -= 1
        lower_row = self.config.get('lower_row')
        if lower_row == None:
            sys.exit('Unable to read lower_row from config')
        lower_center_offset = self.config.get('lower_center_offset')
        if lower_center_offset == None:
            sys.exit('Unable to read lower_center_offset from config')
        upper_row = self.config.get('upper_row')
        if upper_row == None:
            sys.exit('Unable to read upper_row from config')
        upper_center_offset = self.config.get('upper_center_offset')
        if upper_center_offset == None:
            sys.exit('Unable to read upper_center_offset from config')
        assert(lower_row>=0 and lower_row<self.tomo_sets.shape[1]-1)
        assert(upper_row>lower_row and upper_row<self.tomo_sets.shape[1])
        center_slope = (upper_center_offset-lower_center_offset)/(upper_row-lower_row)
        shift = upper_center_offset-lower_center_offset
        if lower_row == 0:
            logging.warning(f'lower_row == 0: one row offset between both planes')
        else:
            lower_row -= 1
            lower_center_offset -= center_slope

        # stack order: stack,row,theta,column
        if center_set_index:
            set1 = center_set_index-1
            set2 = center_set_index
            if not self.loaded_tomo_sets[set1]:
                logging.error(f'Unable to load required tomo set {set1}')
            elif not self.loaded_tomo_sets[set2]:
                logging.error(f'Unable to load required tomo set {set1}')
            else:
                plane1 = self.tomo_sets[set1,upper_row,:]
                plane2 = self.tomo_sets[set2,lower_row,:]
                for i in range(-2, 3):
                    shift_i = shift+2*i
                    plane1_shifted = spi.shift(plane2, [0, shift_i])
                    msnc.quickPlot((plane1[0,:],), (plane1_shifted[0,:],),
                            title=f'sets {set1} {set2} shifted {2*i} theta={self.start_theta}',
                            save_fig=self.save_plots, save_only=self.save_plots_only)
        if center_set_index < self.num_tomo_data_sets-1:
            set1 = center_set_index
            set2 = center_set_index+1
            if not self.loaded_tomo_sets[set1]:
                logging.error(f'Unable to load required tomo set {set1}')
            elif not self.loaded_tomo_sets[set2]:
                logging.error(f'Unable to load required tomo set {set2}')
            else:
                plane1 = self.tomo_sets[set1,upper_row,:]
                plane2 = self.tomo_sets[set2,lower_row,:]
                for i in range(-2, 3):
                    shift_i = -shift+2*i
                    plane1_shifted = spi.shift(plane2, [0, shift_i])
                    msnc.quickPlot((plane1[0,:],), (plane1_shifted[0,:],), 
                            title=f'sets {set1} {set2} shifted {2*i} theta={self.start_theta}',
                            save_fig=self.save_plots, save_only=self.save_plots_only)
        del plane1, plane2, plane1_shifted

        # Update config file
        if 'check_centers' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'check_centers', True)
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'find_centers', 
                    'check_centers = True')

    def reconstructTomoSets(self):
        """Combine and reconstruct tomo stacks."""
        # stack order: stack,row,theta,column
        if self.num_tomo_data_sets != self.loaded_tomo_sets.sum():
            logging.error('Unable to load all tomo sets')
            return
        center = self.tomo_sets.shape[3]/2

        # Get center stack boundaries and rotation axis centers
        row_bounds = self.config.get('row_bounds')
        if row_bounds == None or type(row_bounds) != list or len(row_bounds) != 2:
            logging.error('Unable to read row_bounds from config')
            return
        lower_row = self.config.get('lower_row')
        if lower_row == None:
            logging.error('Unable to read lower_row from config')
            return
        lower_center_offset = self.config.get('lower_center_offset')
        if lower_center_offset == None:
            logging.error('Unable to read lower_center_offset from config')
            return
        upper_row = self.config.get('upper_row')
        if upper_row == None:
            logging.error('Unable to read upper_row from config')
            return
        upper_center_offset = self.config.get('upper_center_offset')
        if upper_center_offset == None:
            logging.error('Unable to read upper_center_offset from config')
            return
        logging.debug(f'row_bounds = {row_bounds} N = {row_bounds[1]-row_bounds[0]}')
        logging.debug(f'lower_row = {lower_row} upper_row = {upper_row}')
        assert(row_bounds[0]>=0 and row_bounds[0]<self.tomo_sets.shape[1])
        assert(row_bounds[1]>0 and row_bounds[1]<=self.tomo_sets.shape[1])
        assert(lower_row>=row_bounds[0] and lower_row<row_bounds[1]-1)
        assert(upper_row>row_bounds[0] and upper_row<row_bounds[1])
        center_slope = (upper_center_offset-lower_center_offset)/(upper_row-lower_row)

        # Set thetas (in radians)
        num_theta_skip = self.config.get('num_theta_skip')
        if num_theta_skip == None:
            num_theta_skip = 0
        thetas = np.radians(np.linspace(self.start_theta, self.end_theta,
                int(self.num_thetas/(num_theta_skip+1)), endpoint=False))

        # Reconstruct tomo sets
        tomo_recon_list = []
        zoom_perc = self.config.get('zoom_perc')
        if zoom_perc == None:
            zoom_perc = 100
        if self.num_tomo_data_sets == 1:
            upp_bound = self.tomo_sets.shape[1]
        else:
            upp_bound = row_bounds[1]
        center_offsets = np.linspace(lower_center_offset-lower_row*center_slope,
                upper_center_offset+(upp_bound-1-upper_row)*center_slope,
                upp_bound, endpoint=False)
        if zoom_perc == 100:
            basetitle = 'recon stack full'
        else:
            basetitle = f'recon stack {zoom_perc}p'
        if self.tomo_data_indices[0] == None:
            title = basetitle
        else:
            title = f'{basetitle} {1+self.tomo_data_indices[0]}'
        msnc.quickImshow(self.tomo_sets[0,:upp_bound,0,:], title=title,
                save_fig=self.save_plots, save_only=self.save_plots_only)
        t0 = time()
        tomo_recon = self._reconstructOneTomoSet(self.tomo_sets[0,:upp_bound,:,:],
                thetas, center_offsets, sigma=0.1, ncore=self.ncore,
                algorithm='gridrec', run_secondary_sirt=False, secondary_iter=50)
        logging.info(f'tomo recon took {time()-t0:.2f} seconds!')
        row_slice = int(upp_bound/2) 
        title += f' slice{row_slice}'
        msnc.quickImshow(tomo_recon[row_slice,:,:], title=title,
                save_fig=self.save_plots, save_only=self.save_plots_only)
        msnc.quickPlot(tomo_recon[row_slice,int(tomo_recon.shape[2]/2),:],
                title=f'{title} cut{int(tomo_recon.shape[2]/2)}',
                save_fig=self.save_plots, save_only=self.save_plots_only)
        tomo_recon_list.append(tomo_recon)
        del tomo_recon
        if self.num_tomo_data_sets > 2:
            for i in range(1, self.num_tomo_data_sets-1):
                center_offsets = np.linspace(
                        lower_center_offset+(row_bounds[0]-lower_row)*center_slope,
                        upper_center_offset+(row_bounds[1]-1-upper_row)*center_slope,
                        row_bounds[1]-row_bounds[0], endpoint=False)
                title = f'{basetitle} {1+self.tomo_data_indices[i]}'
                msnc.quickImshow(self.tomo_sets[i,row_bounds[0]:row_bounds[1],0,:],
                        title=title, save_fig=self.save_plots,
                        save_only=self.save_plots_only)
                t0 = time()
                tomo_recon = self._reconstructOneTomoSet(
                        self.tomo_sets[i,row_bounds[0]:row_bounds[1],:,:], thetas,
                        center_offsets, sigma=0.1, ncore=self.ncore, algorithm='gridrec', 
                        run_secondary_sirt=False, secondary_iter=50)
                logging.info(f'tomo recon took {time()-t0:.2f} seconds!')
                row_slice = int((row_bounds[0]+row_bounds[1])/2) 
                title += f' slice{row_slice}'
                msnc.quickImshow(tomo_recon[row_slice,:,:], title=title,
                        save_fig=self.save_plots, save_only=self.save_plots_only)
                msnc.quickPlot(tomo_recon[row_slice,int(tomo_recon.shape[2]/2),:],
                        title=f'{title} cut{int(tomo_recon.shape[2]/2)}',
                        save_fig=self.save_plots, save_only=self.save_plots_only)
                tomo_recon_list.append(tomo_recon)
                del tomo_recon
        if self.num_tomo_data_sets > 1:
            center_offsets = np.linspace(
                    lower_center_offset+(row_bounds[0]-lower_row)*center_slope,
                    upper_center_offset+(self.tomo_sets.shape[1]-1-upper_row)*center_slope,
                    self.tomo_sets.shape[1]-row_bounds[0], endpoint=False)
            title = f'{basetitle} {1+self.tomo_data_indices[self.num_tomo_data_sets-1]}'
            msnc.quickImshow(self.tomo_sets[self.num_tomo_data_sets-1,
                    row_bounds[0]:self.tomo_sets.shape[1],0,:], title=title,
                    save_fig=self.save_plots, save_only=self.save_plots_only)
            t0 = time()
            tomo_recon = self._reconstructOneTomoSet(self.tomo_sets[self.num_tomo_data_sets-1,
                    row_bounds[0]:self.tomo_sets.shape[1],:,:], thetas,
                    center_offsets, sigma=0.1, ncore=self.ncore, algorithm='gridrec', 
                    run_secondary_sirt=False, secondary_iter=50)
            logging.info(f'tomo recon took {time()-t0:.2f} seconds!')
            row_slice = int((row_bounds[0]+self.tomo_sets.shape[1])/2) 
            title += f' slice{row_slice}'
            msnc.quickImshow(tomo_recon[row_slice,:,:], title=title,
                    save_fig=self.save_plots, save_only=self.save_plots_only)
            msnc.quickPlot(tomo_recon[row_slice,int(tomo_recon.shape[2]/2),:],
                    title=f'{title} cut{int(tomo_recon.shape[2]/2)}',
                    save_fig=self.save_plots, save_only=self.save_plots_only)
            tomo_recon_list.append(tomo_recon)
            del tomo_recon

        # Combine tomo sets
        tomosum = 0
        [tomosum := tomosum+np.sum(tomo_recon, axis=(0,2)) for tomo_recon in tomo_recon_list]
        msnc.quickPlot(tomosum, title='tomo recon stack sum xz')
        if pyip.inputYesNo(
                '\nDo you want to change the image y-bounds (y/n)? ') == 'no':
            yBounds = [0, tomo_recon_list[0].shape[1]]
        else:
            yBounds = msnc.selectArrayBounds(tomosum, title='tomo recon stack sum xz')
        tomosum = 0
        [tomosum := tomosum+np.sum(tomo_recon, axis=(0,1)) for tomo_recon in tomo_recon_list]
        msnc.quickPlot(tomosum, title='tomo recon stack sum xy')
        if pyip.inputYesNo(
                '\nDo you want to change the image z-bounds (y/n)? ') == 'no':
            zBounds = [0, tomo_recon_list[0].shape[2]]
        else:
            zBounds = msnc.selectArrayBounds(tomosum, title='tomo recon stack sum xy')
        logging.info(f'combining reconstructed sets ...')
        t0 = time()
        tomo_recon_stack = np.concatenate([
                tomo_recon[:,yBounds[0]:yBounds[1],zBounds[0]:zBounds[1]]
                for tomo_recon in tomo_recon_list])
        logging.info(f'... done in {time()-t0:.2f} seconds!')
        print(f'tomo_recon_list[0].shape: {tomo_recon_list[0].shape}')
        print(f'tomo_recon_stack.shape: {tomo_recon_stack.shape}')
        del tomo_recon_list
        tomosum = np.sum(tomo_recon_stack, axis=(1,2))
        msnc.quickPlot(tomosum, title='tomo recon stack sum yz')
        if pyip.inputYesNo(
                '\nDo you want to change the image x-bounds (y/n)? ') == 'no':
            xBounds = [0, tomo_recon_stack.shape[0]]
        else:
            xBounds = msnc.selectArrayBounds(tomosum, title='tomo recon stack sum yz')
            if xBounds[0] != 0 or xBounds[1] != tomo_recon_stack.shape[0]:
                tomo_recon_stack = tomo_recon_stack[xBounds[0]:xBounds[1],:,:]
        logging.info(f'resized tomo_recon_stack shape = {tomo_recon_stack.shape}')
        msnc.quickImshow(tomo_recon_stack[int(tomo_recon_stack.shape[0]/2),:,:],
                title=f'tomo recon stack xslice{int(tomo_recon_stack.shape[0]/2)}',
                save_fig=self.save_plots, save_only=self.save_plots_only)
        msnc.quickImshow(tomo_recon_stack[:,int(tomo_recon_stack.shape[1]/2),:],
                title=f'tomo recon stack yslice{int(tomo_recon_stack.shape[1]/2)}',
                save_fig=self.save_plots, save_only=self.save_plots_only)
        msnc.quickImshow(tomo_recon_stack[:,:,int(tomo_recon_stack.shape[2]/2)],
                title=f'tomo recon stack zslice{int(tomo_recon_stack.shape[2]/2)}',
                save_fig=self.save_plots, save_only=self.save_plots_only)

        # Save tomo sets
        if zoom_perc == 100:
            filepath = f'recon_stack_fullres.npy'
        else:
            filepath = f'recon_stack_{zoom_perc}p.npy'
        logging.info(f'saving {filepath} ...')
        t0 = time()
        np.save(filepath, tomo_recon_stack)
        logging.info(f'... done in {time()-t0:.2f} seconds!')

        if 'combine_sets' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'combine_sets', True)
        else:
            self.config = msnc.addtoConfigFile('config.txt', 'find_centers', 
                    'combine_sets= True')

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

#%%============================================================================
#% Preprocess the image files
#==============================================================================
    if tomo.config.get('pre_processor', 0) != tomo.num_tomo_data_sets:
        tomo.loadTomoSets('red_stack')
        tomo.genTomoSets()
        if not tomo.is_valid:
            sys.exit('Unable to load all required image files.')
        if 'find_centers' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'find_centers', False)
        if 'check_centers' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'check_centers', False)
        if 'combine_sets' in self.config:
            self.config = msnc.updateConfigFile('config.txt', 'combine_sets', False)

    print(f"find_centers = {tomo.config.get('find_centers', False)}")
    print(f"check_centers = {tomo.config.get('check_centers', False)}")
    print(f"combine_sets = {tomo.config.get('combine_sets', False)}")

#%%============================================================================
#% Find centers
#==============================================================================
    if not tomo.config.get('find_centers', False):
        tomo.loadTomoSets('red_stack')
        tomo.findCenters()

#%%============================================================================
#% Check centers
#==============================================================================
    if tomo.num_tomo_data_sets > 1 and not tomo.config.get('check_centers', False):
        tomo.loadTomoSets('red_stack')
        tomo.checkCenters()

#%%============================================================================
#% Combine and reconstruct tomography sets
#==============================================================================
    if not tomo.config.get('combine_sets', False):
        tomo.loadTomoSets('red_stack')
        tomo.reconstructTomoSets()


#%%============================================================================
    input('Press any key to continue')
#%%============================================================================

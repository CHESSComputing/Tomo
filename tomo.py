# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:54:37 2021

@author: rv43
"""

import logging
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
stream_handler.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s - %(message)s'))

import os
import sys
import getopt
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
    
    def __init__(self, config_file=None, config_dict=None, output_folder='.',log_level='INFO',
            test_mode=False):
        """Initialize with optional config input file or dictionary"""
        self.ncore = mp.cpu_count()
        self.config = {}
        self.detector = {}
        self.output_folder = output_folder
        self.is_valid = False
        self.start_theta = 0.
        self.end_theta = 180.
        self.num_thetas = None
        self.num_tomo_data_sets = 1
        self.pixel_size = None
        self.img_x_bounds = None
        self.img_y_bounds = None
        self.data_filetype = None
        self.tdf_data_path = None
        self.tdf_img_start = None
        self.tdf_num_imgs = 0
        self.tdf = np.array([])
        self.tbf_data_path = None
        self.tbf_img_start = None
        self.tbf_num_imgs = 0
        self.tbf = np.array([])
        self.tomo_data_paths = []
        self.tomo_data_indices = []
        self.tomo_ref_heights = []
        self.tomo_img_starts = []
        self.tomo_sets = []
        self.tomo_recon_sets = []
        self.save_plots = True
        self.save_plots_only = True
        self.test_mode = test_mode

        # Set log configuration
        logging.basicConfig(level=logging.INFO,
                format='%(asctime)s : %(levelname)s - %(module)s : %(funcName)s - %(message)s',
                handlers=[logging.FileHandler(f'{output_folder}/tomo.log', mode='w'),
                stream_handler])
        if self.test_mode:
            self.save_plots_only = True
            logging.basicConfig(filename=f'{output_folder}/tomo.log', filemode='w',
                    uevel=logging.WARNING)
            logging.warning('Ignoring command line log_level argument in test mode')
        else:
            level = getattr(logging, log_level.upper(), None)
            if not isinstance(level, int):
                raise ValueError(f'Invalid log_level: {log_level}')
            stream_handler.setLevel(level)

        # Load config file 
        if not config_file is None:
           self._loadConfigFile(config_file)
        elif not config_dict is None:
           self._loadConfigDict(config_dict)
        else:
           pass

        # Check config info
        pars_missing = []
        self.is_valid = self._validateConfig(pars_missing)
        if len(pars_missing) > 0:
            logging.error(f'Missing item(s) in config file: {", ".join(pars_missing)}')

        # Load detector file
        self._loadDetectorFile()

        # Check detector info
        pars_missing = []
        is_valid = self._validateDetector(pars_missing)
        if len(pars_missing) > 0:
            logging.error(f'Missing item(s) in detector file: {", ".join(pars_missing)}')
        self.is_valid = is_valid and self.is_valid

        # Check required info for the image directories
        pars_missing = []
        is_valid = self._validateInputData(pars_missing)
        if len(pars_missing) > 0:
            logging.error(f'Missing input item(s): {", ".join(pars_missing)}')
        self.is_valid = is_valid and self.is_valid

        logging.info(f'ncore = {self.ncore}')
        logging.info(f'save_plots = {self.save_plots}')
        logging.info(f'save_plots_only = {self.save_plots_only}')
        logging.info(f'test_mode = {self.test_mode}')

    def _loadConfigFile(self, config_file):
        '''Takes the full/relative path to a text or yml file and loads it into 
        the dictionary self.config'''

        # Ensure config file exists before opening
        if not os.path.isfile(config_file):
            logging.error(f'File does not exist: {config_file}')
            return

        # Load config file
        suffix = Path(config_file).suffix
        if suffix == '.yml' or suffix == '.yaml':
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        elif suffix == '.txt':
            config = msnc.loadConfigFile(config_file)
        else:
            logging.error(f'Illegal config file extension: {suffix}')

        # Make sure yaml could load config file as a dictionary
        if isinstance(config, dict):
            self.config = config
        else:
            logging.error(f'Could not load dictionary from config file: {config_file}')

    def _loadConfigDict(self, config_dict):
        '''Takes a dictionary and places it into self.config.'''

        # Ensure a dictionary was actually supplied
        if isinstance(config, dict):
            self.config = config_dict
        else:
            logging.error(f'Could not pass dictionary config object: {config_dict}')

    def _validateConfig(self, pars_missing):
        '''Returns False if config parameters are missing or other errors are
        present.'''

        is_valid = True

        # Check for required first-level keys
        pars_needed = ['tdf_data_path', 'tbf_data_path', 'detector_id']
        pars_missing.extend([p for p in pars_needed if p not in self.config])
        if len(pars_missing) > 0:
            is_valid = False

        # Check tomo angle (theta) range keys
        start_theta = self.config.get('start_theta', 0.)
        if msnc.is_num(start_theta, 0.):
            self.start_theta = start_theta
        else:
            msnc.illegal_value('start_theta', start_theta, 'config file')
            is_valid = False
        end_theta = self.config.get('end_theta', 180.)
        if msnc.is_num(end_theta, self.start_theta):
            self.end_theta = end_theta
        else:
            msnc.illegal_value('end_theta', end_theta, 'config file')
            is_valid = False
        if 'num_thetas' in self.config:
            num_thetas = self.config['num_thetas']
            if not msnc.is_int(num_thetas, 0):
                msnc.illegal_value('num_thetas', num_thetas, 'config file')
                num_thetas = None
                is_valid = False
        else:
            num_thetas = pyip.inputInt('\nEnter the number of thetas (>0): ', greaterThan=0)
            # Update config file
            self.config = msnc.updateConfigFile('config.txt', 'num_thetas', num_thetas,
                    '# Scan parameters', '# Scan parameters')
        self.num_thetas = num_thetas
        logging.debug(f'start_theta = {self.start_theta}')
        logging.debug(f'end_theta = {self.end_theta}')
        logging.debug(f'num_thetas = {self.num_thetas}')

        # Check number of tomography image stacks
        num_tomo_data_sets = self.config.get('num_tomo_data_sets', 1)
        if not msnc.is_int(num_tomo_data_sets, 1):
            num_tomo_data_sets = None
            msnc.illegal_value('num_tomo_data_sets', num_tomo_data_sets, 'config file')
            is_valid = False
        self.num_tomo_data_sets = num_tomo_data_sets
        logging.info(f'num_tomo_data_sets = {self.num_tomo_data_sets}')
        if self.num_tomo_data_sets:
            self.tomo_sets = [np.array([]) for _ in range(self.num_tomo_data_sets)]
            self.tomo_recon_sets = [np.array([]) for _ in range(self.num_tomo_data_sets)]

        return is_valid

    def _loadDetectorFile(self):
        '''looks for a detector yml file and loads it into the dictionary self.detector'''

        if not self.config.get('detector_id'):
            return

        # Ensure config file exists before opening
        detector_file = self.config['detector_id']+'.yml'
        if not os.path.isfile(detector_file):
            detector_file = self.config['detector_id']+'.yaml'
            if not os.path.isfile(detector_file):
                logging.error(f'File does not exist: {detector_file}')
                return

        # Load detector yml file
        with open(detector_file, 'r') as f:
            detector = yaml.safe_load(f)

        # Make sure yaml could load detector file as a dictionary
        if isinstance(detector, dict):
            self.detector = detector
        else:
            logging.error(f'Could not load dictionary from file: {detector_file}')

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
        if not msnc.is_num(lens_magnification, 0.):
            msnc.illegal_value('lens_magnification', lens_magnification, 'config file')
            lens_magnification = 1.
            is_valid = False
        pixels = self.detector['detector'].get('pixels')
        if not pixels:
            pars_missing.append('detector:pixels')
        else:
            num_rows = pixels.get('rows')
            if not num_rows:
                pars_missing.append('detector:pixels:rows')
            else:
                if not msnc.is_int(num_rows, 0):
                    msnc.illegal_value('rows', num_rows, 'detector file')
                    is_valid = False
            num_columns = pixels.get('columns')
            if not num_columns:
                pars_missing.append('detector:pixels:columns')
            else:
                if not msnc.is_int(num_columns, 0):
                    msnc.illegal_value('columns', num_columns, 'detector file')
                    is_valid = False
            pixel_size = pixels.get('size')
            if not pixel_size:
                pars_missing.append('detector:pixels:size')
            else:
                if msnc.is_num(pixel_size, 0.):
                    self.pixel_size = pixel_size/lens_magnification
                elif isinstance(pixel_size, list):
                    if ((len(pixel_size) > 2) or 
                            (len(pixel_size) == 2 and pixel_size[0] != pixel_size[1])):
                        msnc.illegal_value('pixel size', pixel_size, 'detector file')
                        is_valid = False
                    elif not msnc.is_num(pixel_size[0], 0.):
                        msnc.illegal_value('pixel size', pixel_size, 'detector file')
                        is_valid = False
                    else:
                        self.pixel_size = pixel_size[0]/lens_magnification
                else:
                    msnc.illegal_value('pixel size', pixel_size, 'detector file')
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
        
        # Check data filetype
        self.data_filetype = self.config.get('data_filetype', 'tif')
        if not isinstance(self.data_filetype, str) or (self.data_filetype != 'tif' and
                self.data_filetype != 'h5'):
            msnc.illegal_value('data_filetype', self.data_filetype, 'config file')
            return False
        logging.info(f'data filetype: {self.data_filetype}')

        # Check work_folder
        work_folder = os.path.abspath(self.config.get('work_folder', ''))
        if not os.path.isdir(work_folder):
            msnc.illegal_value('work_folder', work_folder, 'config file')
            is_valid = False

        # Find tomography dark field images file/folder
        self.tdf_data_path = self.config.get('tdf_data_path')
        if self.tdf_data_path == None:
            pass
            #RV FIX pars_missing.append('tdf_data_path')
        else:
            if self.data_filetype == 'h5':
                if isinstance(self.tdf_data_path, str):
                    if not os.path.isabs(self.tdf_data_path):
                        self.tdf_data_path = os.path.abspath(f'{work_folder}/{self.tdf_data_path}')
                else:
                    msnc.illegal_value('tdf_data_path', self.tdf_data_fil, 'config file')
                    is_valid = False
            else:
                if isinstance(self.tdf_data_path, int):
                    self.tdf_data_path = os.path.abspath(f'{work_folder}/{self.tdf_data_path}/nf')
                elif isinstance(self.tdf_data_path, str):
                    if not os.path.isabs(self.tdf_data_path):
                        self.tdf_data_path = os.path.abspath(f'{work_folder}/{self.tdf_data_path}')
                else:
                    msnc.illegal_value('tdf_data_path', self.tdf_data_path, 'config file')
                    is_valid = False
 
        # Find tomography bright field images file/folder
        self.tbf_data_path = self.config.get('tbf_data_path')
        if self.tbf_data_path == None:
            pars_missing.append('tbf_data_path')
        else:
            if self.data_filetype == 'h5':
                if isinstance(self.tbf_data_path, str):
                    if not os.path.isabs(self.tbf_data_path):
                        self.tbf_data_path = os.path.abspath(f'{work_folder}/{self.tbf_data_path}')
                else:
                    msnc.illegal_value('tbf_data_path', self.tbf_data_fil, 'config file')
                    is_valid = False
            else:
                if isinstance(self.tbf_data_path, int):
                    self.tbf_data_path = os.path.abspath(f'{work_folder}/{self.tbf_data_path}/nf')
                elif isinstance(self.tbf_data_path, str):
                    if not os.path.isabs(self.tbf_data_path):
                        self.tbf_data_path = os.path.abspath(f'{work_folder}/{self.tbf_data_path}')
                else:
                    msnc.illegal_value('tbf_data_path', self.tbf_data_path, 'config file')
                    is_valid = False
 
        # Find tomography images file/folders and stack parameters
        tomo_data_paths = sorted({key:value for key,value in self.config.items()
            if 'tomo_data_path' in key}.items())
        if len(tomo_data_paths) != self.num_tomo_data_sets:
            logging.error(f'Incorrect number of tomography data path names in config file')
            return False
        self.tomo_data_paths = [tomo_data_paths[i][1] for i in
                range(self.num_tomo_data_sets)]
        self.tomo_data_indices = [msnc.get_trailing_int(tomo_data_paths[i][0])
                if msnc.get_trailing_int(tomo_data_paths[i][0]) else None
                for i in range(self.num_tomo_data_sets)]
        if self.num_tomo_data_sets > 1 and None in self.tomo_data_indices:
            logging.error('Illegal tomography data file/folder name indices in config file')
            return False
        tomo_ref_heights = sorted({key:value for key,value in self.config.items()
                if 'z_pos' in key}.items())
        if self.num_tomo_data_sets > 1 and len(tomo_ref_heights) != self.num_tomo_data_sets:
            logging.error(f'Incorrect number of tomography reference heights in config file')
            return False
        if len(tomo_ref_heights):
            self.tomo_ref_heights = [tomo_ref_heights[i][1] for i in range(self.num_tomo_data_sets)]
            ref_height_indices = [msnc.get_trailing_int(tomo_ref_heights[i][0])
                    if msnc.get_trailing_int(tomo_ref_heights[i][0]) else None
                    for i in range(self.num_tomo_data_sets)]
        else:
            ref_height_indices = [None]*self.num_tomo_data_sets
        if len(ref_height_indices) > 1 or ref_height_indices[0] != None:
            if self.tomo_data_indices != ref_height_indices:
                logging.error(f'Incompatible tomography data path indices '+
                        'and reference heights indices in config file')
                return False
        for i in range(self.num_tomo_data_sets):
            if self.data_filetype == 'h5':
                if isinstance(self.tomo_data_paths[i], str):
                    if not os.path.isabs(self.tomo_data_paths[i]):
                        self.tomo_data_paths[i] = os.path.abspath(f'{work_folder}/'+
                                f'{self.tomo_data_paths[i]}')
                else:
                    msnc.illegal_value(f'tomo_data_paths[{i+1}]', self.tomo_data_paths[i],
                            'config file')
            else:
                if isinstance(self.tomo_data_paths[i], int):
                    self.tomo_data_paths[i] = os.path.abspath(f'{work_folder}/'+
                            f'{self.tomo_data_paths[i]}/nf')
                elif isinstance(self.tomo_data_paths[i], str):
                    if not os.path.isabs(self.tomo_data_paths[i]):
                        self.tomo_data_paths[i] = os.path.abspath(f'{work_folder}/'+
                                f'{self.tomo_data_paths[i]}')
                else:
                    msnc.illegal_value(f'tomo_data_paths[{i+1}]', self.tomo_data_paths[i],
                            'config file')
                    return False
            if len(self.tomo_ref_heights) and not msnc.is_num(self.tomo_ref_heights[i]):
                logging.error('Illegal tomography reference height in config file'+
                        f'{self.tomo_ref_heights[i]} {type(self.tomo_ref_heights[i])}')
                return False
            # Set reference heights relative to first set
            if i:
                self.tomo_ref_heights[i] -= self.tomo_ref_heights[0]
        if len(self.tomo_ref_heights):
            self.tomo_ref_heights[0] = 0.
        else:
            self.tomo_ref_heights = [0.]
        logging.info(f'dark field images path = {self.tdf_data_path}')
        logging.info(f'bright field images path = {self.tbf_data_path}')
        logging.info('tomography data paths:')
        for i in range(self.num_tomo_data_sets):
            logging.info(f'    {self.tomo_data_paths[i]}')
        logging.info(f'tomography data path indices: {self.tomo_data_indices}')
        logging.info(f'tomography reference heights: {self.tomo_ref_heights}')

        # Update config file
        search_list = ['# Reduced stack parameters', 'z_ref']+[f'z_ref_{index}'
                for index in self.tomo_data_indices if index != None]
        for i in range(self.num_tomo_data_sets):
            key = 'z_ref'
            if self.tomo_data_indices[i] != None:
                key += f'_{self.tomo_data_indices[i]}'
            self.config = msnc.updateConfigFile('config.txt', key, self.tomo_ref_heights[i],
                    search_list[::-1], '# Reduced stack parameters')

        if len(pars_missing) > 0:
            is_valid = False

        return is_valid

    def _getImageFiles(self, available_sets):
        """Find and check all required image files."""
        is_valid = True
        if len(available_sets) != self.num_tomo_data_sets:
            logging.warning('Illegal dimension of available_sets in _getImageFiles '+
                    f'({len(available_sets)}');
            available_sets = [False]*self.num_tomo_data_sets

        # Find tomography dark field images
        self.tdf_img_start = self.config.get('tdf_img_start')
        tdf_img_offset = self.config.get('tdf_img_offset', -1)
        self.tdf_num_imgs = self.config.get('tdf_num_imgs')
        use_input = 'no'
        if (msnc.is_int(self.tdf_img_start, 0) and msnc.is_int(tdf_img_offset) and
               msnc.is_int(self.tdf_num_imgs, 0)):
            if self.test_mode:
                use_input = 'yes'
            else:
                if tdf_img_offset < 0:
                    use_input = pyip.inputYesNo('\nCurrent dark field starting index = '+
                            f'{self.tdf_img_start}, use this value ([y]/n)? ',
                            blank=True)
                else:
                    use_input = pyip.inputYesNo('\nCurrent dark field starting index/offset = '+
                            f'{self.tdf_img_start}/{tdf_img_offset}, use these values ([y]/n)? ',
                            blank=True)
                if use_input != 'no':
                    use_input = pyip.inputYesNo('Current number of dark field images = '+
                            f'{self.tdf_num_imgs}, use this value ([y]/n)? ',
                            blank=True)
        if use_input == 'no':
            self.tdf_img_start, tdf_img_offset, self.tdf_num_imgs = \
                    msnc.selectImageRange(self.tdf_data_path, self.data_filetype, 'dark field')
            if not self.tdf_img_start or not self.tdf_num_imgs:
                logging.error('Unable to find suitable dark field images')
                if self.tdf_data_path != None:
                    is_valid = False
        logging.debug(f'tdf_img_start = {self.tdf_img_start}')
        logging.debug(f'tdf_img_offset = {tdf_img_offset}')
        logging.debug(f'tdf_num_imgs = {self.tdf_num_imgs}')

        # Find tomography bright field images
        self.tbf_img_start = self.config.get('tbf_img_start')
        tbf_img_offset = self.config.get('tbf_img_offset', -1)
        self.tbf_num_imgs = self.config.get('tbf_num_imgs')
        use_input = 'no'
        if (msnc.is_int(self.tbf_img_start, 0) and msnc.is_int(tbf_img_offset) and
               msnc.is_int(self.tbf_num_imgs, 0)):
            if self.test_mode:
                use_input = 'yes'
            else:
                if tbf_img_offset < 0:
                    use_input = pyip.inputYesNo('\nCurrent bright field starting index = '+
                            f'{self.tbf_img_start}, use this value ([y]/n)? ',
                            blank=True)
                else:
                    use_input = pyip.inputYesNo('\nCurrent bright field starting index/offset = '+
                            f'{self.tbf_img_start}/{tbf_img_offset}, use these values ([y]/n)? ',
                            blank=True)
                if use_input != 'no':
                    use_input = pyip.inputYesNo('Current number of bright field images = '+
                            f'{self.tbf_num_imgs}, use this value ([y]/n)? ',
                            blank=True)
        if use_input == 'no':
            self.tbf_img_start, tbf_img_offset, self.tbf_num_imgs = \
                    msnc.selectImageRange(self.tbf_data_path, self.data_filetype, 'bright field')
            if not self.tbf_img_start or not self.tbf_num_imgs:
                logging.error('Unable to find suitable bright field images')
                is_valid = False
        logging.debug(f'tbf_img_start = {self.tbf_img_start}')
        logging.debug(f'tbf_img_offset = {tbf_img_offset}')
        logging.debug(f'tbf_num_imgs = {self.tbf_num_imgs}')

        # Find tomography images
        self.tomo_img_starts = [None]*self.num_tomo_data_sets
        tomo_img_offsets = [-1]*self.num_tomo_data_sets
        for i in range(self.num_tomo_data_sets):
            # Check if stack is already loaded or available
            if self.tomo_sets[i].size or available_sets[i]:
                continue
            key1 = 'tomo_img_start'
            key2 = 'tomo_img_offset'
            if self.tomo_data_indices[i] != None:
                key1 += f'_{self.tomo_data_indices[i]}'
                key2 += f'_{self.tomo_data_indices[i]}'
            self.tomo_img_starts[i] = self.config.get(key1)
            tomo_img_offsets[i] = self.config.get(key2, -1)
            use_input = 'no'
            if msnc.is_int(self.tomo_img_starts[i], 0) and msnc.is_int(tomo_img_offsets[i]):
                if self.test_mode:
                    use_input = 'yes'
                else:
                    if tomo_img_offsets[i] < 0:
                        use_input = pyip.inputYesNo('\nCurrent tomography starting index '+
                                f'for set {i+1} = {self.tomo_img_starts[i]}, '+
                                'use this value ([y]/n)? ', blank=True)
                    else:
                        use_input = pyip.inputYesNo(
                                f'\nCurrent tomography starting index/offset for set {i+1} = '+
                                f'{self.tomo_img_starts[i]}/{tomo_img_offsets[i]}, '+
                                'use these values ([y]/n)? ', blank=True)
            if use_input == 'no':
                name = 'tomography'
                if self.tomo_data_indices[i] != None:
                    name += f' set {self.tomo_data_indices[i]}'
                self.tomo_img_starts[i], tomo_img_offsets[i], tomo_num_imgs = \
                        msnc.selectImageRange(self.tomo_data_paths[i],
                        self.data_filetype, name, self.num_thetas)
                if not self.tomo_img_starts[i] or not tomo_num_imgs:
                    logging.error('Unable to find suitable tomography images')
                    is_valid = False
                logging.debug(f'tomo_num_imgs = {tomo_num_imgs}')
        logging.debug(f'tomography image start indices: {self.tomo_img_starts}')
        logging.debug(f'tomography image offsets: {tomo_img_offsets}')

        # Update config file
        msnc.updateConfigFile('config.txt', 'tdf_img_start', self.tdf_img_start,
                '# Image set info', '# Image set info')
        msnc.updateConfigFile('config.txt', 'tdf_img_offset', tdf_img_offset,
                'tdf_img_start')
        msnc.updateConfigFile('config.txt', 'tdf_num_imgs', self.tdf_num_imgs,
                'tdf_img_offset')
        msnc.updateConfigFile('config.txt', 'tbf_img_start', self.tbf_img_start,
                'tdf_num_imgs')
        msnc.updateConfigFile('config.txt', 'tbf_img_offset', tbf_img_offset,
                'tbf_img_start')
        msnc.updateConfigFile('config.txt', 'tbf_num_imgs', self.tbf_num_imgs,
                'tbf_img_offset')
        search_list = ['tbf_num_imgs', 'tomo_img_start']+[f'tomo_img_start_{index}'
                for index in self.tomo_data_indices if index != None]
        for i in range(self.num_tomo_data_sets):
            if self.tomo_sets[i].size or available_sets[i]:
                continue
            key1 = 'tomo_img_start'
            key2 = 'tomo_img_offset'
            if self.tomo_data_indices[i] != None:
                key1 += f'_{self.tomo_data_indices[i]}'
                key2 += f'_{self.tomo_data_indices[i]}'
            config = msnc.updateConfigFile('config.txt', key1, self.tomo_img_starts[i],
                    search_list[::-1])
            self.config = msnc.updateConfigFile('config.txt', key2, tomo_img_offsets[i],
                    key1)

        return is_valid

    def _genDark(self):
        """Generate dark field."""
        if not self.tdf_data_path:
            logging.error('Invalid dark field image path.')
            return
            raise ValueError('Invalid dark field image path.')
        if not self.tdf_num_imgs:
            raise ValueError('Invalid number of dark field images.')
        if self.tdf_img_start == None:
            raise ValueError('Invalid starting index for dark field images.')
        if not self.img_x_bounds or not self.img_y_bounds:
            raise ValueError('Invalid detector dimensions.')

        # Load a stack of dark field images
        t0 = time()
        logging.debug('Loading data to get median dark field...')
        tdf_stack = msnc.loadImageStack(self.tdf_data_path, self.data_filetype, self.tdf_img_start,
                self.tdf_num_imgs, 0, self.img_x_bounds, self.img_y_bounds)

        # Take median
        self.tdf = np.median(tdf_stack, axis=0)
        logging.debug(f'... done in {time()-t0:.2f} seconds!')
        del tdf_stack

        # RV make input of some kind (not always needed)
        tdf_cutoff = 21
        self.tdf[self.tdf > tdf_cutoff] = np.nan
        tdf_mean = np.nanmean(self.tdf)
        logging.debug(f'tdf_cutoff = {tdf_cutoff}')
        logging.debug(f'tdf_mean = {tdf_mean}')
        np.nan_to_num(self.tdf, copy=False, nan=tdf_mean, posinf=tdf_mean, neginf=0.)
        if not self.test_mode:
            msnc.quickImshow(self.tdf, title='dark field', path=self.output_folder,
                    save_fig=self.save_plots, save_only=self.save_plots_only)

    def _genBright(self):
        """Generate bright field."""
        if not self.tbf_data_path:
            raise ValueError('Invalid bright field image path')
        if not self.tbf_num_imgs:
            raise ValueError('Invalid number of bright field images')
        if self.tbf_img_start == None:
            raise ValueError('Invalid starting index for bright field images.')
        if not self.img_x_bounds or not self.img_y_bounds:
            raise ValueError('Invalid detector dimensions.')

        # Load a stack of bright field images
        logging.debug('Loading data to get median bright field...')
        tbf_stack = msnc.loadImageStack(self.tbf_data_path, self.data_filetype, self.tbf_img_start,
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
        if self.tdf.size:
            self.tbf -= self.tdf
        else:
            logging.error('Invalid dark field image path.')
            #RV FIX raise ValueError('Dark field unavailable')
        if not self.test_mode:
            msnc.quickImshow(self.tbf, title='bright field', path=self.output_folder,
                    save_fig=self.save_plots, save_only=self.save_plots_only)

    def _setDetectorBounds(self):
        """Set vertical detector bounds for image stack."""
        img_x_bounds = self.config.get('img_x_bounds', [None, None])
        if self.test_mode:
            self.img_x_bounds = img_x_bounds
            return

        # Check reference heights
        if self.pixel_size == None:
            raise ValueError('pixel_size unavailable')
        if not self.tbf.size:
            raise ValueError('Bright field unavailable')
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

        # Select image bounds
        # For one tomography set only: load the first image
        if self.num_tomo_data_sets == 1:
            tomo_stack = msnc.loadImageStack(self.tomo_data_paths[0], self.data_filetype,
                    self.tomo_img_starts[0], 1)
            msnc.quickImshow(self.tbf, title='bright field')
            msnc.quickImshow(tomo_stack[0,:,:], title='tomography image at theta = '+
                    f'{self.start_theta}')
            tomo_or_bright = pyip.inputNum('\nSelect image bounds from bright field (0) or '+
                    'from first tomography image (1): ', min=0, max=1)
        else:
            print('\nSelect image bounds from bright field')
            msnc.quickImshow(self.tbf, title='bright field')
            tomo_or_bright = 0
        if self.save_plots_only:
            msnc.clearFig('bright field')
        if tomo_or_bright:
            x_sum = np.sum(tomo_stack[0,:,:], 1)
            use_bounds = 'no'
            if img_x_bounds[0] != None and img_x_bounds[1] != None:
                if img_x_bounds[0] < 0:
                    msnc.illegal_value('img_x_bounds[0]', img_x_bounds[0], 'config file')
                    img_x_bounds[0] = 0
                if not img_x_bounds[0] < img_x_bounds[1] <= x_sum.size:
                    msnc.illegal_value('img_x_bounds[1]', img_x_bounds[1], 'config file')
                    img_x_bounds[1] = x_sum.size
                tomo_tmp = tomo_stack[0,:,:]
                tomo_tmp[img_x_bounds[0],:] = tomo_stack[0,:,:].max()
                tomo_tmp[img_x_bounds[1],:] = tomo_stack[0,:,:].max()
                msnc.quickImshow(tomo_stack[0,:,:], title='tomography image at theta = '+
                        f'{self.start_theta}')
                msnc.quickPlot((range(x_sum.size), x_sum),
                        ([img_x_bounds[0], img_x_bounds[0]], [x_sum.min(), x_sum.max()], 'r-'),
                        ([img_x_bounds[1], img_x_bounds[1]], [x_sum.min(), x_sum.max()], 'r-'),
                        title='sum over theta and y')
                print(f'lower bound = {img_x_bounds[0]} (inclusive)\n'+
                        f'upper bound = {img_x_bounds[1]} (exclusive)]')
                use_bounds =  pyip.inputYesNo('Accept these bounds ([y]/n)?: ', blank=True)
            if use_bounds == 'no':
                img_x_bounds = msnc.selectImageBounds(tomo_stack[0,:,:], 0,
                        img_x_bounds[0], img_x_bounds[1], num_x_min,
                        f'tomography image at theta = {self.start_theta}')
                if num_x_min != None and img_x_bounds[1]-img_x_bounds[0]+1 < num_x_min:
                    logging.warning('Image bounds and pixel size prevent seamless stacking')
                tomo_tmp = tomo_stack[0,:,:]
                tomo_tmp[img_x_bounds[0],:] = tomo_stack[0,:,:].max()
                tomo_tmp[img_x_bounds[1],:] = tomo_stack[0,:,:].max()
                msnc.quickImshow(tomo_stack[0,:,:], title='tomography image at theta = '+
                        f'{self.start_theta}', path=self.output_folder,
                        save_fig=self.save_plots, save_only=True)
                msnc.quickPlot(range(img_x_bounds[0], img_x_bounds[1]),
                        x_sum[img_x_bounds[0]:img_x_bounds[1]],
                        title='sum over theta and y', path=self.output_folder,
                        save_fig=self.save_plots, save_only=True)
        else:
            x_sum = np.sum(self.tbf, 1)
            use_bounds = 'no'
            if img_x_bounds[0] != None and img_x_bounds[1] != None:
                if img_x_bounds[0] < 0:
                    msnc.illegal_value('img_x_bounds[0]', img_x_bounds[0], 'config file')
                    img_x_bounds[0] = 0
                if not img_x_bounds[0] < img_x_bounds[1] <= x_sum.size:
                    msnc.illegal_value('img_x_bounds[1]', img_x_bounds[1], 'config file')
                    img_x_bounds[1] = x_sum.size
                msnc.quickPlot((range(x_sum.size), x_sum),
                        ([img_x_bounds[0], img_x_bounds[0]], [x_sum.min(), x_sum.max()], 'r-'),
                        ([img_x_bounds[1], img_x_bounds[1]], [x_sum.min(), x_sum.max()], 'r-'),
                        title='sum over theta and y')
                print(f'lower bound = {img_x_bounds[0]} (inclusive)\n'+
                        f'upper bound = {img_x_bounds[1]} (exclusive)]')
                use_bounds =  pyip.inputYesNo('Accept these bounds ([y]/n)?: ', blank=True)
            if use_bounds == 'no':
                fit = msnc.fitStep(y=x_sum, model='rectangle', form='atan')
                x_low = fit.get('center1', None)
                x_upp = fit.get('center2', None)
                if x_low != None and x_low >= 0 and x_upp != None and x_low < x_upp < x_sum.size:
                    x_low = int(x_low-(x_upp-x_low)/10)
                    if x_low < 0:
                        x_low = 0
                    x_upp = int(x_upp+(x_upp-x_low)/10)
                    if x_upp >= x_sum.size:
                        x_upp = x_sum.size
                    msnc.quickPlot((range(x_sum.size), x_sum),
                            ([x_low, x_low], [x_sum.min(), x_sum.max()], 'r-'),
                            ([x_upp, x_upp], [x_sum.min(), x_sum.max()], 'r-'),
                            title='sum over theta and y')
                    print(f'lower bound = {x_low} (inclusive)\nupper bound = {x_upp} (exclusive)]')
                    use_fit =  pyip.inputYesNo('Accept these bounds ([y]/n)?: ', blank=True)
                if use_fit == 'no':
                    img_x_bounds = msnc.selectArrayBounds(x_sum, img_x_bounds[0], img_x_bounds[1],
                            num_x_min, 'sum over theta and y')
                else:
                    img_x_bounds = [x_low, x_upp]
                if num_x_min != None and img_x_bounds[1]-img_x_bounds[0]+1 < num_x_min:
                    logging.warning('Image bounds and pixel size prevent seamless stacking')
                msnc.quickPlot(range(img_x_bounds[0], img_x_bounds[1]),
                        x_sum[img_x_bounds[0]:img_x_bounds[1]],
                        title='sum over theta and y', path=self.output_folder,
                        save_fig=self.save_plots, save_only=True)
        self.img_x_bounds = img_x_bounds
        logging.debug(f'img_x_bounds: {self.img_x_bounds}')
        logging.debug(f'img_y_bounds: {self.img_y_bounds}')

        # Update config file
        self.config = msnc.updateConfigFile('config.txt', 'img_x_bounds', img_x_bounds,
                '# Reduced stack parameters', '# Reduced stack parameters')

    def _setZoomOrSkip(self):
        """Set zoom and/or theta skip to reduce memory the requirement for the analysis."""
        if 'zoom_perc' in self.config:
            zoom_perc = self.config['zoom_perc']
            if msnc.is_num(zoom_perc, 1., 100.):
                zoom_perc = int(zoom_perc)
            else:
                msnc.illegal_value('zoom_perc', zoom_perc, 'config file')
                zoom_perc = 100
        else:
            if pyip.inputYesNo(
                    '\nDo you want to zoom in to reduce memory requirement (y/[n])? ',
                    blank=True) == 'yes':
                zoom_perc = pyip.inputInt('    Enter zoom percentage [1, 100]: ',
                        min=1, max=100)
            else:
                zoom_perc = 100
        if 'num_theta_skip' in self.config:
            num_theta_skip = int(self.config['num_theta_skip'])
            if not msnc.is_int(num_theta_skip, 0):
                msnc.illegal_value('num_theta_skip', num_theta_skip, 'config file')
                num_theta_skip = 0
        else:
            if pyip.inputYesNo(
                    'Do you want to skip thetas to reduce memory requirement (y/[n])? ',
                    blank=True) == 'yes':
                num_theta_skip = pyip.inputInt('    Enter the number skip theta interval'+
                        f' [0, {self.num_thetas-1}]: ', min=0, max=self.num_thetas-1)
            else:
                num_theta_skip = 0
        logging.info(f'zoom_perc = {zoom_perc}')
        logging.info(f'num_theta_skip = {num_theta_skip}')

        # Update config file
        msnc.updateConfigFile('config.txt', 'zoom_perc', zoom_perc,
                '# Reduced stack parameters', '# Reduced stack parameters')
        self.config = msnc.updateConfigFile('config.txt', 'num_theta_skip', num_theta_skip,
                'zoom_perc')

    def _saveTomo(self, base_name, stack, i=None):
        """Save a tomography stack."""
        zoom_perc = self.config.get('zoom_perc')
        if zoom_perc == None or zoom_perc == 100:
            title = f'{base_name} fullres'
        else:
            title = f'{base_name} {zoom_perc}p'
        if msnc.is_index(i, 0, len(self.tomo_data_indices)) and self.tomo_data_indices[i] != None:
            title += f'_{self.tomo_data_indices[i]}'
        tomo_file = re.sub(r"\s+", '_', f'{self.output_folder}/{title}.npy')
        t0 = time()
        logging.info(f'Saving {tomo_file} ...')
        np.save(tomo_file, stack)
        logging.info(f'... done in {time()-t0:.2f} seconds!')

    def _loadTomo(self, base_name, i=None, required=False):
        """Load a tomography stack."""
        # stack order: row,theta,column
        zoom_perc = self.config.get('zoom_perc')
        if zoom_perc == None or zoom_perc == 100:
            title = f'{base_name} fullres'
        else:
            title = f'{base_name} {zoom_perc}p'
        if msnc.is_index(i, 0, len(self.tomo_data_indices)) and self.tomo_data_indices[i] != None:
            title += f' {self.tomo_data_indices[i]}'
        tomo_file = re.sub(r"\s+", '_', f'{self.output_folder}/{title}.npy')
        load_flag = 'no'
        available = False
        if os.path.isfile(tomo_file):
            available = True
            if required:
                load_flag = 'yes'
            else:
                load_flag = pyip.inputYesNo(f'\nDo you want to load {tomo_file} (y/n)? ')
        stack = np.array([])
        if load_flag == 'yes':
            t0 = time()
            logging.info(f'Loading {tomo_file} ...')
            try:
                stack = np.load(tomo_file)
            except IOError or ValueError:
                stack = np.array([])
                logging.error(f'Error loading {tomo_file}')
            logging.info(f'... done in {time()-t0:.2f} seconds!')
        if stack.size:
            msnc.quickImshow(stack[:,0,:], title=title, path=self.output_folder,
                    save_fig=self.save_plots, save_only=self.save_plots_only)
        return stack, available

    def _genTomo(self, available_sets):
        """Generate tomography fields."""
        if len(available_sets) != self.num_tomo_data_sets:
            logging.warning('Illegal dimension of available_sets in _genTomo'+
                    f'({len(available_sets)}');
            available_sets = [False]*self.num_tomo_data_sets
        if not self.img_x_bounds or not self.img_y_bounds:
            raise ValueError('Invalid image dimensions.')
        zoom_perc = self.config.get('zoom_perc')
        if zoom_perc == None:
            zoom_perc = 100
        num_theta_skip = self.config.get('num_theta_skip')
        if num_theta_skip == None:
            num_theta_skip = 0
        if self.tdf.size:
            tdf = self.tdf[self.img_x_bounds[0]:self.img_x_bounds[1],
                    self.img_y_bounds[0]:self.img_y_bounds[1]]
        else:
            logging.error('Dark field unavailable')
            #RV FIX raise ValueError('Dark field unavailable')
        if not self.tbf.size:
            raise ValueError('Bright field unavailable')
        tbf = self.tbf[self.img_x_bounds[0]:self.img_x_bounds[1],
                self.img_y_bounds[0]:self.img_y_bounds[1]]
        for i in range(self.num_tomo_data_sets):
            # Check if stack is already loaded or available
            if self.tomo_sets[i].size or available_sets[i]:
                continue

            # Load a stack of tomography images
            t0 = time()
            tomo_stack = msnc.loadImageStack(self.tomo_data_paths[i], self.data_filetype,
                    self.tomo_img_starts[i], self.num_thetas, num_theta_skip,
                    self.img_x_bounds, self.img_y_bounds)
            tomo_stack = tomo_stack.astype('float64')
            logging.debug(f'loading took {time()-t0:.2f} seconds!')

            # Subtract dark field
            if self.tdf.size:
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
            logging.debug('removing non-positive values and linearizing data took '+
                    f'{time()-t0:.2f} seconds!')

            # Get rid of nans/infs that may be introduced by normalization
            t0 = time()
            np.where(np.isfinite(tomo_stack), tomo_stack, 0.)
            logging.debug(f'remove nans/infs took {time()-t0:.2f} seconds!')

            # Downsize tomography stack to smaller size
            tomo_stack = tomo_stack.astype('float32')
            if self.tomo_data_indices[i] == None:
                title = 'red stack fullres'
            else:
                title = f'red stack fullres {self.tomo_data_indices[i]}'
            if not self.test_mode:
                msnc.quickImshow(tomo_stack[0,:,:], title=title, path=self.output_folder,
                        save_fig=self.save_plots, save_only=self.save_plots_only)
            if zoom_perc != 100:
                t0 = time()
                logging.info(f'Zooming in ...')
                tomo_zoom_list = []
                for j in range(tomo_stack.shape[0]):
                    tomo_zoom = spi.zoom(tomo_stack[j,:,:], 0.01*zoom_perc)
                    tomo_zoom_list.append(tomo_zoom)
                tomo_stack = np.stack([tomo_zoom for tomo_zoom in tomo_zoom_list])
                logging.info(f'... done in {time()-t0:.2f} seconds!')
                del tomo_zoom_list
                title = f'red stack {zoom_perc}p'
                if self.tomo_data_indices[i] != None:
                    title += f' {self.tomo_data_indices[i]}'
                if not self.test_mode:
                    msnc.quickImshow(tomo_stack[0,:,:], title=title, path=self.output_folder,
                            save_fig=self.save_plots, save_only=self.save_plots_only)
    
            # Convert tomography stack from theta,row,column to row,theta,column
            tomo_stack = np.swapaxes(tomo_stack, 0, 1)

            # Save tomography stack to file
            if not self.test_mode:
                self._saveTomo('red stack', tomo_stack, i)
            else:
                np.savetxt(self.output_folder+f'red_stack_{self.tomo_data_indices[i]}.txt',
                        tomo_stack[0,:,:], fmt='%.6e')
                
            # Combine stacks
            t0 = time()
            self.tomo_sets[i] = tomo_stack
            logging.debug(f'combining stack took {time()-t0:.2f} seconds!')

        if self.tdf.size:
            del tdf
        del tbf

    def _reconstructOnePlane(self, tomo_plane_T, center, plot_sinogram=True):
        """Invert the sinogram for a single tomography plane."""
        # tomo_plane_T index order: column,theta
        assert(0 <= center < tomo_plane_T.shape[0])
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
                    path=self.output_folder, save_fig=self.save_plots,
                    save_only=self.save_plots_only, aspect='auto')

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

    def _plotEdgesOnePlane(self, recon_plane, base_name, weight=0.001):
        # RV parameters for the denoise, gaussian, and ring removal will be different for different feature sizes
        edges = denoise_tv_chambolle(recon_plane, weight = weight)
        vmax = np.max(edges[0,:,:])
        vmin = -vmax
        msnc.quickImshow(edges[0,:,:], f'{base_name} coolwarm', path=self.output_folder,
                save_fig=self.save_plots, save_only=self.save_plots_only, cmap='coolwarm')
        msnc.quickImshow(edges[0,:,:], f'{base_name} gray', path=self.output_folder,
                save_fig=self.save_plots, save_only=self.save_plots_only, cmap='gray',
                vmin=vmin, vmax=vmax)
        del edges

    def _findCenterOnePlane(self, sinogram, row, tol=0.1):
        """Find center for a single tomography plane."""
        # sinogram index order: theta,column
        # need index order column,theta for iradon, so take transpose
        sinogram_T = sinogram.T
        center = sinogram.shape[1]/2

        # try automatic center finding routines for initial value
        tomo_center = tomopy.find_center_vo(sinogram)
        center_offset_vo = tomo_center-center
        print(f'Center at row {row} using Nghia Vo’s method = {center_offset_vo:.2f}')
        recon_plane = self._reconstructOnePlane(sinogram_T, tomo_center, False)
        base_name=f'edges row{row} center_offset_vo{center_offset_vo:.2f}'
        self._plotEdgesOnePlane(recon_plane, base_name)
        if pyip.inputYesNo('Try finding center using phase correlation (y/[n])? ',
                    blank=True) == 'yes':
            tomo_center = tomopy.find_center_pc(sinogram, sinogram, tol=0.1,
                    rotc_guess=tomo_center)
            error = 1.
            while error > tol:
                prev = tomo_center
                tomo_center = tomopy.find_center_pc(sinogram, sinogram, tol=tol,
                        rotc_guess=tomo_center)
                error = np.abs(tomo_center-prev)
            center_offset = tomo_center-center
            print(f'Center at row {row} using phase correlation = {center_offset:.2f}')
            recon_plane = self._reconstructOnePlane(sinogram_T, tomo_center, False)
            base_name=f'edges row{row} center_offset{center_offset:.2f}'
            self._plotEdgesOnePlane(recon_plane, base_name)
        if pyip.inputYesNo('Accept a center location ([y]) or continue search (n)? ',
                    blank=True) != 'no':
            del sinogram_T
            del recon_plane
            center_offset = pyip.inputNum(
                    f'    Enter chosen center offset [{-int(center)}, {int(center)}] '+
                    f'([{center_offset_vo}])): ', blank=True)
            if center_offset == '':
                center_offset = center_offset_vo
            return center_offset

        while True:
            center_offset_low = pyip.inputInt('\nEnter lower bound for center offset '+
                    f'[{-int(center)}, {int(center)}]: ', min=-int(center), max=int(center))
            center_offset_upp = pyip.inputInt('Enter upper bound for center offset '+
                    f'[{center_offset_low}, {int(center)}]: ',
                    min=center_offset_low, max=int(center))
            if center_offset_upp == center_offset_low:
                center_offset_step = 1
            else:
                center_offset_step = pyip.inputInt('Enter step size for center offset search '+
                        f'[1, {center_offset_upp-center_offset_low}]: ',
                        min=1, max=center_offset_upp-center_offset_low)
            for center_offset in range(center_offset_low, center_offset_upp+center_offset_step, 
                        center_offset_step):
                logging.info(f'center_offset = {center_offset}')
                recon_plane = self._reconstructOnePlane(sinogram_T, center_offset+center, False)
                base_name=f'edges row{row} center_offset{center_offset}'
                self._plotEdgesOnePlane(recon_plane, base_name)
            if pyip.inputInt('\nContinue (0) or end the search (1): ', min=0, max=1):
                break

        del sinogram_T
        del recon_plane
        return pyip.inputNum(f'    Enter chosen center offset '+
                f'[{-int(center)}, {int(center)}]: ', min=-int(center), max=int(center))

    def _reconstructOneTomoSet(self, tomo_stack, thetas, row_bounds=None,
            center_offsets=[], sigma=0.1, rwidth=30, ncore=1, algorithm='gridrec',
            run_secondary_sirt=False, secondary_iter=100):
        """reconstruct a single tomography stack."""
        # stack order: row,theta,column
        # thetas must be in radians
        # centers_offset: tomography axis shift in pixels relative to column center
        # RV should we remove stripes?
        # https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html
        # RV should we remove rings?
        # https://tomopy.readthedocs.io/en/latest/api/tomopy.misc.corr.html
        # RV: Add an option to do (extra) secondary iterations later or to do some sort of convergence test?
        if row_bounds == None:
            row_bounds = [0, tomo_stack.shape[0]]
        else:
            if not (0 <= row_bounds[0] <= row_bounds[1] <= tomo_stack.shape[0]):
                raise ValueError('Illegal row bounds in reconstructOneTomoSet')
        if thetas.size != tomo_stack.shape[1]:
            raise ValueError('theta dimension mismatch in reconstructOneTomoSet')
        if not len(center_offsets):
            centers = np.zeros((row_bounds[1]-row_bounds[0]))
        elif len(center_offsets) == 2:
            centers = np.linspace(center_offsets[0], center_offsets[1],
                    row_bounds[1]-row_bounds[0])
        else:
            if center_offsets.size != row_bounds[1]-row_bounds[0]:
                raise ValueError('center_offsets dimension mismatch in reconstructOneTomoSet')
            centers = center_offsets
        centers += tomo_stack.shape[2]/2
        if True:
            tomo_stack = tomopy.prep.stripe.remove_stripe_fw(tomo_stack[row_bounds[0]:row_bounds[1]],
                    sigma=sigma, ncore=ncore)
        else:
            tomo_stack = tomo_stack[row_bounds[0]:row_bounds[1]]
        tomo_recon_stack = tomopy.recon(tomo_stack, thetas, centers, sinogram_order=True,
                algorithm=algorithm, ncore=ncore)
        if run_secondary_sirt and secondary_iter > 0:
            #options = {'method':'SIRT_CUDA', 'proj_type':'cuda', 'num_iter':secondary_iter}
            #RV: doesn't work for me: "Error: CUDA error 803: system has unsupported display driver /
            #                          cuda driver combination."
            #options = {'method':'SIRT', 'proj_type':'linear', 'MinConstraint': 0, 'num_iter':secondary_iter}
            #SIRT did not finish while running overnight
            #options = {'method':'SART', 'proj_type':'linear', 'num_iter':secondary_iter}
            options = {'method':'SART', 'proj_type':'linear', 'MinConstraint': 0, 'num_iter':secondary_iter}
            tomo_recon_stack  = tomopy.recon(tomo_stack, thetas, centers, init_recon=tomo_recon_stack,
                    options=options, sinogram_order=True, algorithm=tomopy.astra, ncore=ncore)
        if True:
            tomopy.misc.corr.remove_ring(tomo_recon_stack, rwidth=rwidth, out=tomo_recon_stack)
        return tomo_recon_stack

    def genTomoSets(self):
        """Preprocess tomography images."""
        logging.debug('Preprocess tomography images')

        # Try loading any already preprocessed sets."""
        # preprocessed stack order for each one in set: row,theta,column
        preprocessed_all_sets = True
        available_sets = [False]*self.num_tomo_data_sets
        for i in range(self.num_tomo_data_sets):
            if not self.tomo_sets[i].size:
                self.tomo_sets[i], available_sets[i] = self._loadTomo('red stack', i)
            if not self.tomo_sets[i].size and not available_sets[i]:
                preprocessed_all_sets = False

        # Preprocess any unloaded sets
        if not preprocessed_all_sets:

            logging.debug('Preprocessing tomography images')
            # Find and check required image files
            self.is_valid = self._getImageFiles(available_sets)
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
            self._genTomo(available_sets)

        # Adjust sample reference height and update config file
        for i in range(self.num_tomo_data_sets):
            self.tomo_ref_heights[i] += self.img_x_bounds[0]*self.pixel_size
            key = 'z_ref'
            if self.tomo_data_indices[i] != None:
                key += f'_{self.tomo_data_indices[i]}'
            msnc.updateConfigFile('config.txt', key, self.tomo_ref_heights[i])
        self.config = msnc.updateConfigFile('config.txt', 'pre_processor',
                self.num_tomo_data_sets, '# Analysis progress', '# Analysis progress')

    def findCenters(self):
        """Find rotation axis centers for the tomography stacks."""
        logging.debug('Find centers for tomography stacks')

        if 'center_stack_index' in self.config:
            center_stack_index = self.config['center_stack_index']
            if not msnc.is_int(center_stack_index, self.tomo_data_indices[0],
                    self.tomo_data_indices[self.num_tomo_data_sets-1]):
                msnc.illegal_value('center_stack_index', center_stack_index, 'config file')
            else:
                if self.test_mode:
                    self.config = msnc.updateConfigFile('config.txt', 'find_centers',
                            True, 'pre_processor')
                    return
                print('\nFound calibration center offset info for stack '+
                        f'{center_stack_index}')
                if pyip.inputYesNo('Do you want to use this again (y/n)? ') == 'yes':
                    self.config = msnc.updateConfigFile('config.txt', 'find_centers',
                            True, 'pre_processor')
                    return

        # Load the required preprocessed set."""
        # preprocessed stack order: row,theta,column
        if self.num_tomo_data_sets == 1:
            center_stack_index = self.tomo_data_indices[0]
            if not self.tomo_sets[0].size:
                self.tomo_sets[0], available = self._loadTomo('red stack', 0, required=True)
            center_stack = self.tomo_sets[0]
            if not center_stack.size:
                logging.error('Unable to load the required preprocessed tomography set')
                return
        else:
            while True:
                center_stack_index = pyip.inputInt(
                        '\nEnter tomography set index to get rotation axis centers '+
                        f'[{self.tomo_data_indices[0]},'+
                        f'{self.tomo_data_indices[self.num_tomo_data_sets-1]}]: ',
                        min=self.tomo_data_indices[0],
                        max=self.tomo_data_indices[self.num_tomo_data_sets-1])
                tomo_set_index = self.tomo_data_indices.index(center_stack_index)
                if not self.tomo_sets[tomo_set_index].size:
                    self.tomo_sets[tomo_set_index], available = self._loadTomo(
                            'red stack', tomo_set_index, required=True)
                center_stack = self.tomo_sets[tomo_set_index]
                if not center_stack.size:
                    logging.error(f'Unable to load the {center_stack_index}th '+
                        'preprocessed tomography set, pick another one')
                else:
                    break

        # Get non-overlapping sample row boundaries
        zoom_perc = self.config.get('zoom_perc')
        if zoom_perc == None:
            zoom_perc = 100
        self.eff_pixel_size = 100.*self.pixel_size/zoom_perc
        logging.debug(f'eff_pixel_size = {self.eff_pixel_size}')
        self.tomo_ref_heights = [None]*self.num_tomo_data_sets
        for i in range(self.num_tomo_data_sets):
            key = 'z_ref'
            if self.tomo_data_indices[i] != None:
                key += f'_{self.tomo_data_indices[i]}'
            if key in self.config:
                self.tomo_ref_heights[i] = self.config[key]
            else:
                raise ValueError(f'Unable to read {key} from config.txt')
        if self.num_tomo_data_sets == 1:
            n1 = 0
            height = center_stack.shape[0]*self.eff_pixel_size
            if pyip.inputYesNo('\nDo you want to reconstruct the full samply height '+
                    f'({height:.3f} mm) (y/n)? ') == 'no':
                height = pyip.inputNum('\nEnter the desired reconstructed sample height '+
                        f'in mm [0, {height:.3f}]: ', min=0, max=height)
                n1 = int(0.5*(center_stack.shape[0]-height/self.eff_pixel_size))
        else:
            n1 = int((1.+(self.tomo_ref_heights[0]+
                center_stack.shape[0]*self.eff_pixel_size-
                self.tomo_ref_heights[1])/self.eff_pixel_size)/2)
        n2 = center_stack.shape[0]-n1
        logging.info(f'n1 = {n1}, n2 = {n2} (n2-n1) = {(n2-n1)*self.eff_pixel_size:.3f} mm')
        if not center_stack.size:
            RuntimeError('Center stack not loaded')
        msnc.quickImshow(center_stack[:,0,:], title=f'center stack theta={self.start_theta}',
                path=self.output_folder, save_fig=self.save_plots, save_only=self.save_plots_only)

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
            msnc.quickImshow(center_stack[:,0,:], title=f'theta={self.start_theta}',
                    aspect='auto')
            row = int(self.config['lower_row'])
            use_row = pyip.inputYesNo('\nCurrent row index for lower center = '
                    f'{row}, use this value (y/n)? ')
            if self.save_plots_only:
                msnc.clearFig(f'theta={self.start_theta}')
            if use_row:
                if self.config.get('lower_center_offset'):
                    center_offset = self.config['lower_center_offset']
                    if msnc.is_num(center_offset):
                        use_center = pyip.inputYesNo('Current lower center offset = '+
                                f'{center_offset}, use this value (y/n)? ')
        if not use_center:
            if not use_row:
                msnc.quickImshow(center_stack[:,0,:], title=f'theta={self.start_theta}',
                        aspect='auto')
                row = pyip.inputInt('\nEnter row index to find lower center '+
                        f'[[{n1}], {n2-2}]: ', min=n1, max=n2-2, blank=True)
                if row == '':
                    row = n1
                if self.save_plots_only:
                    msnc.clearFig(f'theta={self.start_theta}')
            # center_stack order: row,theta,column
            center_offset = self._findCenterOnePlane(center_stack[row,:,:], row)
        lower_row = row
        lower_center_offset = center_offset
        logging.info(f'lower_center_offset = {lower_center_offset}')

        # Update config file
        msnc.updateConfigFile('config.txt', 'center_stack_index',
                center_stack_index, '# Calibration center offset info',
                '# Calibration center offset info')
        msnc.updateConfigFile('config.txt', 'row_bounds', [n1, n2], 'center_stack_index')
        msnc.updateConfigFile('config.txt', 'lower_row', lower_row, 'row_bounds')
        self.config = msnc.updateConfigFile('config.txt', 'lower_center_offset',
                lower_center_offset, 'lower_row')

        # Upper row center
        use_row = False
        use_center = False
        if self.config.get('upper_row'):
            msnc.quickImshow(center_stack[:,0,:], title=f'theta={self.start_theta}',
                    aspect='auto')
            row = int(self.config['upper_row'])
            use_row = pyip.inputYesNo('\nCurrent row index for upper center = '
                    f'{row}, use this value (y/n)? ')
            if self.save_plots_only:
                msnc.clearFig(f'theta={self.start_theta}')
            if use_row:
                if self.config.get('upper_center_offset'):
                    center_offset = int(self.config['upper_center_offset'])
                    if msnc.is_num(center_offset):
                        use_center = pyip.inputYesNo('Current upper center offset = '+
                                f'{center_offset}, use this value (y/n)? ')
        if not use_center:
            if not use_row:
                msnc.quickImshow(center_stack[:,0,:], title=f'theta={self.start_theta}',
                        aspect='auto')
                row = pyip.inputInt('\nEnter row index to find upper center '+
                        f'[{lower_row+1}, [{n2-1}]]: ', min=lower_row+1, max=n2-1, blank=True)
                if self.save_plots_only:
                    msnc.clearFig(f'theta={self.start_theta}')
                if row == '':
                    row = n2-1
            # center_stack order: row,theta,column
            center_offset = self._findCenterOnePlane(center_stack[row,:,:], row)
        upper_row = row
        upper_center_offset = center_offset
        logging.info(f'upper_center_offset = {upper_center_offset}')
        del center_stack

        # Update config file
        msnc.updateConfigFile('config.txt', 'upper_row', upper_row, 'lower_center_offset')
        msnc.updateConfigFile('config.txt', 'upper_center_offset',
                upper_center_offset, 'upper_row')
        self.config = msnc.updateConfigFile('config.txt', 'find_centers', True,
                'pre_processor')

    def checkCenters(self):
        """Check centers for the tomography stacks."""
        #RV TODO load all sets and check at all stack boundaries
        return
        logging.debug('Check centers for tomography stacks')
        center_stack_index = self.config.get('center_stack_index')
        if center_stack_index == None:
            raise ValueError('Unable to read center_stack_index from config')
        center_stack_index = self.tomo_sets[self.tomo_data_indices.index(center_stack_index)]
        lower_row = self.config.get('lower_row')
        if lower_row == None:
            raise ValueError('Unable to read lower_row from config')
        lower_center_offset = self.config.get('lower_center_offset')
        if lower_center_offset == None:
            raise ValueError('Unable to read lower_center_offset from config')
        upper_row = self.config.get('upper_row')
        if upper_row == None:
            raise ValueError('Unable to read upper_row from config')
        upper_center_offset = self.config.get('upper_center_offset')
        if upper_center_offset == None:
            raise ValueError('Unable to read upper_center_offset from config')
        center_slope = (upper_center_offset-lower_center_offset)/(upper_row-lower_row)
        shift = upper_center_offset-lower_center_offset
        if lower_row == 0:
            logging.warning(f'lower_row == 0: one row offset between both planes')
        else:
            lower_row -= 1
            lower_center_offset -= center_slope

        # stack order: set,row,theta,column
        if center_stack_index:
            set1 = self.tomo_sets[center_stack_index-1]
            set2 = self.tomo_sets[center_stack_index]
            if not set1.size:
                logging.error(f'Unable to load required tomography set {set1}')
            elif not set2.size:
                logging.error(f'Unable to load required tomography set {set1}')
            else:
                assert(0 <= lower_row < set2.shape[0])
                assert(0 <= upper_row < set1.shape[0])
                plane1 = set1[upper_row,:]
                plane2 = set2[lower_row,:]
                for i in range(-2, 3):
                    shift_i = shift+2*i
                    plane1_shifted = spi.shift(plane2, [0, shift_i])
                    msnc.quickPlot((plane1[0,:],), (plane1_shifted[0,:],),
                            title=f'sets {set1} {set2} shifted {2*i} theta={self.start_theta}',
                            path=self.output_folder, save_fig=self.save_plots,
                            save_only=self.save_plots_only)
        if center_stack_index < self.num_tomo_data_sets-1:
            set1 = self.tomo_sets[center_stack_index]
            set2 = self.tomo_sets[center_stack_index+1]
            if not set1.size:
                logging.error(f'Unable to load required tomography set {set1}')
            elif not set2.size:
                logging.error(f'Unable to load required tomography set {set1}')
            else:
                assert(0 <= lower_row < set2.shape[0])
                assert(0 <= upper_row < set1.shape[0])
                plane1 = set1[upper_row,:]
                plane2 = set2[lower_row,:]
                for i in range(-2, 3):
                    shift_i = -shift+2*i
                    plane1_shifted = spi.shift(plane2, [0, shift_i])
                    msnc.quickPlot((plane1[0,:],), (plane1_shifted[0,:],), 
                            title=f'sets {set1} {set2} shifted {2*i} theta={self.start_theta}',
                            path=self.output_folder, save_fig=self.save_plots,
                            save_only=self.save_plots_only)
        del plane1, plane2, plane1_shifted

        # Update config file
        self.config = msnc.updateConfigFile('config.txt', 'check_centers', True,
                'find_centers')

    def reconstructTomoSets(self):
        """Reconstruct tomography sets."""
        logging.debug('Reconstruct tomography sets')

        # Get rotation axis rows and centers
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
        logging.debug(f'lower_row = {lower_row} upper_row = {upper_row}')
        assert(lower_row < upper_row)
        center_slope = (upper_center_offset-lower_center_offset)/(upper_row-lower_row)

        # Set thetas (in radians)
        num_theta_skip = self.config.get('num_theta_skip')
        if num_theta_skip == None:
            num_theta_skip = 0
        thetas = np.radians(np.linspace(self.start_theta, self.end_theta,
                int(self.num_thetas/(num_theta_skip+1)), endpoint=False))

        # Reconstruct tomo sets
        zoom_perc = self.config.get('zoom_perc')
        if zoom_perc == None or zoom_perc == 100:
            basetitle = 'recon stack full'
        else:
            basetitle = f'recon stack {zoom_perc}p'
        load_error = False
        available_sets = [False]*self.num_tomo_data_sets
        for i in range(self.num_tomo_data_sets):
            # Check if stack can be loaded
            # reconstructed stack order for each one in set : row/z,x,y
            # preprocessed stack order for each one in set: row,theta,column
            self.tomo_recon_sets[i], available_sets[i] = self._loadTomo('recon stack', i)
            if self.tomo_recon_sets[i].size or available_sets[i]:
                if self.tomo_sets[i].size:
                    self.tomo_sets[i] = np.array([])
                continue
            else:
                if not self.tomo_sets[i].size:
                    self.tomo_sets[i], available = self._loadTomo('red stack', i, required=True)
                if not self.tomo_sets[i].size:
                    logging.error('Unable to load tomography set '+
                            f'{self.tomo_data_indices[i]} for reconstruction')
                    load_error = True
                    continue
                assert(0 <= lower_row < upper_row < self.tomo_sets[i].shape[0])
                center_offsets = [lower_center_offset-lower_row*center_slope,
                        upper_center_offset+(self.tomo_sets[i].shape[0]-1-upper_row)*center_slope]
                t0 = time()
                self.tomo_recon_sets[i]= self._reconstructOneTomoSet(self.tomo_sets[i], thetas,
                        center_offsets=center_offsets, sigma=0.1, ncore=self.ncore,
                        algorithm='gridrec', run_secondary_sirt=True, secondary_iter=25)
                logging.info(f'Reconstruction of set {i+1} took {time()-t0:.2f} seconds!')
                if not self.test_mode:
                    row_slice = int(self.tomo_sets[i].shape[0]/2) 
                    if self.tomo_data_indices[i] == None:
                        title = f'{basetitle} slice{row_slice}'
                    else:
                        title = f'{basetitle} {self.tomo_data_indices[i]} slice{row_slice}'
                    msnc.quickImshow(self.tomo_recon_sets[i][row_slice,:,:], title=title,
                            path=self.output_folder, save_fig=self.save_plots,
                            save_only=self.save_plots_only)
                    msnc.quickPlot(self.tomo_recon_sets[i]
                            [row_slice,int(self.tomo_recon_sets[i].shape[2]/2),:],
                            title=f'{title} cut{int(self.tomo_recon_sets[i].shape[2]/2)}',
                            path=self.output_folder, save_fig=self.save_plots,
                            save_only=self.save_plots_only)
                    self._saveTomo('recon stack', self.tomo_recon_sets[i], i)
#                else:
#                    np.savetxt(self.output_folder+f'recon_stack_{self.tomo_data_indices[i]}.txt',
#                            self.tomo_recon_sets[i][row_slice,:,:], fmt='%.6e')
                self.tomo_sets[i] = np.array([])

        # Update config file
        if not load_error:
            self.config = msnc.updateConfigFile('config.txt', 'reconstruct_sets', True,
                    ['check_centers', 'find_centers'])

    def combineTomoSets(self):
        """Combine the reconstructed tomography stacks."""
#        if self.num_tomo_data_sets == 1:
#            return
        # stack order: set,row(z),x,y
        logging.debug('Combine reconstructed tomography stacks')
        # Load any unloaded reconstructed sets."""
        for i in range(self.num_tomo_data_sets):
            if not self.tomo_recon_sets[i].size:
                self.tomo_recon_sets[i], available = self._loadTomo('recon stack',
                        i, required=True)
            if not self.tomo_recon_sets[i].size:
                logging.error(f'Unable to load reconstructed set {self.tomo_data_indices[i]}')
                return
            if i:
                if (self.tomo_recon_sets[i].shape[1] != self.tomo_recon_sets[0].shape[1] or
                        self.tomo_recon_sets[i].shape[1] != self.tomo_recon_sets[0].shape[1]):
                    logging.error('Incompatible reconstructed tomography set dimensions')
                    return

        # Get center stack boundaries
        row_bounds = self.config.get('row_bounds')
        if not msnc.is_index_range(row_bounds):
            msnc.illegal_value('row_bounds', row_bounds, 'config file')
            return
        if row_bounds[1] > self.tomo_recon_sets[0].shape[0]:
            msnc.illegal_value('row_bounds[1]', row_bounds[1], 'config file')
            row_bounds[1] = self.tomo_recon_sets[0].shape[0]

        # Selecting xy bounds
        tomosum = 0
        [tomosum := tomosum+np.sum(tomo_recon_stack, axis=(0,2)) for tomo_recon_stack in
                self.tomo_recon_sets]
        if self.config.get('x_bounds'):
            x_bounds = self.config['x_bounds']
            if not msnc.is_index_range(x_bounds, 0, self.tomo_recon_sets[0].shape[1]):
                msnc.illegal_value('x_bounds', x_bounds, 'config file')
            elif not self.test_mode:
                msnc.quickPlot(tomosum, title='recon stack sum yz')
                if pyip.inputYesNo('\nCurrent image x-bounds: '+
                        f'[{x_bounds[0]}, {x_bounds[1]}], use these values ([y]/n)? ',
                        blank=True) == 'no':
                    if pyip.inputYesNo(
                            'Do you want to change the image x-bounds ([y]/n)? ',
                            blank=True) == 'no':
                        x_bounds = [0, self.tomo_recon_sets[0].shape[1]]
                    else:
                        x_bounds = msnc.selectArrayBounds(tomosum, title='recon stack sum yz')
        else:
            msnc.quickPlot(tomosum, title='recon stack sum yz')
            if pyip.inputYesNo('\nDo you want to change the image x-bounds (y/n)? ') == 'no':
                x_bounds = [0, self.tomo_recon_sets[0].shape[1]]
            else:
                x_bounds = msnc.selectArrayBounds(tomosum, title='recon stack sum yz')
        if False and self.test_mode:
            np.savetxt(self.output_folder+'recon_stack_sum_yz.txt',
                    tomosum[x_bounds[0]:x_bounds[1]], fmt='%.6e')
        if self.save_plots_only:
            msnc.clearFig('recon stack sum yz')
        tomosum = 0
        [tomosum := tomosum+np.sum(tomo_recon_stack, axis=(0,1)) for tomo_recon_stack in
                self.tomo_recon_sets]
        if self.config.get('y_bounds'):
            y_bounds = self.config['y_bounds']
            if not msnc.is_index_range(x_bounds, 0, self.tomo_recon_sets[0].shape[1]):
                msnc.illegal_value('y_bounds', y_bounds, 'config file')
            elif not self.test_mode:
                msnc.quickPlot(tomosum, title='recon stack sum xz')
                if pyip.inputYesNo('\nCurrent image y-bounds: '+
                        f'[{y_bounds[0]}, {y_bounds[1]}], use these values ([y]/n)? ',
                        blank=True) == 'no':
                    if pyip.inputYesNo(
                            'Do you want to change the image y-bounds ([y]/n)? ',
                            blank=True) == 'no':
                        y_bounds = [0, self.tomo_recon_sets[0].shape[1]]
                    else:
                        y_bounds = msnc.selectArrayBounds(tomosum, title='recon stack sum yz')
        else:
            msnc.quickPlot(tomosum, title='recon stack sum xz')
            if pyip.inputYesNo('\nDo you want to change the image y-bounds (y/n)? ') == 'no':
                y_bounds = [0, self.tomo_recon_sets[0].shape[1]]
            else:
                y_bounds = msnc.selectArrayBounds(tomosum, title='recon stack sum xz')
        if False and self.test_mode:
            np.savetxt(self.output_folder+'recon_stack_sum_xz.txt',
                    tomosum[y_bounds[0]:y_bounds[1]], fmt='%.6e')
        if self.save_plots_only:
            msnc.clearFig('recon stack sum xz')

        # Combine reconstructed tomography sets
        logging.info(f'Combining reconstructed sets ...')
        t0 = time()
        if self.num_tomo_data_sets == 1:
            low_bound = row_bounds[0]
        else:
            low_bound = 0
        tomo_recon_combined = self.tomo_recon_sets[0][low_bound:row_bounds[1]:,
                x_bounds[0]:x_bounds[1],y_bounds[0]:y_bounds[1]]
        if self.num_tomo_data_sets > 2:
            tomo_recon_combined = np.concatenate([tomo_recon_combined]+
                    [self.tomo_recon_sets[i][row_bounds[0]:row_bounds[1],
                    x_bounds[0]:x_bounds[1],y_bounds[0]:y_bounds[1]]
                    for i in range(1, self.num_tomo_data_sets-1)])
        if self.num_tomo_data_sets > 1:
            tomo_recon_combined = np.concatenate([tomo_recon_combined]+
                    [self.tomo_recon_sets[self.num_tomo_data_sets-1][row_bounds[0]:,
                    x_bounds[0]:x_bounds[1],y_bounds[0]:y_bounds[1]]])
        logging.info(f'... done in {time()-t0:.2f} seconds!')
        tomosum = np.sum(tomo_recon_combined, axis=(1,2))
        if self.test_mode:
            zoom_perc = self.config.get('zoom_perc')
            filename = 'recon combined sum xy'
            if zoom_perc == None or zoom_perc == 100:
                filename += ' fullres.dat'
            else:
                filename += f' {zoom_perc}p.dat'
            msnc.quickPlot(tomosum, title='recon combined sum xy',
                    path=self.output_folder, save_fig=self.save_plots,
                    save_only=self.save_plots_only)
            if False:
                np.savetxt(self.output_folder+'recon_combined_sum_xy.txt',
                        tomosum, fmt='%.6e')
            np.savetxt(self.output_folder+'recon_combined.txt',
                    tomo_recon_combined[int(tomo_recon_combined.shape[0]/2),:,:], fmt='%.6e')
            return
        msnc.quickPlot(tomosum, title='recon combined sum xy')
        if pyip.inputYesNo(
                '\nDo you want to change the image z-bounds (y/[n])? ',
                blank=True) != 'yes':
            z_bounds = [0, tomo_recon_combined.shape[0]]
        else:
            z_bounds = msnc.selectArrayBounds(tomosum, title='recon combined sum xy')
        if z_bounds[0] != 0 or z_bounds[1] != tomo_recon_combined.shape[0]:
            tomo_recon_combined = tomo_recon_combined[z_bounds[0]:z_bounds[1],:,:]
        logging.info(f'tomo_recon_combined.shape = {tomo_recon_combined.shape}')
        if self.save_plots_only:
            msnc.clearFig('recon combined sum xy')

        # Plot center slices
        msnc.quickImshow(tomo_recon_combined[int(tomo_recon_combined.shape[0]/2),:,:],
                title=f'recon combined xslice{int(tomo_recon_combined.shape[0]/2)}',
                path=self.output_folder, save_fig=self.save_plots,
                save_only=self.save_plots_only)
        msnc.quickImshow(tomo_recon_combined[:,int(tomo_recon_combined.shape[1]/2),:],
                title=f'recon combined yslice{int(tomo_recon_combined.shape[1]/2)}',
                path=self.output_folder, save_fig=self.save_plots,
                save_only=self.save_plots_only)
        msnc.quickImshow(tomo_recon_combined[:,:,int(tomo_recon_combined.shape[2]/2)],
                title=f'recon combined zslice{int(tomo_recon_combined.shape[2]/2)}',
                path=self.output_folder, save_fig=self.save_plots,
                save_only=self.save_plots_only)

        # Save combined reconstructed tomo sets
        base_name = 'recon combined'
        for i in range(self.num_tomo_data_sets):
            if self.tomo_data_indices[i] != None:
                base_name += f' {self.tomo_data_indices[i]}'
        self._saveTomo(base_name, tomo_recon_combined)

        # Update config file
        msnc.updateConfigFile('config.txt', 'x_bounds', x_bounds,
                '# Combined reconstruction info', '# Combined reconstruction info')
        msnc.updateConfigFile('config.txt', 'y_bounds', y_bounds,
                ['x_bounds', '# Combined reconstruction info'])
        self.config = msnc.updateConfigFile('config.txt', 'combine_sets', True,
                'reconstruct_sets')

def runTomo(config_file=None, config_dict=None, output_folder='.', log_level='INFO',
        test_mode=False):
    """Run a tomography analysis"""
    tomo = Tomo(config_file=config_file, output_folder=output_folder, log_level=log_level,
            test_mode=test_mode)
    if not tomo.is_valid:
        raise ValueError('Invalid config and/or detector file provided.')

    # Preprocess the image files
    if tomo.config.get('pre_processor', 0) != tomo.num_tomo_data_sets:
        tomo.genTomoSets()
        if not tomo.is_valid:
            IOError('Unable to load all required image files.')
        if 'check_centers' in tomo.config:
            tomo.config = msnc.updateConfigFile('config.txt', 'check_centers', False)
        if 'reconstruct_sets' in tomo.config:
            tomo.config = msnc.updateConfigFile('config.txt', 'reconstruct_sets', False)
        if 'combine_sets' in tomo.config:
            tomo.config = msnc.updateConfigFile('config.txt', 'combine_sets', False)

    # Find centers
    if not tomo.config.get('find_centers', False):
        tomo.findCenters()

    # Check centers
    if tomo.num_tomo_data_sets > 1 and not tomo.config.get('check_centers', False):
        tomo.checkCenters()

    # Reconstruct tomography sets
    if not tomo.config.get('reconstruct_sets', False):
        tomo.reconstructTomoSets()

    # Combine reconstructed tomography sets
    if not tomo.config.get('combine_sets', False):
        tomo.combineTomoSets()

#%%============================================================================
if __name__ == '__main__':
    arguments = sys.argv[1:]
    config_file = 'config.txt'
    output_folder = '.'
    log_level = 'INFO'
    test_mode = False
    try:
        opts, args = getopt.getopt(arguments,"hc:o:l:t")
    except getopt.GetoptError:
        print('usage: tomo.py -c <config_file> -o <output_folder> -l <log_level> -t')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: tomo.py -c <config_file> -o <output_folder> -l <log_level> -t')
            sys.exit()
        elif opt in ("-c"):
            config_file = arg
        elif opt in ("-o"):
            output_folder = arg
        elif opt in ("-l"):
            log_level = arg
        elif opt in ("-t"):
            test_mode = True

    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s : %(levelname)s - %(module)s : %(funcName)s - %(message)s',
            handlers=[logging.FileHandler(f'{output_folder}/tomo.log', mode='w'), stream_handler])
    if test_mode:
        logging.basicConfig(filename=f'{output_folder}/tomo.log', filemode='w',
                uevel=logging.WARNING)
        if len(log_level):
            logging.warning('Ignoring command line log_level argument in test mode')
    else:
        level = getattr(logging, log_level.upper(), None)
        if not isinstance(level, int):
            raise ValueError(f'Invalid log_level: {log_level}')
        stream_handler.setLevel(level)
    logging.info(f'config_file = {config_file}')
    logging.info(f'output_folder = {output_folder}')
    logging.info(f'log_level = {log_level}')

    runTomo(config_file=config_file, output_folder=output_folder, log_level=log_level,
            test_mode=test_mode)

#%%============================================================================
    input('Press any key to continue')
#%%============================================================================

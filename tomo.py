#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:54:37 2021

@author: rv43
"""

import logging

import os
import sys
import getopt
import re
import io
import argparse
import numpy as np
try:
    import numexpr as ne
except:
    pass
import multiprocessing as mp
try:
    import scipy.ndimage as spi
except:
    pass
try:
    import tomopy
except:
    pass
from time import time
try:
    from skimage.transform import iradon
except:
    pass
try:
    from skimage.restoration import denoise_tv_chambolle
except:
    pass

from detector import TomoDetectorConfig
from fit import Fit
#from msnctools.general import illegal_value, is_int, is_num, is_index_range, get_trailing_int, \
from general import illegal_value, is_int, is_num, is_index_range, get_trailing_int, \
        input_int, input_num, input_yesno, input_menu, findImageFiles, loadImageStack, clearPlot, \
        draw_mask_1d, quickPlot, clearImshow, quickImshow, combine_tiffs_in_h5, Config
from general import selectArrayBounds,selectImageRange, selectImageBounds

# the following tomopy routines don't run with more than 24 cores on Galaxy-Dev
#   - tomopy.find_center_vo
#   - tomopy.prep.stripe.remove_stripe_fw
num_core_tomopy_limit = 24

class set_numexpr_threads:

    def __init__(self, num_core):
        cpu_count = mp.cpu_count()
        logging.debug(f'start: num_core={num_core} cpu_count={cpu_count}')
        if num_core is None or num_core < 1 or num_core > cpu_count:
            self.num_core = cpu_count
        else:
            self.num_core = num_core
        logging.debug(f'self.num_core={self.num_core}')

    def __enter__(self):
        self.num_core_org = ne.set_num_threads(self.num_core)
        logging.debug(f'self.num_core={self.num_core}')

    def __exit__(self, exc_type, exc_value, traceback):
        ne.set_num_threads(self.num_core_org)

class ConfigTomo(Config):
    """Class for processing a config file.
    """

    def __init__(self, config_file=None, config_dict=None):
        super().__init__(config_file, config_dict)

    def _validate_txt(self):
        """Returns False if any required config parameter is illegal or missing.
        """
        is_valid = True

        # Check for required first-level keys
        pars_required = ['tdf_data_path', 'tbf_data_path', 'detector_id']
        pars_missing = []
        is_valid = super().validate(pars_required, pars_missing)
        if len(pars_missing) > 0:
            logging.error(f'Missing item(s) in config file: {", ".join(pars_missing)}')
        self.detector_id = self.config.get('detector_id')

        # Find tomography dark field images file/folder
        self.tdf_data_path = self.config.get('tdf_data_path')

        # Find tomography bright field images file/folder
        self.tbf_data_path = self.config.get('tbf_data_path')

        # Check number of tomography image stacks
        self.num_tomo_stacks = self.config.get('num_tomo_stacks', 1)
        if not is_int(self.num_tomo_stacks, 1):
            self.num_tomo_stacks = None
            illegal_value(self.num_tomo_stacks, 'num_tomo_stacks', 'config file')
            return False
        logging.info(f'num_tomo_stacks = {self.num_tomo_stacks}')

        # Find tomography images file/folders and stack parameters
        tomo_data_paths_indices = sorted({key:value for key,value in self.config.items()
            if 'tomo_data_path' in key}.items())
        if len(tomo_data_paths_indices) != self.num_tomo_stacks:
            logging.error(f'Incorrect number of tomography data path names in config file')
            is_valid = False
        self.tomo_data_paths = [tomo_data_paths_indices[i][1] for i in range(self.num_tomo_stacks)]
        self.tomo_data_indices = [get_trailing_int(tomo_data_paths_indices[i][0])
                if get_trailing_int(tomo_data_paths_indices[i][0]) else None
                for i in range(self.num_tomo_stacks)]
        tomo_ref_height_indices = sorted({key:value for key,value in self.config.items()
                if 'z_pos' in key}.items())
        if self.num_tomo_stacks > 1 and len(tomo_ref_height_indices) != self.num_tomo_stacks:
            logging.error(f'Incorrect number of tomography reference heights in config file')
            is_valid = False
        if len(tomo_ref_height_indices):
            self.tomo_ref_heights = [
                    tomo_ref_height_indices[i][1] for i in range(self.num_tomo_stacks)]
        else:
            self.tomo_ref_heights = [0.0]*self.num_tomo_stacks

        # Check tomo angle (theta) range
        self.start_theta = self.config.get('start_theta', 0.)
        if not is_num(self.start_theta, 0.):
            illegal_value(self.start_theta, 'start_theta', 'config file')
            is_valid = False
        logging.debug(f'start_theta = {self.start_theta}')
        self.end_theta = self.config.get('end_theta', 180.)
        if not is_num(self.end_theta, self.start_theta):
            illegal_value(self.end_theta, 'end_theta', 'config file')
            is_valid = False
        logging.debug(f'end_theta = {self.end_theta}')
        self.num_thetas = self.config.get('num_thetas')
        if not (self.num_thetas is None or is_int(self.num_thetas, 1)):
            illegal_value(self.num_thetas, 'num_thetas', 'config file')
            self.num_thetas = None
            is_valid = False
        logging.debug(f'num_thetas = {self.num_thetas}')

        return is_valid

    def _validate_yaml(self):
        """Returns False if any required config parameter is illegal or missing.
        """
        is_valid = True

        # Check for required first-level keys
        pars_required = ['dark_field', 'bright_field', 'stack_info', 'detector']
        pars_missing = []
        is_valid = super().validate(pars_required, pars_missing)
        if len(pars_missing) > 0:
            logging.error(f'Missing item(s) in config file: {", ".join(pars_missing)}')
        self.detector_id = self.config['detector'].get('id')

        # Find tomography dark field images file/folder
        self.tdf_data_path = self.config['dark_field'].get('data_path')

        # Find tomography bright field images file/folder
        self.tbf_data_path = self.config['bright_field'].get('data_path')

        # Check number of tomography image stacks
        stack_info = self.config['stack_info']
        self.num_tomo_stacks = stack_info.get('num', 1)
        if not is_int(self.num_tomo_stacks, 1):
            self.num_tomo_stacks = None
            illegal_value(self.num_tomo_stacks, 'num_tomo_stacks', 'config file')
            return False
        logging.info(f'num_tomo_stacks = {self.num_tomo_stacks}')

        # Find tomography images file/folders and stack parameters
        stacks = stack_info.get('stacks')
        if stacks is None or len(stacks) is not self.num_tomo_stacks:
            illegal_value(stacks, 'stacks', 'config file')
            return False
        self.tomo_data_paths = []
        self.tomo_data_indices = []
        self.tomo_ref_heights = []
        for stack in stacks:
            self.tomo_data_paths.append(stack.get('data_path'))
            self.tomo_data_indices.append(stack.get('index'))
            self.tomo_ref_heights.append(stack.get('ref_height'))

        # Check tomo angle (theta) range
        theta_range = self.config.get('theta_range')
        if theta_range is None:
            self.start_theta = 0.
            self.end_theta = 180.
            self.num_thetas = None
        else:
            self.start_theta = theta_range.get('start', 0.)
            if not is_num(self.start_theta, 0.):
                illegal_value(self.start_theta, 'theta_range:start', 'config file')
                is_valid = False
            logging.debug(f'start_theta = {self.start_theta}')
            self.end_theta = theta_range.get('end', 180.)
            if not is_num(self.end_theta, self.start_theta):
                illegal_value(self.end_theta, 'theta_range:end', 'config file')
                is_valid = False
            logging.debug(f'end_theta = {self.end_theta}')
            self.num_thetas = theta_range.get('num')
            if self.num_thetas and not is_int(self.num_thetas, 1):
                illegal_value(self.num_thetas, 'theta_range:num', 'config file')
                self.num_thetas = None
                is_valid = False
            logging.debug(f'num_thetas = {self.num_thetas}')

        return is_valid

    def validate(self):
        """Returns False if any required config parameter is illegal or missing.
        """
        is_valid = True

        # Check work_folder (shared by both file formats)
        work_folder = os.path.abspath(self.config.get('work_folder', ''))
        if not os.path.isdir(work_folder):
            illegal_value(work_folder, 'work_folder', 'config file')
            is_valid = False
        logging.info(f'work_folder: {work_folder}')

        # Check data filetype (shared by both file formats)
        self.data_filetype = self.config.get('data_filetype', 'tif')
        if not isinstance(self.data_filetype, str) or (self.data_filetype != 'tif' and
                self.data_filetype != 'h5'):
            illegal_value(self.data_filetype, 'data_filetype', 'config file')

        if self.suffix == '.yml' or self.suffix == '.yaml':
            is_valid = self._validate_yaml()
        elif self.suffix == '.txt':
            is_valid = self._validate_txt()
        else:
            logging.error(f'Undefined or illegal config file extension: {self.suffix}')
            is_valid = False

        # Find tomography bright field images file/folder
        if self.tdf_data_path:
            if self.data_filetype == 'h5':
                if isinstance(self.tdf_data_path, str):
                    if not os.path.isabs(self.tdf_data_path):
                        self.tdf_data_path = os.path.abspath(
                                f'{work_folder}/{self.tdf_data_path}')
                else:
                    illegal_value(tdf_data_fil, 'tdf_data_path', 'config file')
                    is_valid = False
            else:
                if isinstance(self.tdf_data_path, int):
                    self.tdf_data_path = os.path.abspath(
                            f'{work_folder}/{self.tdf_data_path}/nf')
                elif isinstance(self.tdf_data_path, str):
                    if not os.path.isabs(self.tdf_data_path):
                        self.tdf_data_path = os.path.abspath(
                                f'{work_folder}/{self.tdf_data_path}')
                else:
                    illegal_value(self.tdf_data_path, 'tdf_data_path', 'config file')
                    is_valid = False
        logging.info(f'dark field images path = {self.tdf_data_path}')

        # Find tomography bright field images file/folder
        if self.tbf_data_path:
            if self.data_filetype == 'h5':
                if isinstance(self.tbf_data_path, str):
                    if not os.path.isabs(self.tbf_data_path):
                        self.tbf_data_path = os.path.abspath(
                                f'{work_folder}/{self.tbf_data_path}')
                else:
                    illegal_value(tbf_data_fil, 'tbf_data_path', 'config file')
                    is_valid = False
            else:
                if isinstance(self.tbf_data_path, int):
                    self.tbf_data_path = os.path.abspath(
                            f'{work_folder}/{self.tbf_data_path}/nf')
                elif isinstance(self.tbf_data_path, str):
                    if not os.path.isabs(self.tbf_data_path):
                        self.tbf_data_path = os.path.abspath(
                                f'{work_folder}/{self.tbf_data_path}')
                else:
                    illegal_value(self.tbf_data_path, 'tbf_data_path', 'config file')
                    is_valid = False
        logging.info(f'bright field images path = {self.tbf_data_path}')

        # Find tomography images file/folders and stack parameters
        tomo_data_paths = []
        tomo_data_indices = []
        tomo_ref_heights = []
        for data_path, index, ref_height in zip(self.tomo_data_paths, self.tomo_data_indices,
                self.tomo_ref_heights):
            if self.data_filetype == 'h5':
                if isinstance(data_path, str):
                    if not os.path.isabs(data_path):
                        data_path = os.path.abspath(f'{work_folder}/{data_path}')
                else:
                    illegal_value(data_path, 'stack_info:stacks:data_path', 'config file')
                    is_valid = False
                    data_path = None
            else:
                if isinstance(data_path, int):
                    data_path = os.path.abspath(f'{work_folder}/{data_path}/nf')
                elif isinstance(data_path, str):
                    if not os.path.isabs(data_path):
                        data_path = os.path.abspath(f'{work_folder}/{data_path}')
                else:
                    illegal_value(data_path, 'stack_info:stacks:data_path', 'config file')
                    is_valid = False
                    data_path = None
            tomo_data_paths.append(data_path)
            if index is None:
                if self.num_tomo_stacks > 1:
                    logging.error('Missing stack_info:stacks:index in config file')
                    is_valid = False
                    index = None
                else:
                    index = 1
            elif not isinstance(index, int):
                illegal_value(index, 'stack_info:stacks:index', 'config file')
                is_valid = False
                index = None
            tomo_data_indices.append(index)
            if ref_height is None:
                if self.num_tomo_stacks > 1:
                    logging.error('Missing stack_info:stacks:ref_height in config file')
                    is_valid = False
                    ref_height = None
                else:
                    ref_height = 0.
            elif not is_num(ref_height):
                illegal_value(ref_height, 'stack_info:stacks:ref_height', 'config file')
                is_valid = False
                ref_height = None
            # Set reference heights relative to first stack
            if (len(tomo_ref_heights) and is_num(ref_height) and
                    is_num(tomo_ref_heights[0])):
                ref_height = (round(ref_height-tomo_ref_heights[0], 3))
            tomo_ref_heights.append(ref_height)
        tomo_ref_heights[0] = 0.0
        logging.info('tomography data paths:')
        for i in range(self.num_tomo_stacks):
            logging.info(f'    {tomo_data_paths[i]}')
        logging.info(f'tomography data path indices: {tomo_data_indices}')
        logging.info(f'tomography reference heights: {tomo_ref_heights}')

        # Update config in memory
        if self.suffix == '.txt':
            self.config = {}
        dark_field = self.config.get('dark_field')
        if dark_field is None:
            self.config['dark_field'] = {'data_path' : self.tdf_data_path}
        else:
            self.config['dark_field']['data_path'] = self.tdf_data_path
        bright_field = self.config.get('bright_field')
        if bright_field is None:
            self.config['bright_field'] = {'data_path' : self.tbf_data_path}
        else:
            self.config['bright_field']['data_path'] = self.tbf_data_path
        detector = self.config.get('detector')
        if detector is None:
            self.config['detector'] = {'id' : self.detector_id}
        else:
            detector['id'] = self.detector_id
        self.config['work_folder'] = work_folder
        self.config['data_filetype'] = self.data_filetype
        stack_info = self.config.get('stack_info')
        if stack_info is None:
            stacks = []
            for i in range(self.num_tomo_stacks):
                stacks.append({'data_path' : tomo_data_paths[i], 'index' : tomo_data_indices[i],
                        'ref_height' : tomo_ref_heights[i]})
            self.config['stack_info'] = {'num' : self.num_tomo_stacks, 'stacks' : stacks}
        else:
            stack_info['num'] = self.num_tomo_stacks
            stacks = stack_info.get('stacks')
            for i,stack in enumerate(stacks):
                stack['data_path'] = tomo_data_paths[i]
                stack['index'] = tomo_data_indices[i]
                stack['ref_height'] = tomo_ref_heights[i]
        if self.num_thetas:
            theta_range = {'start' : self.start_theta, 'end' : self.end_theta,
                    'num' : self.num_thetas}
        else:
            theta_range = {'start' : self.start_theta, 'end' : self.end_theta}
        self.config['theta_range'] = theta_range

        # Cleanup temporary validation variables
        del self.tdf_data_path
        del self.tbf_data_path
        del self.detector_id
        del self.data_filetype
        del self.num_tomo_stacks
        del self.tomo_data_paths
        del self.tomo_data_indices
        del self.tomo_ref_heights
        del self.start_theta
        del self.end_theta
        del self.num_thetas

        return is_valid

class Tomo:
    """Processing tomography data with misalignment.
    """
    
    def __init__(self, config_file=None, config_dict=None, config_out=None, output_folder='.',
            log_level='INFO', log_stream='tomo.log', galaxy_flag=False, test_mode=False,
            num_core=-1):
        """Initialize with optional config input file or dictionary
        """
        self.num_core = None
        self.config_out = config_out
        self.output_folder = output_folder
        self.galaxy_flag = galaxy_flag
        self.test_mode = test_mode
        self.save_plots = True # Make input argument?
        self.save_plots_only = True # Make input argument?
        self.cf = None
        self.config = None
        self.is_valid = True
        self.tdf = np.array([])
        self.tbf = np.array([])
        self.tomo_stacks = []
        self.tomo_recon_stacks = []

        # Validate input parameters
        if config_file is not None and not os.path.isfile(config_file):
            raise OSError(f'Invalid config_file input {config_file} {type(config_file)}')
        if config_dict is not None and not isinstance(config_dict, dict):
            raise ValueError(f'Invalid config_dict input {config_dict} {type(config_dict)}')
        if self.config_out is not None and not isinstance(self.config_out, str):
            raise OSError(f'Invalid config_out input {self.config_out} {type(self.config_out)}')
        if not os.path.isdir(output_folder):
            os.mkdir(os.path.abspath(output_folder))
        if isinstance(log_stream, str):
            path = os.path.split(log_stream)[0]
            if path and not os.path.isdir(path):
                raise OSError(f'Invalid log_stream path')
            if not os.path.isabs(path):
                log_stream = f'{output_folder}/{log_stream}'
        if not isinstance(galaxy_flag, bool):
            raise ValueError(f'Invalid galaxy_flag input {galaxy_flag} {type(galaxy_flag)}')
        if not isinstance(self.test_mode, bool):
            raise ValueError(f'Invalid test_mode input {self.test_mode} {type(self.test_mode)}')
        if not isinstance(num_core, int) or num_core < -1 or num_core == 0:
            raise ValueError(f'Invalid num_core input {num_core} {type(num_core)}')
        if num_core == -1:
            self.num_core = mp.cpu_count()
        else:
            self.num_core = num_core

        # Set log configuration
        logging_format = '%(asctime)s : %(levelname)s - %(module)s : %(funcName)s - %(message)s'
        if self.test_mode:
            self.save_plots_only = True
            if isinstance(log_stream, str):
                logging.basicConfig(filename=f'{log_stream}', filemode='w',
                        format=logging_format, level=logging.INFO, force=True)
                        #format=logging_format, level=logging.WARNING, force=True)
            elif isinstance(log_stream, io.TextIOWrapper):
                #logging.basicConfig(filemode='w', format=logging_format, level=logging.WARNING,
                logging.basicConfig(filemode='w', format=logging_format, level=logging.INFO,
                        stream=log_stream, force=True)
            else:
                raise ValueError(f'Invalid log_stream: {log_stream}')
            logging.warning('Ignoring log_level argument in test mode')
        else:
            level = getattr(logging, log_level.upper(), None)
            if not isinstance(level, int):
                raise ValueError(f'Invalid log_level: {log_level}')
            if log_stream is sys.stdout:
                logging.basicConfig(format=logging_format, level=level, force=True,
                        handlers=[logging.StreamHandler()])
            else:
                if isinstance(log_stream, str):
                    logging.basicConfig(filename=f'{log_stream}', filemode='w',
                            format=logging_format, level=level, force=True)
                elif isinstance(log_stream, io.TextIOWrapper):
                    logging.basicConfig(filemode='w', format=logging_format, level=level,
                            stream=log_stream, force=True)
                else:
                    raise ValueError(f'Invalid log_stream: {log_stream}')
                stream_handler = logging.StreamHandler()
                logging.getLogger().addHandler(stream_handler)
                stream_handler.setLevel(logging.WARNING)
                stream_handler.setFormatter(logging.Formatter(logging_format))

        # Check/set output config file name
        if self.config_out is None:
            self.config_out = f'{self.output_folder}/config.yaml'
        elif (self.config_out is os.path.basename(self.config_out) and
                not os.path.isabs(self.config_out)):
            self.config_out = f'{self.output_folder}/{self.config_out}'

        # Create config object and load config file 
        self.cf = ConfigTomo(config_file, config_dict)
        if not self.cf.load_flag:
            self.is_valid = False
            return

        if self.galaxy_flag:
            assert(self.output_folder == '.')
            assert(self.test_mode is False)
            self.save_plots = True
            self.save_plots_only = True
        else:
            # Input validation is already performed during link_data_to_galaxy

            # Check config file parameters
            self.is_valid =  self.cf.validate()

            # Load detector info file
            df = TomoDetectorConfig(self.cf.config['detector']['id'])

            # Check detector info file parameters
            if df.valid:
                pixel_size = df.pixel_size
                num_rows, num_columns = df.dimensions
                if not pixel_size or not num_rows or not num_columns:
                    self.is_valid = False
            else:
                pixel_size = None
                num_rows = None
                num_columns = None
                self.is_valid = False

            # Update config
            self.cf.config['detector']['pixel_size'] = pixel_size
            self.cf.config['detector']['rows'] = num_rows
            self.cf.config['detector']['columns'] = num_columns
            logging.debug(f'pixel_size = self.cf.config["detector"]["pixel_size"]')
            logging.debug(f'num_rows: {self.cf.config["detector"]["rows"]}')
            logging.debug(f'num_columns: {self.cf.config["detector"]["columns"]}')

        # Safe config to file
        if self.is_valid:
            self.cf.saveFile(self.config_out)

        # Initialize shortcut to config
        self.config = self.cf.config

        # Initialize tomography stack
        num_tomo_stacks = self.config['stack_info']['num']
        if num_tomo_stacks:
            self.tomo_stacks = [np.array([]) for _ in range(num_tomo_stacks)]
            self.tomo_recon_stacks = [np.array([]) for _ in range(num_tomo_stacks)]

        logging.debug(f'num_core = {self.num_core}')
        logging.debug(f'config_file = {config_file}')
        logging.debug(f'config_dict = {config_dict}')
        logging.debug(f'config_out = {self.config_out}')
        logging.debug(f'output_folder = {self.output_folder}')
        logging.debug(f'log_stream = {log_stream}')
        logging.debug(f'log_level = {log_level}')
        logging.debug(f'galaxy_flag = {self.galaxy_flag}')
        logging.debug(f'test_mode = {self.test_mode}')
        logging.debug(f'save_plots = {self.save_plots}')
        logging.debug(f'save_plots_only = {self.save_plots_only}')

    def _selectImageRanges(self, available_stacks=None):
        """Select image files to be included in analysis.
        """
        self.is_valid = True
        stack_info = self.config['stack_info']
        if available_stacks is None:
            available_stacks = [False]*stack_info['num']
        elif len(available_stacks) != stack_info['num']:
            logging.warning('Illegal dimension of available_stacks in getImageFiles '+
                    f'({len(available_stacks)}');
            available_stacks = [False]*stack_info['num']

        # Check number of tomography angles/thetas
        num_thetas = self.config['theta_range'].get('num')
        if num_thetas is None:
            num_thetas = input_int('\nEnter the number of thetas', 1)
        elif not is_int(num_thetas, 0):
            illegal_value(num_thetas, 'num_thetas', 'config file')
            self.is_valid = False
            return
        self.config['theta_range']['num'] = num_thetas
        logging.debug(f'num_thetas = {self.config["theta_range"]["num"]}')

        # Find tomography dark field images
        dark_field = self.config['dark_field']
        img_start = dark_field.get('img_start', -1)
        img_offset = dark_field.get('img_offset', -1)
        num_imgs = dark_field.get('num', 0)
        if not self.test_mode:
            img_start, img_offset, num_imgs = selectImageRange(img_start, img_offset,
                num_imgs, 'dark field')
        if img_start < 0 or num_imgs < 1:
            logging.error('Unable to find suitable dark field images')
            if dark_field['data_path']:
                self.is_valid = False
        dark_field['img_start'] = img_start
        dark_field['img_offset'] = img_offset
        dark_field['num'] = num_imgs
        logging.debug(f'Dark field image start index: {dark_field["img_start"]}')
        logging.debug(f'Dark field image offset: {dark_field["img_offset"]}')
        logging.debug(f'Number of dark field images: {dark_field["num"]}')

        # Find tomography bright field images
        bright_field = self.config['bright_field']
        img_start = bright_field.get('img_start', -1)
        img_offset = bright_field.get('img_offset', -1)
        num_imgs = bright_field.get('num', 0)
        if not self.test_mode:
            img_start, img_offset, num_imgs = selectImageRange(img_start, img_offset,
                num_imgs, 'bright field')
        if img_start < 0 or num_imgs < 1:
            logging.error('Unable to find suitable bright field images')
            self.is_valid = False
        bright_field['img_start'] = img_start
        bright_field['img_offset'] = img_offset
        bright_field['num'] = num_imgs
        logging.debug(f'Bright field image start index: {bright_field["img_start"]}')
        logging.debug(f'Bright field image offset: {bright_field["img_offset"]}')
        logging.debug(f'Number of bright field images: {bright_field["num"]}')

        # Find tomography images
        for i,stack in enumerate(stack_info['stacks']):
            # Check if stack is already loaded or available
            if self.tomo_stacks[i].size or available_stacks[i]:
                continue
            index = stack['index']
            img_start = stack.get('img_start', -1)
            img_offset = stack.get('img_offset', -1)
            num_imgs = stack.get('num', 0)
            if not self.test_mode:
                img_start, img_offset, num_imgs = selectImageRange(img_start, img_offset,
                        num_imgs, f'tomography stack {index}', num_thetas)
                if img_start < 0 or num_imgs != num_thetas:
                    logging.error('Unable to find suitable tomography images')
                    self.is_valid = False
            stack['img_start'] = img_start
            stack['img_offset'] = img_offset
            stack['num'] = num_imgs
            logging.debug(f'Tomography stack {index} image start index: {stack["img_start"]}')
            logging.debug(f'Tomography stack {index} image offset: {stack["img_offset"]}')
            logging.debug(f'Number of tomography images for stack {index}: {stack["num"]}')

        # Safe updated config to file
        if self.is_valid:
            self.cf.saveFile(self.config_out)

        return

    def _genDark(self, tdf_files):
        """Generate dark field.
        """
        # Load the dark field images
        logging.debug('Loading dark field...')
        dark_field = self.config['dark_field']
        tdf_stack = loadImageStack(tdf_files, self.config['data_filetype'],
                dark_field['img_offset'], dark_field['num'])

        # Take median
        self.tdf = np.median(tdf_stack, axis=0)
        del tdf_stack

        # Remove dark field intensities above the cutoff
        tdf_cutoff = dark_field.get('cutoff')
        if tdf_cutoff is not None:
            if not is_num(tdf_cutoff, 0):
                logging.warning(f'Ignoring illegal value of tdf_cutoff {tdf_cutoff}')
            else:
                self.tdf[self.tdf > tdf_cutoff] = np.nan
                logging.debug(f'tdf_cutoff = {tdf_cutoff}')

        tdf_mean = np.nanmean(self.tdf)
        logging.debug(f'tdf_mean = {tdf_mean}')
        np.nan_to_num(self.tdf, copy=False, nan=tdf_mean, posinf=tdf_mean, neginf=0.)
        if self.galaxy_flag:
            quickImshow(self.tdf, title='dark field', path='setup_pngs',
                    save_fig=True, save_only=True)
        elif not self.test_mode:
            quickImshow(self.tdf, title='dark field', path=self.output_folder,
                    save_fig=self.save_plots, save_only=self.save_plots_only)

    def _genBright(self, tbf_files):
        """Generate bright field.
        """
        # Load the bright field images
        logging.debug('Loading bright field...')
        bright_field = self.config['bright_field']
        tbf_stack = loadImageStack(tbf_files, self.config['data_filetype'],
                bright_field['img_offset'], bright_field['num'])

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
            logging.warning('Dark field unavailable')
        if self.galaxy_flag:
            quickImshow(self.tbf, title='bright field', path='setup_pngs',
                    save_fig=True, save_only=True)
        elif not self.test_mode:
            quickImshow(self.tbf, title='bright field', path=self.output_folder,
                    save_fig=self.save_plots, save_only=self.save_plots_only)

    def _setDetectorBounds(self, tomo_stack_files):
        """Set vertical detector bounds for image stack.
        """
        preprocess = self.config.get('preprocess')
        if preprocess is None:
            img_x_bounds = [None, None]
        else:
            img_x_bounds = preprocess.get('img_x_bounds', [0, self.tbf.shape[0]])
            if img_x_bounds[0] is not None and img_x_bounds[1] is not None:
                if img_x_bounds[0] < 0:
                    illegal_value(img_x_bounds[0], 'preprocess:img_x_bounds[0]', 'config file')
                    img_x_bounds[0] = 0
                if not is_index_range(img_x_bounds, 0, self.tbf.shape[0]):
                    illegal_value(img_x_bounds[1], 'preprocess:img_x_bounds[1]', 'config file')
                    img_x_bounds[1] = self.tbf.shape[0]
        if self.test_mode:
            # Update config and save to file
            if preprocess is None:
                self.cf.config['preprocess'] = {'img_x_bounds' : [0, self.tbf.shape[0]]}
            else:
                preprocess['img_x_bounds'] = img_x_bounds
            self.cf.saveFile(self.config_out)
            return

        # Check reference heights
        pixel_size = self.config['detector']['pixel_size']
        if pixel_size is None:
            raise ValueError('Detector pixel size unavailable')
        if not self.tbf.size:
            raise ValueError('Bright field unavailable')
        num_x_min = None
        num_tomo_stacks = self.config['stack_info']['num']
        stacks = self.config['stack_info']['stacks']
        if num_tomo_stacks > 1:
            delta_z = stacks[1]['ref_height']-stacks[0]['ref_height']
            for i in range(2, num_tomo_stacks):
                delta_z = min(delta_z, stacks[i]['ref_height']-stacks[i-1]['ref_height'])
            logging.debug(f'delta_z = {delta_z}')
            num_x_min = int((delta_z-0.5*pixel_size)/pixel_size)
            logging.debug(f'num_x_min = {num_x_min}')
            if num_x_min > self.tbf.shape[0]:
                logging.warning('Image bounds and pixel size prevent seamless stacking')
                num_x_min = None

        # Select image bounds
        if self.galaxy_flag:
            x_sum = np.sum(self.tbf, 1)
            x_sum_min = x_sum.min()
            x_sum_max = x_sum.max()
            x_low = 0
            x_upp = x_sum.size
            if num_x_min is not None:
                fit = Fit.fit_data(np.array(range(len(x_sum))), x_sum, models='rectangle',
                        form='atan', guess=True)
                parameters = fit.best_values
                x_low = parameters.get('center1', None)
                x_upp = parameters.get('center2', None)
                sig_low = parameters.get('sigma1', None)
                sig_upp = parameters.get('sigma2', None)
                if (x_low is not None and x_upp is not None and sig_low is not None and
                        sig_upp is not None and 0 <= x_low < x_upp <= x_sum.size and
                        (sig_low+sig_upp)/(x_upp-x_low) < 0.1):
                    if num_tomo_stacks == 1 or num_x_min is None:
                        x_low = int(x_low-(x_upp-x_low)/10)
                        x_upp = int(x_upp+(x_upp-x_low)/10)
                    else:
                        x_low = int((x_low+x_upp)/2-num_x_min/2)
                        x_upp = x_low+num_x_min
                    if x_low < 0:
                        x_low = 0
                    if x_upp > x_sum.size:
                        x_upp = x_sum.size
                else:
                    x_low = 0
                    x_upp = x_sum.size
            quickPlot((range(x_sum.size), x_sum),
                    ([x_low, x_low], [x_sum_min, x_sum_max], 'r-'),
                    ([x_upp-1, x_upp-1], [x_sum_min, x_sum_max], 'r-'),
                    title=f'sum bright field over theta/y (row bounds: [{x_low}, {x_upp}])',
                    path='setup_pngs', name='detectorbounds.png', save_fig=True, save_only=True,
                    show_grid=True)
            for i,stack in enumerate(stacks):
                tomo_stack = loadImageStack(tomo_stack_files[i], self.config['data_filetype'],
                    stack['img_offset'], 1)
                tomo_stack = tomo_stack[0,:,:]
                if num_x_min is not None:
                    tomo_stack_max = tomo_stack.max()
                    tomo_stack[x_low,:] = tomo_stack_max
                    tomo_stack[x_upp-1,:] = tomo_stack_max
                title = f'tomography image at theta={self.config["theta_range"]["start"]}'
                quickImshow(tomo_stack, title=title, path='setup_pngs',
                        name=f'tomo_{stack["index"]}.png', save_fig=True, save_only=True,
                        show_grid=True)
                del tomo_stack
            
            # Update config and save to file
            img_x_bounds = [x_low, x_upp]
            logging.debug(f'img_x_bounds: {img_x_bounds}')
            if preprocess is None:
                self.cf.config['preprocess'] = {'img_x_bounds' : img_x_bounds}
            else:
                preprocess['img_x_bounds'] = img_x_bounds
            self.cf.saveFile(self.config_out)
            del x_sum
            return

        # For one tomography stack only: load the first image
        title = None
        quickImshow(self.tbf, title='bright field')
        if num_tomo_stacks == 1:
            tomo_stack = loadImageStack(tomo_stack_files[0], self.config['data_filetype'],
                stacks[0]['img_offset'], 1)
            title = f'tomography image at theta={self.config["theta_range"]["start"]}'
            quickImshow(tomo_stack[0,:,:], title=title)
            tomo_or_bright = input_menu(['bright field', 'first tomography image'], 
                    header='\nSelect image bounds from')
        else:
            print('\nSelect image bounds from bright field')
            tomo_or_bright = 0
        if tomo_or_bright:
            x_sum = np.sum(tomo_stack[0,:,:], 1)
            use_bounds = False
            if img_x_bounds[0] is not None and img_x_bounds[1] is not None:
                tmp = np.copy(tomo_stack[0,:,:])
                tmp_max = tmp.max()
                tmp[img_x_bounds[0],:] = tmp_max
                tmp[img_x_bounds[1]-1,:] = tmp_max
                title = f'tomography image at theta={self.config["theta_range"]["start"]}'
                quickImshow(tmp, title=title)
                del tmp
                x_sum_min = x_sum.min()
                x_sum_max = x_sum.max()
                quickPlot((range(x_sum.size), x_sum),
                        ([img_x_bounds[0], img_x_bounds[0]], [x_sum_min, x_sum_max], 'r-'),
                        ([img_x_bounds[1]-1, img_x_bounds[1]-1], [x_sum_min, x_sum_max], 'r-'),
                        title='sum over theta and y')
                print(f'lower bound = {img_x_bounds[0]} (inclusive)\n'+
                        f'upper bound = {img_x_bounds[1]} (exclusive)]')
                use_bounds =  input_yesno('Accept these bounds (y/n)?', 'y')
            if not use_bounds:
                img_x_bounds = list(selectImageBounds(tomo_stack[0,:,:], 0,
                        img_x_bounds[0], img_x_bounds[1], num_x_min,
                        f'tomography image at theta={self.config["theta_range"]["start"]}'))
                if num_x_min is not None and img_x_bounds[1]-img_x_bounds[0]+1 < num_x_min:
                    logging.warning('Image bounds and pixel size prevent seamless stacking')
                title = f'tomography image at theta={self.config["theta_range"]["start"]}'
                quickImshow(tomo_stack[0,:,:], title=title, path=self.output_folder,
                        save_fig=self.save_plots, save_only=True)
                quickPlot(range(img_x_bounds[0], img_x_bounds[1]),
                        x_sum[img_x_bounds[0]:img_x_bounds[1]],
                        title='sum over theta and y', path=self.output_folder,
                        save_fig=self.save_plots, save_only=True)
        else:
            x_sum = np.sum(self.tbf, 1)
            x_sum_min = x_sum.min()
            x_sum_max = x_sum.max()
            use_bounds = False
            if img_x_bounds[0] is not None and img_x_bounds[1] is not None:
                tmp = np.copy(self.tbf)
                tmp_max = tmp.max()
                tmp[img_x_bounds[0],:] = tmp_max
                tmp[img_x_bounds[1]-1,:] = tmp_max
                title = 'bright field'
                quickImshow(tmp, title=title)
                del tmp
                quickPlot((range(x_sum.size), x_sum),
                        ([img_x_bounds[0], img_x_bounds[0]], [x_sum_min, x_sum_max], 'r-'),
                        ([img_x_bounds[1]-1, img_x_bounds[1]-1], [x_sum_min, x_sum_max], 'r-'),
                        title='sum over theta and y')
                print(f'lower bound = {img_x_bounds[0]} (inclusive)\n'+
                        f'upper bound = {img_x_bounds[1]} (exclusive)]')
                use_bounds =  input_yesno('Accept these bounds (y/n)?', 'y')
            if not use_bounds:
                use_fit = False
                fit = Fit.fit_data(np.array(range(len(x_sum))), x_sum, models='rectangle',
                        form='atan', guess=True)
                parameters = fit.best_values
                x_low = parameters.get('center1', None)
                x_upp = parameters.get('center2', None)
                sig_low = parameters.get('sigma1', None)
                sig_upp = parameters.get('sigma2', None)
                if (x_low is not None and x_upp is not None and sig_low is not None and
                        sig_upp is not None and 0 <= x_low < x_upp <= x_sum.size and
                        (sig_low+sig_upp)/(x_upp-x_low) < 0.1):
                    if num_tomo_stacks == 1 or num_x_min is None:
                        x_low = int(x_low-(x_upp-x_low)/10)
                        x_upp = int(x_upp+(x_upp-x_low)/10)
                    else:
                        x_low = int((x_low+x_upp)/2-num_x_min/2)
                        x_upp = x_low+num_x_min
                    if x_low < 0:
                        x_low = 0
                    if x_upp > x_sum.size:
                        x_upp = x_sum.size
                    tmp = np.copy(self.tbf)
                    tmp_max = tmp.max()
                    tmp[x_low,:] = tmp_max
                    tmp[x_upp-1,:] = tmp_max
                    title = 'bright field'
                    quickImshow(tmp, title=title)
                    del tmp
                    quickPlot((range(x_sum.size), x_sum),
                            ([x_low, x_low], [x_sum_min, x_sum_max], 'r-'),
                            ([x_upp, x_upp], [x_sum_min, x_sum_max], 'r-'),
                            title='sum over theta and y')
                    print(f'lower bound = {x_low} (inclusive)')
                    print(f'upper bound = {x_upp} (exclusive)]')
                    use_fit =  input_yesno('Accept these bounds (y/n)?', 'y')
                if use_fit:
                    img_x_bounds = [x_low, x_upp]
                else:
                    accept = False
                    while not accept:
                        mask, img_x_bounds = draw_mask_1d(x_sum, title='select x data range',
                                legend='sum over theta and y')
                        print(f'img_x_bounds = {img_x_bounds}')
                        while (len(img_x_bounds) != 1 or (len(x_sum) >= num_x_min and
                                img_x_bounds[0][1]-img_x_bounds[0][0]+1 < num_x_min)):
                            exit('Should not be here')
                            print('Please select exactly one continuous range')
                            mask, img_x_bounds = draw_mask_1d(x_sum, title='select x data range',
                                    legend='sum over theta and y')
                        img_x_bounds = list(img_x_bounds[0])
                        quickPlot(x_sum, vlines=img_x_bounds, title='sum over theta and y')
                        print(f'img_x_bounds = {img_x_bounds} (lower bound inclusive, upper bound '+
                                'exclusive)')
                        accept = input_yesno('Accept these bounds (y/n)?', 'y')
                        if not accept:
                            img_x_bounds = None
                if num_x_min is not None and img_x_bounds[1]-img_x_bounds[0]+1 < num_x_min:
                    logging.warning('Image bounds and pixel size prevent seamless stacking')
                #quickPlot(range(img_x_bounds[0], img_x_bounds[1]),
                #        x_sum[img_x_bounds[0]:img_x_bounds[1]],
                #        title='sum over theta and y', path=self.output_folder,
                #        save_fig=self.save_plots, save_only=True)
                quickPlot((range(x_sum.size), x_sum),
                        ([img_x_bounds[0], img_x_bounds[0]], [x_sum_min, x_sum_max], 'r-'),
                        ([img_x_bounds[1], img_x_bounds[1]], [x_sum_min, x_sum_max], 'r-'),
                        title='sum over theta and y', path=self.output_folder,
                        save_fig=self.save_plots, save_only=True)
            del x_sum
            for i,stack in enumerate(stacks):
                tomo_stack = loadImageStack(tomo_stack_files[i], self.config['data_filetype'],
                    stack['img_offset'], 1)
                tomo_stack = tomo_stack[0,:,:]
                if num_x_min is not None:
                    tomo_stack_max = tomo_stack.max()
                    tomo_stack[img_x_bounds[0],:] = tomo_stack_max
                    tomo_stack[img_x_bounds[1]-1,:] = tomo_stack_max
                title = f'tomography image at theta={self.config["theta_range"]["start"]}'
                if self.galaxy_flag:
                    quickImshow(tomo_stack, title=title, path='setup_pngs',
                            name=f'tomo_{stack["index"]}.png', save_fig=True, save_only=True,
                            show_grid=True)
                else:
                    quickImshow(tomo_stack, title=title, path=self.output_folder,
                            name=f'tomo_{stack["index"]}.png', save_fig=self.save_plots,
                            save_only=True, show_grid=True)
                del tomo_stack
        logging.debug(f'img_x_bounds: {img_x_bounds}')

        if self.save_plots_only:
            clearImshow('bright field')
            clearPlot('sum over theta and y')
            if title:
                clearPlot(title)

        # Update config and save to file
        if preprocess is None:
            self.cf.config['preprocess'] = {'img_x_bounds' : img_x_bounds}
        else:
            preprocess['img_x_bounds'] = img_x_bounds
        self.cf.saveFile(self.config_out)

    def _setZoomOrSkip(self):
        """Set zoom and/or theta skip to reduce memory the requirement for the analysis.
        """
        preprocess = self.config.get('preprocess')
        zoom_perc = 100
        if not self.galaxy_flag:
            if preprocess is None or 'zoom_perc' not in preprocess:
                if input_yesno('\nDo you want to zoom in to reduce memory requirement (y/n)?', 'n'):
                    zoom_perc = input_int('    Enter zoom percentage', 1, 100)
            else:
                zoom_perc = preprocess['zoom_perc']
                if is_num(zoom_perc, 1., 100.):
                    zoom_perc = int(zoom_perc)
                else:
                    illegal_value(zoom_perc, 'preprocess:zoom_perc', 'config file')
                    zoom_perc = 100
        num_theta_skip = 0
        if not self.galaxy_flag:
            if preprocess is None or 'num_theta_skip' not in preprocess:
                if input_yesno('Do you want to skip thetas to reduce memory requirement (y/n)?',
                        'n'):
                    num_theta_skip = input_int('    Enter the number skip theta interval', 0,
                            self.num_thetas-1)
            else:
                num_theta_skip = preprocess['num_theta_skip']
                if not is_int(num_theta_skip, 0):
                    illegal_value(num_theta_skip, 'preprocess:num_theta_skip', 'config file')
                    num_theta_skip = 0
        logging.debug(f'zoom_perc = {zoom_perc}')
        logging.debug(f'num_theta_skip = {num_theta_skip}')

        # Update config and save to file
        if preprocess is None:
            self.cf.config['preprocess'] = {'zoom_perc' : zoom_perc,
                    'num_theta_skip' : num_theta_skip}
        else:
            preprocess['zoom_perc'] = zoom_perc
            preprocess['num_theta_skip'] = num_theta_skip
        self.cf.saveFile(self.config_out)

    def _loadTomo(self, base_name, index, required=False):
        """Load a tomography stack.
        """
        # stack order: row,theta,column
        zoom_perc = None
        preprocess = self.config.get('preprocess')
        if preprocess:
            zoom_perc = preprocess.get('zoom_perc')
        if zoom_perc is None or zoom_perc == 100:
            title = f'{base_name} fullres'
        else:
            title = f'{base_name} {zoom_perc}p'
        title += f'_{index}'
        tomo_file = re.sub(r"\s+", '_', f'{self.output_folder}/{title}.npy')
        load_flag = False
        available = False
        if os.path.isfile(tomo_file):
            available = True
            if required:
                load_flag = True
            else:
                load_flag = input_yesno(f'\nDo you want to load {tomo_file} (y/n)?')
        stack = np.array([])
        if load_flag:
            t0 = time()
            logging.info(f'Loading {tomo_file} ...')
            try:
                stack = np.load(tomo_file)
            except IOError or ValueError:
                stack = np.array([])
                logging.error(f'Error loading {tomo_file}')
            logging.info(f'... done in {time()-t0:.2f} seconds!')
        if stack.size:
            quickImshow(stack[:,0,:], title=title, path=self.output_folder,
                    save_fig=self.save_plots, save_only=self.save_plots_only)
        return stack, available

    def _saveTomo(self, base_name, stack, index=None):
        """Save a tomography stack.
        """
        zoom_perc = None
        preprocess = self.config.get('preprocess')
        if preprocess:
            zoom_perc = preprocess.get('zoom_perc')
        if zoom_perc is None or zoom_perc == 100:
            title = f'{base_name} fullres'
        else:
            title = f'{base_name} {zoom_perc}p'
        if index:
            title += f'_{index}'
        tomo_file = re.sub(r"\s+", '_', f'{self.output_folder}/{title}.npy')
        t0 = time()
        logging.info(f'Saving {tomo_file} ...')
        np.save(tomo_file, stack)
        logging.info(f'... done in {time()-t0:.2f} seconds!')

    def _genTomo(self, tomo_stack_files, available_stacks, num_core=None):
        """Generate tomography fields.
        """
        if num_core is None:
            num_core = self.num_core
        stacks = self.config['stack_info']['stacks']
        assert(len(self.tomo_stacks) == self.config['stack_info']['num'])
        assert(len(self.tomo_stacks) == len(stacks))
        if len(available_stacks) != len(stacks):
            logging.warning('Illegal dimension of available_stacks in _genTomo'+
                    f'({len(available_stacks)}');
            available_stacks = [False]*self.num_tomo_stacks

        preprocess = self.config.get('preprocess')
        if preprocess is None:
            img_x_bounds = [0, self.tbf.shape[0]]
            img_y_bounds = [0, self.tbf.shape[1]]
            zoom_perc = 100
            num_theta_skip = 0
        else:
            img_x_bounds = preprocess.get('img_x_bounds', [0, self.tbf.shape[0]])
            img_y_bounds = preprocess.get('img_y_bounds', [0, self.tbf.shape[1]])
            zoom_perc = preprocess.get('zoom_perc', 100)
            num_theta_skip = preprocess.get('num_theta_skip', 0)

        if self.tdf.size:
            tdf = self.tdf[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]
        else:
            logging.warning('Dark field unavailable')
        if not self.tbf.size:
            raise ValueError('Bright field unavailable')
        tbf = self.tbf[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]

        for i,stack in enumerate(stacks):
            # Check if stack is already loaded or available
            if self.tomo_stacks[i].size or available_stacks[i]:
                continue

            # Load a stack of tomography images
            index = stack['index']
            t0 = time()
            tomo_stack = loadImageStack(tomo_stack_files[i], self.config['data_filetype'],
                    stack['img_offset'], self.config['theta_range']['num'], num_theta_skip,
                    img_x_bounds, img_y_bounds)
            tomo_stack = tomo_stack.astype('float64')
            logging.debug(f'loading stack {index} took {time()-t0:.2f} seconds!')

            # Subtract dark field
            if self.tdf.size:
                t0 = time()
                with set_numexpr_threads(self.num_core):
                    ne.evaluate('tomo_stack-tdf', out=tomo_stack)
                logging.debug(f'subtracting dark field took {time()-t0:.2f} seconds!')

            # Normalize
            t0 = time()
            with set_numexpr_threads(self.num_core):
                ne.evaluate('tomo_stack/tbf', out=tomo_stack, truediv=True)
            logging.debug(f'normalizing took {time()-t0:.2f} seconds!')

            # Remove non-positive values and linearize data
            t0 = time()
            cutoff = 1.e-6
            with set_numexpr_threads(self.num_core):
                ne.evaluate('where(tomo_stack<cutoff, cutoff, tomo_stack)', out=tomo_stack)
            with set_numexpr_threads(self.num_core):
                ne.evaluate('-log(tomo_stack)', out=tomo_stack)
            logging.debug('removing non-positive values and linearizing data took '+
                    f'{time()-t0:.2f} seconds!')

            # Get rid of nans/infs that may be introduced by normalization
            t0 = time()
            np.where(np.isfinite(tomo_stack), tomo_stack, 0.)
            logging.debug(f'remove nans/infs took {time()-t0:.2f} seconds!')

            # Downsize tomography stack to smaller size
            tomo_stack = tomo_stack.astype('float32')
            if not self.galaxy_flag:
                title = f'red stack fullres {index}'
                if not self.test_mode:
                    quickImshow(tomo_stack[0,:,:], title=title, path=self.output_folder,
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
                if not self.galaxy_flag:
                    title = f'red stack {zoom_perc}p {index}'
                    if not self.test_mode:
                        quickImshow(tomo_stack[0,:,:], title=title, path=self.output_folder,
                                save_fig=self.save_plots, save_only=self.save_plots_only)

            # Convert tomography stack from theta,row,column to row,theta,column
            tomo_stack = np.swapaxes(tomo_stack, 0, 1)

            # Save tomography stack to file
            if not self.galaxy_flag:
                if not self.test_mode:
                    self._saveTomo('red stack', tomo_stack, index)
                else:
                    np.savetxt(f'{self.output_folder}/red_stack_{index}.txt',
                            tomo_stack[0,:,:], fmt='%.6e')
                
            # Combine stacks
            t0 = time()
            self.tomo_stacks[i] = tomo_stack
            logging.debug(f'combining nstack took {time()-t0:.2f} seconds!')

            # Update config and save to file
            stack['preprocessed'] = True
            stack.pop('reconstructed', 'reconstructed not found')
            find_center = self.config.get('find_center')
            if find_center:
                find_center.pop('completed', 'completed not found')
                if self.test_mode:
                    find_center.pop('lower_center_offset', 'lower_center_offset not found')
                    find_center.pop('upper_center_offset', 'upper_center_offset not found')
            self.cf.saveFile(self.config_out)

        if self.tdf.size:
            del tdf
        del tbf

    def _reconstructOnePlane(self, tomo_plane_T, center, thetas_deg, eff_pixel_size,
            cross_sectional_dim, plot_sinogram=True, num_core=1):
        """Invert the sinogram for a single tomography plane.
        """
        # tomo_plane_T index order: column,theta
        assert(0 <= center < tomo_plane_T.shape[0])
        center_offset = center-tomo_plane_T.shape[0]/2
        two_offset = 2*int(np.round(center_offset))
        two_offset_abs = np.abs(two_offset)
        max_rad = int(0.5*(cross_sectional_dim/eff_pixel_size)*1.1) # 10% slack to avoid edge effects
        if max_rad > 0.5*tomo_plane_T.shape[0]:
            max_rad = 0.5*tomo_plane_T.shape[0]
        dist_from_edge = max(1, int(np.floor((tomo_plane_T.shape[0]-two_offset_abs)/2.)-max_rad))
        if two_offset >= 0:
            logging.debug(f'sinogram range = [{two_offset+dist_from_edge}, {-dist_from_edge}]')
            sinogram = tomo_plane_T[two_offset+dist_from_edge:-dist_from_edge,:]
        else:
            logging.debug(f'sinogram range = [{dist_from_edge}, {two_offset-dist_from_edge}]')
            sinogram = tomo_plane_T[dist_from_edge:two_offset-dist_from_edge,:]
        if plot_sinogram and not self.test_mode:
            quickImshow(sinogram.T, f'sinogram center offset{center_offset:.2f}',
                    path=self.output_folder, save_fig=self.save_plots,
                    save_only=self.save_plots_only, aspect='auto')

        # Inverting sinogram
        t0 = time()
        recon_sinogram = iradon(sinogram, theta=thetas_deg, circle=True)
        logging.debug(f'inverting sinogram took {time()-t0:.2f} seconds!')
        del sinogram

        # Performing Gaussian filtering and removing ring artifacts
        recon_parameters = self.config.get('recon_parameters')
        if recon_parameters is None:
            sigma = 1.0
            ring_width = 15
        else:
            sigma = recon_parameters.get('gaussian_sigma', 1.0)
            if not is_num(sigma, 0.0):
                logging.warning(f'Illegal gaussian_sigma ({sigma}) in _reconstructOnePlane, '+
                        'set to a default value of 1.0')
                sigma = 1.0
            ring_width = recon_parameters.get('ring_width', 15)
            if not is_int(ring_width, 0):
                logging.warning(f'Illegal ring_width ({ring_width}) in _reconstructOnePlane, '+
                        'set to a default value of 15')
                ring_width = 15
        t0 = time()
        recon_sinogram = spi.gaussian_filter(recon_sinogram, sigma, mode='nearest')
        recon_clean = np.expand_dims(recon_sinogram, axis=0)
        del recon_sinogram
        t1 = time()
        logging.debug(f'running remove_ring on {num_core} cores ...')
        recon_clean = tomopy.misc.corr.remove_ring(recon_clean, rwidth=ring_width, ncore=num_core)
        logging.debug(f'... remove_ring took {time()-t1:.2f} seconds!')
        logging.debug(f'filtering and removing ring artifact took {time()-t0:.2f} seconds!')
        return recon_clean

    def _plotEdgesOnePlane(self, recon_plane, title, path=None):
        vis_parameters = self.config.get('vis_parameters')
        if vis_parameters is None:
            weight = 0.1
        else:
            weight = vis_parameters.get('denoise_weight', 0.1)
            if not is_num(weight, 0.0):
                logging.warning(f'Illegal weight ({weight}) in _plotEdgesOnePlane, '+
                        'set to a default value of 0.1')
                weight = 0.1
        edges = denoise_tv_chambolle(recon_plane, weight=weight)
        vmax = np.max(edges[0,:,:])
        vmin = -vmax
        if self.galaxy_flag:
            quickImshow(edges[0,:,:], title, path=path, save_fig=True, save_only=True,
                    cmap='gray', vmin=vmin, vmax=vmax)
        else:
            if path is None:
                path=self.output_folder
            quickImshow(edges[0,:,:], f'{title} coolwarm', path=path,
                    save_fig=self.save_plots, save_only=self.save_plots_only, cmap='coolwarm')
            quickImshow(edges[0,:,:], f'{title} gray', path=path,
                    save_fig=self.save_plots, save_only=self.save_plots_only, cmap='gray',
                    vmin=vmin, vmax=vmax)
        del edges

    def _findCenterOnePlane(self, sinogram, row, thetas_deg, eff_pixel_size, cross_sectional_dim,
            tol=0.1, num_core=1, galaxy_param=None):
        """Find center for a single tomography plane.
        """
        if self.galaxy_flag:
            assert(isinstance(galaxy_param, dict))
            if not os.path.exists('find_center_pngs'):
                os.mkdir('find_center_pngs')
        # sinogram index order: theta,column
        # need index order column,theta for iradon, so take transpose
        sinogram_T = sinogram.T
        center = sinogram.shape[1]/2

        # try automatic center finding routines for initial value
        t0 = time()
        if num_core > num_core_tomopy_limit:
            logging.debug(f'running find_center_vo on {num_core_tomopy_limit} cores ...')
            tomo_center = tomopy.find_center_vo(sinogram, ncore=num_core_tomopy_limit)
        else:
            logging.debug(f'running find_center_vo on {num_core} cores ...')
            tomo_center = tomopy.find_center_vo(sinogram, ncore=num_core)
        logging.debug(f'... find_center_vo took {time()-t0:.2f} seconds!')
        center_offset_vo = tomo_center-center
        if self.test_mode:
            logging.info(f'Center at row {row} using Nghia Vo’s method = {center_offset_vo:.2f}')
            del sinogram_T
            return float(center_offset_vo)
        elif self.galaxy_flag:
            logging.info(f'Center at row {row} using Nghia Vo’s method = {center_offset_vo:.2f}')
            t0 = time()
            logging.debug(f'running _reconstructOnePlane on {num_core} cores ...')
            recon_plane = self._reconstructOnePlane(sinogram_T, tomo_center, thetas_deg,
                    eff_pixel_size, cross_sectional_dim, False, num_core)
            logging.debug(f'... _reconstructOnePlane took {time()-t0:.2f} seconds!')
            title = f'edges row{row} center offset{center_offset_vo:.2f} Vo'
            self._plotEdgesOnePlane(recon_plane, title, path='find_center_pngs')
            del recon_plane
            if not galaxy_param['center_type_selector']:
                del sinogram_T
                return float(center_offset_vo)
        else:
            print(f'Center at row {row} using Nghia Vo’s method = {center_offset_vo:.2f}')
            recon_plane = self._reconstructOnePlane(sinogram_T, tomo_center, thetas_deg,
                    eff_pixel_size, cross_sectional_dim, False, num_core)
            title = f'edges row{row} center offset{center_offset_vo:.2f} Vo'
            self._plotEdgesOnePlane(recon_plane, title)
        if not self.galaxy_flag:
            if input_yesno('Try finding center using phase correlation (y/n)?', 'n'):
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
                recon_plane = self._reconstructOnePlane(sinogram_T, tomo_center, thetas_deg,
                        eff_pixel_size, cross_sectional_dim, False, num_core)
                title = f'edges row{row} center_offset{center_offset:.2f} PC'
                self._plotEdgesOnePlane(recon_plane, title)
            if input_yesno('Accept a center location (y) or continue search (n)?', 'y'):
                center_offset = input_num('    Enter chosen center offset', -center, center,
                        center_offset_vo)
                del sinogram_T
                del recon_plane
                return float(center_offset)

        # perform center finding search
        while True:
            if self.galaxy_flag and galaxy_param and galaxy_param['center_type_selector']:
                set_center = center_offset_vo
                if galaxy_param['center_type_selector'] == 'user':
                    set_center = galaxy_param['set_center']
                set_range = galaxy_param['set_range']
                center_offset_step = galaxy_param['set_step']
                if (not is_num(set_range, 0) or not is_num(center_offset_step) or
                        center_offset_step <= 0):
                    logging.warning('Illegal center finding search parameter, skip search')
                    del sinogram_T
                    return float(center_offset_vo)
                set_range = center_offset_step*max(1, int(set_range/center_offset_step))
                center_offset_low = set_center-set_range
                center_offset_upp = set_center+set_range
            else:
                center_offset_low = input_int('\nEnter lower bound for center offset',
                        -center, center)
                center_offset_upp = input_int('Enter upper bound for center offset',
                        center_offset_low, center)
                if center_offset_upp == center_offset_low:
                    center_offset_step = 1
                else:
                    center_offset_step = input_int('Enter step size for center offset search',
                            1, center_offset_upp-center_offset_low)
            num_center_offset = 1+int((center_offset_upp-center_offset_low)/center_offset_step)
            center_offsets = np.linspace(center_offset_low, center_offset_upp, num_center_offset)
            for center_offset in center_offsets:
                if center_offset == center_offset_vo:
                    continue
                t0 = time()
                logging.debug(f'running _reconstructOnePlane on {num_core} cores ...')
                recon_plane = self._reconstructOnePlane(sinogram_T, center_offset+center,
                        thetas_deg, eff_pixel_size, cross_sectional_dim, False, num_core)
                logging.debug(f'... _reconstructOnePlane took {time()-t0:.2f} seconds!')
                title = f'edges row{row} center_offset{center_offset:.2f}'
                if self.galaxy_flag:
                    self._plotEdgesOnePlane(recon_plane, title, path='find_center_pngs')
                else:
                    self._plotEdgesOnePlane(recon_plane, title)
            if self.galaxy_flag or input_int('\nContinue (0) or end the search (1)', 0, 1):
                break

        del sinogram_T
        del recon_plane
        if self.galaxy_flag:
            center_offset = center_offset_vo
        else:
            center_offset = input_num('    Enter chosen center offset', -center, center)
        return float(center_offset)

    def _reconstructOneTomoStack(self, tomo_stack, thetas, row_bounds=None, center_offsets=[],
            num_core=1, algorithm='gridrec'):
        """reconstruct a single tomography stack.
        """
        # stack order: row,theta,column
        # thetas must be in radians
        # centers_offset: tomography axis shift in pixels relative to column center
        # RV should we remove stripes?
        # https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html
        # RV should we remove rings?
        # https://tomopy.readthedocs.io/en/latest/api/tomopy.misc.corr.html
        # RV: Add an option to do (extra) secondary iterations later or to do some sort of convergence test?
        if row_bounds is None:
            row_bounds = [0, tomo_stack.shape[0]]
        else:
            if not is_index_range(row_bounds, 0, tomo_stack.shape[0]):
                raise ValueError('Illegal row bounds in reconstructOneTomoStack')
        if thetas.size != tomo_stack.shape[1]:
            raise ValueError('theta dimension mismatch in reconstructOneTomoStack')
        if not len(center_offsets):
            centers = np.zeros((row_bounds[1]-row_bounds[0]))
        elif len(center_offsets) == 2:
            centers = np.linspace(center_offsets[0], center_offsets[1],
                    row_bounds[1]-row_bounds[0])
        else:
            if center_offsets.size != row_bounds[1]-row_bounds[0]:
                raise ValueError('center_offsets dimension mismatch in reconstructOneTomoStack')
            centers = center_offsets
        centers += tomo_stack.shape[2]/2

        # Removing horizontal stripe and ring artifacts and perform secondary iterations
        recon_parameters = self.config.get('recon_parameters')
        if recon_parameters is None:
            sigma = 2.0
            secondary_iters = 0
            ring_width = 15
        else:
            sigma = recon_parameters.get('stripe_fw_sigma', 2.0)
            if not is_num(sigma, 0):
                logging.warning(f'Illegal stripe_fw_sigma ({sigma}) in '+
                        '_reconstructOneTomoStack, set to a default value of 2.0')
                ring_width = 15
            secondary_iters = recon_parameters.get('secondary_iters', 0)
            if not is_int(secondary_iters, 0):
                logging.warning(f'Illegal secondary_iters ({secondary_iters}) in '+
                        '_reconstructOneTomoStack, set to a default value of 0 (skip them)')
                ring_width = 0
            ring_width = recon_parameters.get('ring_width', 15)
            if not is_int(ring_width, 0):
                logging.warning(f'Illegal ring_width ({ring_width}) in _reconstructOnePlane, '+
                        'set to a default value of 15')
                ring_width = 15
        if True:
            t0 = time()
            if num_core > num_core_tomopy_limit:
                logging.debug('running remove_stripe_fw on {num_core_tomopy_limit} cores ...')
                tomo_stack = tomopy.prep.stripe.remove_stripe_fw(
                        tomo_stack[row_bounds[0]:row_bounds[1]], sigma=sigma,
                        ncore=num_core_tomopy_limit)
            else:
                logging.debug(f'running remove_stripe_fw on {num_core} cores ...')
                tomo_stack = tomopy.prep.stripe.remove_stripe_fw(
                        tomo_stack[row_bounds[0]:row_bounds[1]], sigma=sigma, ncore=num_core)
            logging.debug(f'... tomopy.prep.stripe.remove_stripe_fw took {time()-t0:.2f} seconds!')
        else:
            tomo_stack = tomo_stack[row_bounds[0]:row_bounds[1]]
        logging.debug('performing initial reconstruction')
        t0 = time()
        logging.debug(f'running recon on {num_core} cores ...')
        tomo_recon_stack = tomopy.recon(tomo_stack, thetas, centers, sinogram_order=True,
                algorithm=algorithm, ncore=num_core)
        logging.debug(f'... recon took {time()-t0:.2f} seconds!')
        if secondary_iters > 0:
            logging.debug(f'running {secondary_iters} secondary iterations')
            #options = {'method':'SIRT_CUDA', 'proj_type':'cuda', 'num_iter':secondary_iters}
            #RV: doesn't work for me: "Error: CUDA error 803: system has unsupported display driver /
            #                          cuda driver combination."
            #options = {'method':'SIRT', 'proj_type':'linear', 'MinConstraint': 0, 'num_iter':secondary_iters}
            #SIRT did not finish while running overnight
            #options = {'method':'SART', 'proj_type':'linear', 'num_iter':secondary_iters}
            options = {'method':'SART', 'proj_type':'linear', 'MinConstraint': 0,
                    'num_iter':secondary_iters}
            t0 = time()
            logging.debug(f'running recon on {num_core} cores ...')
            tomo_recon_stack  = tomopy.recon(tomo_stack, thetas, centers,
                    init_recon=tomo_recon_stack, options=options, sinogram_order=True,
                    algorithm=tomopy.astra, ncore=num_core)
            logging.debug(f'... recon took {time()-t0:.2f} seconds!')
        if True:
            t0 = time()
            logging.debug(f'running remove_ring on {num_core} cores ...')
            tomopy.misc.corr.remove_ring(tomo_recon_stack, rwidth=ring_width, out=tomo_recon_stack,
                    ncore=num_core)
            logging.debug(f'... remove_ring took {time()-t0:.2f} seconds!')
        return tomo_recon_stack

    def findImageFiles(self, tiff_to_h5_flag = False):
        """Find all available image files.
        """
        self.is_valid = True

        # Find dark field images
        dark_field = self.config['dark_field']
        img_start, num_imgs, dark_files = findImageFiles(
                dark_field['data_path'], self.config['data_filetype'], 'dark field')
        if img_start < 0 or num_imgs < 1:
            logging.error('Unable to find suitable dark field images')
            if dark_field['data_path']:
                self.is_valid = False
        img_start_old = dark_field.get('img_start')
        num_imgs_old = dark_field.get('num')
        if num_imgs_old is None:
            dark_field['num'] = num_imgs
        else:
            if num_imgs_old > num_imgs:
                logging.error('Inconsistent number of availaible dark field images')
                if dark_field['data_path']:
                    self.is_valid = False
        if img_start_old is None:
            dark_field['img_start'] = img_start
        else:
            if img_start_old < img_start:
                logging.error('Inconsistent image start index for dark field images')
                if dark_field['data_path']:
                    self.is_valid = False
        logging.info(f'Number of dark field images = {dark_field["num"]}')
        logging.info(f'Dark field image start index = {dark_field["img_start"]}')
        if num_imgs and tiff_to_h5_flag and self.config['data_filetype'] == 'tif':
            dark_files = combine_tiffs_in_h5(dark_files, num_imgs,
                    f'{self.config["work_folder"]}/dark_field.h5')
            dark_field['data_path'] = dark_files[0]

        # Find bright field images
        bright_field = self.config['bright_field']
        img_start, num_imgs, bright_files  = findImageFiles(
                bright_field['data_path'], self.config['data_filetype'], 'bright field')
        if img_start < 0 or num_imgs < 1:
            logging.error('Unable to find suitable bright field images')
            self.is_valid = False
        img_start_old = bright_field.get('img_start')
        num_imgs_old = bright_field.get('num')
        if num_imgs_old is None:
            bright_field['num'] = num_imgs
        else:
            if num_imgs_old > num_imgs:
                logging.error('Inconsistent number of availaible bright field images')
                self.is_valid = False
        if img_start_old is None:
            bright_field['img_start'] = img_start
        else:
            if img_start_old < img_start:
                logging.warning('Inconsistent image start index for bright field images')
                self.is_valid = False
        logging.info(f'Number of bright field images = {bright_field["num"]}')
        logging.info(f'Bright field image start index = {bright_field["img_start"]}')
        if num_imgs and tiff_to_h5_flag and self.config['data_filetype'] == 'tif':
            bright_files = combine_tiffs_in_h5(bright_files, num_imgs,
                    f'{self.config["work_folder"]}/bright_field.h5')
            bright_field['data_path'] = bright_files[0]

        # Find tomography images
        tomo_stack_files = []
        for stack in self.config['stack_info']['stacks']:
            index = stack['index']
            img_start, num_imgs, tomo_files = findImageFiles(
                    stack['data_path'], self.config['data_filetype'], f'tomography set {index}')
            if img_start < 0 or num_imgs < 1:
                logging.error('Unable to find suitable tomography images')
                self.is_valid = False
            img_start_old = stack.get('img_start')
            num_imgs_old = stack.get('num')
            if num_imgs_old is None:
                stack['num'] = num_imgs
            else:
                if num_imgs_old > num_imgs:
                    logging.error('Inconsistent number of availaible tomography images')
                    self.is_valid = False
            if img_start_old is None:
                stack['img_start'] = img_start
            else:
                if img_start_old < img_start:
                    logging.warning('Inconsistent image start index for tomography images')
                    self.is_valid = False
            logging.info(f'Number of tomography images for set {index} = {stack["num"]}')
            logging.info(f'Tomography set {index} image start index = {stack["img_start"]}')
            if num_imgs and tiff_to_h5_flag and self.config['data_filetype'] == 'tif':
                tomo_files = combine_tiffs_in_h5(tomo_files, num_imgs,
                        f'{self.config["work_folder"]}/tomo_field_{index}.h5')
                stack['data_path'] = tomo_files[0]
            tomo_stack_files.append(tomo_files)
            del tomo_files

        # Safe updated config
        if tiff_to_h5_flag:
            self.config['data_filetype'] == 'h5'
        if self.is_valid:
            self.cf.saveFile(self.config_out)

        return dark_files, bright_files, tomo_stack_files

    def loadTomoStacks(self, input_name, recon_flag=False):
        """Load tomography stacks (only for Galaxy).
        """
        assert(self.galaxy_flag)
        t0 = time()
        logging.info(f'Loading preprocessed tomography stack from {input_name} ...')
        stack_info = self.config['stack_info']
        stacks = stack_info.get('stacks')
        assert(len(self.tomo_stacks) == stack_info['num'])
        with np.load(input_name) as f:
            if recon_flag:
                for i,stack in enumerate(stacks):
                    self.tomo_recon_stacks[i] = f[f'set_{stack["index"]}']
                    logging.info(f'loaded stack {i}: index = {stack["index"]}, shape = '+
                            f'{self.tomo_recon_stacks[i].shape}')
            else:
                for i,stack in enumerate(stacks):
                    self.tomo_stacks[i] = f[f'set_{stack["index"]}']
                    logging.info(f'loaded stack {i}: index = {stack["index"]}, shape = '+
                            f'{self.tomo_stacks[i].shape}')
        logging.info(f'... done in {time()-t0:.2f} seconds!')

    def genTomoStacks(self, galaxy_param=None, num_core=None):
        """Preprocess tomography images.
        """
        if num_core is None:
            num_core = self.num_core
        logging.info(f'num_core = {num_core}')
        # Try loading any already preprocessed stacks (skip in Galaxy)
        # preprocessed stack order for each one in stack: row,theta,column
        stack_info = self.config['stack_info']
        stacks = stack_info['stacks']
        num_tomo_stacks = stack_info['num']
        assert(num_tomo_stacks == len(self.tomo_stacks))
        available_stacks = [False]*num_tomo_stacks
        if self.galaxy_flag:
            assert(isinstance(galaxy_param, dict))
            tdf_files = galaxy_param['tdf_files']
            tbf_files = galaxy_param['tbf_files']
            tomo_stack_files = galaxy_param['tomo_stack_files']
            assert(num_tomo_stacks == len(tomo_stack_files))
            if not os.path.exists('setup_pngs'):
                os.mkdir('setup_pngs')
        else:
            if galaxy_param:
                logging.warning('Ignoring galaxy_param in genTomoStacks (only for Galaxy)')
                galaxy_param = None
            tdf_files, tbf_files, tomo_stack_files = self.findImageFiles()
            if not self.is_valid:
                return
            for i,stack in enumerate(stacks):
                if not self.tomo_stacks[i].size and stack.get('preprocessed', False):
                    self.tomo_stacks[i], available_stacks[i] = \
                            self._loadTomo('red stack', stack['index'])

        # Preprocess any unloaded stacks
        if False in available_stacks:
            logging.debug('Preprocessing tomography images')

            # Check required image files (skip in Galaxy)
            if not self.galaxy_flag:
                self._selectImageRanges(available_stacks)
                if not self.is_valid:
                    return

            # Generate dark field
            if tdf_files:
                self._genDark(tdf_files)

            # Generate bright field
            self._genBright(tbf_files)

            # Set vertical detector bounds for image stack
            self._setDetectorBounds(tomo_stack_files)

            # Set zoom and/or theta skip to reduce memory the requirement
            self._setZoomOrSkip()

            # Generate tomography fields
            self._genTomo(tomo_stack_files, available_stacks, num_core)

            # Save tomography stack to file
            if self.galaxy_flag:
                t0 = time()
                output_name = galaxy_param['output_name']
                logging.info(f'Saving preprocessed tomography stack to {output_name} ...')
                save_stacks = {f'set_{stack["index"]}':tomo_stack
                        for stack,tomo_stack in zip(stacks,self.tomo_stacks)}
                np.savez(output_name, **save_stacks)
                logging.info(f'... done in {time()-t0:.2f} seconds!')

        del available_stacks

        # Adjust sample reference height, update config and save to file
        preprocess = self.config.get('preprocess')
        if preprocess is None:
            img_x_bounds = [0, self.tbf.shape[0]]
        else:
            img_x_bounds = preprocess.get('img_x_bounds', [0, self.tbf.shape[0]])
        pixel_size = self.config['detector']['pixel_size']
        if pixel_size is None:
            raise ValueError('Detector pixel size unavailable')
        pixel_size *= img_x_bounds[0]
        for stack in stacks:
            stack['ref_height'] = stack['ref_height']+pixel_size
        self.cf.saveFile(self.config_out)

    def findCenters(self, galaxy_param=None, num_core=None):
        """Find rotation axis centers for the tomography stacks.
        """
        if num_core is None:
            num_core = self.num_core
        logging.info(f'num_core = {num_core}')
        logging.debug('Find centers for tomography stacks')
        stacks = self.config['stack_info']['stacks']
        available_stacks = [stack['index'] for stack in stacks if stack.get('preprocessed', False)]
        logging.debug('Available stacks: {available_stacks}')
        if self.galaxy_flag:
            row_bounds = galaxy_param['row_bounds']
            center_rows = galaxy_param['center_rows']
            center_type_selector = galaxy_param['center_type_selector']
            if center_type_selector:
                if center_type_selector == 'vo':
                    set_center = None
                elif center_type_selector == 'user':
                    set_center = galaxy_param['set_center']
                else:
                    logging.error('Illegal center_type_selector entry in galaxy_param '+
                            f'({center_type_selector})')
                    galaxy_param['center_type_selector'] = None
            logging.debug(f'row_bounds = {row_bounds}')
            logging.debug(f'center_rows = {center_rows}')
            logging.debug(f'center_type_selector = {center_type_selector}')
        else:
            if galaxy_param:
                logging.warning('Ignoring galaxy_param in findCenters (only for Galaxy)')
                galaxy_param = None

        # Check for valid available center stack index
        find_center = self.config.get('find_center')
        center_stack_index = None
        if find_center and 'center_stack_index' in find_center:
            center_stack_index = find_center['center_stack_index']
            if (not isinstance(center_stack_index, int) or
                    center_stack_index not in available_stacks):
                illegal_value(center_stack_index, 'find_center:center_stack_index', 'config file')
            else:
                if self.test_mode:
                    assert(find_center.get('completed', False) == False)
                else:
                    print('\nFound calibration center offset info for stack '+
                            f'{center_stack_index}')
                    if (input_yesno('Do you want to use this again (y/n)?', 'y')
                            and find_center.get('completed', False) == True):
                        return

        # Load the required preprocessed stack
        # preprocessed stack order: row,theta,column
        num_tomo_stacks = self.config['stack_info']['num']
        assert(len(stacks) == num_tomo_stacks)
        assert(len(self.tomo_stacks) == num_tomo_stacks)
        if num_tomo_stacks == 1:
            if not center_stack_index:
                center_stack_index = stacks[0]['index']
            elif center_stack_index != stacks[0]['index']:
                raise ValueError(f'Inconsistent center_stack_index {center_stack_index}')
            if not self.tomo_stacks[0].size:
                self.tomo_stacks[0], available = self._loadTomo('red stack', center_stack_index,
                    required=True)
            center_stack = self.tomo_stacks[0]
            if not center_stack.size:
                stacks[0]['preprocessed'] = False
                raise OSError('Unable to load the required preprocessed tomography stack')
            assert(stacks[0].get('preprocessed', False) == True)
        elif self.galaxy_flag:
            center_stack_index = stacks[int(num_tomo_stacks/2)]['index']
            tomo_stack_index = available_stacks.index(center_stack_index)
            center_stack = self.tomo_stacks[tomo_stack_index]
            if not center_stack.size:
                stacks[tomo_stack_index]['preprocessed'] = False
                raise OSError('Unable to load the required preprocessed tomography stack')
            assert(stacks[tomo_stack_index].get('preprocessed', False) == True)
        else:
            while True:
                if not center_stack_index:
                    center_stack_index = input_int('\nEnter tomography stack index to get '
                            f'rotation axis centers ({available_stacks})', inset=available_stacks)
                while center_stack_index not in available_stacks:
                    center_stack_index = input_int('\nEnter tomography stack index to get '
                            f'rotation axis centers ({available_stacks})', inset=available_stacks)
                tomo_stack_index = available_stacks.index(center_stack_index)
                if not self.tomo_stacks[tomo_stack_index].size:
                    self.tomo_stacks[tomo_stack_index], available = self._loadTomo(
                            'red stack', center_stack_index, required=True)
                center_stack = self.tomo_stacks[tomo_stack_index]
                if not center_stack.size:
                    stacks[tomo_stack_index]['preprocessed'] = False
                    logging.error(f'Unable to load the {center_stack_index}th '+
                        'preprocessed tomography stack, pick another one')
                else:
                    break
            assert(stacks[tomo_stack_index].get('preprocessed', False) == True)
        if find_center is None:
            self.config['find_center'] = {'center_stack_index' : center_stack_index}
            find_center = self.config['find_center']
        else:
            find_center['center_stack_index'] = center_stack_index
        if not self.galaxy_flag:
            row_bounds = find_center.get('row_bounds', None)
            center_rows = [find_center.get('lower_row', None),
                    find_center.get('upper_row', None)]
        if row_bounds is None:
            row_bounds = [0, center_stack.shape[0]]
        if row_bounds[0] == -1:
            row_bounds[0] = 0
        if row_bounds[1] == -1:
            row_bounds[1] = center_stack.shape[0]
        if center_rows[0] == -1:
            center_rows[0] = 0
        if center_rows[1] == -1:
            center_rows[1] = center_stack.shape[0]-1
        if not is_index_range(row_bounds, 0, center_stack.shape[0]):
            illegal_value(row_bounds, 'row_bounds', 'Tomo:findCenters')
            return

        # Set thetas (in degrees)
        theta_range = self.config['theta_range']
        theta_start = theta_range['start']
        theta_end = theta_range['end']
        num_theta = theta_range['num']
        num_theta_skip = self.config['preprocess'].get('num_theta_skip', 0)
        thetas_deg = np.linspace(theta_start, theta_end, int(num_theta/(num_theta_skip+1)),
            endpoint=False)

        # Get non-overlapping sample row boundaries
        zoom_perc = self.config['preprocess'].get('zoom_perc', 100)
        pixel_size = self.config['detector']['pixel_size']
        if pixel_size is None:
            raise ValueError('Detector pixel size unavailable')
        eff_pixel_size = 100.*pixel_size/zoom_perc
        logging.debug(f'eff_pixel_size = {eff_pixel_size}')
        if num_tomo_stacks == 1:
             accept = True
             if not self.test_mode and not self.galaxy_flag:
                 accept = False
                 print('\nSelect bounds for image reconstruction')
                 if is_index_range(row_bounds, 0, center_stack.shape[0]):
                     a_tmp = np.copy(center_stack[:,0,:])
                     a_tmp_max = a_tmp.max()
                     a_tmp[row_bounds[0],:] = a_tmp_max
                     a_tmp[row_bounds[1]-1,:] = a_tmp_max
                     print(f'lower bound = {row_bounds[0]} (inclusive)')
                     print(f'upper bound = {row_bounds[1]} (exclusive)')
                     quickImshow(a_tmp, title=f'center stack theta={theta_start}',
                         aspect='auto')
                     del a_tmp
                     accept = input_yesno('Accept these bounds (y/n)?', 'y')
             if accept:
                 n1 = row_bounds[0]
                 n2 = row_bounds[1]
             else:
                 n1, n2 = selectImageBounds(center_stack[:,0,:], 0,
                         title=f'center stack theta={theta_start}')
        else:
            tomo_ref_heights = [stack['ref_height'] for stack in stacks]
            n1 = int((1.+(tomo_ref_heights[0]+center_stack.shape[0]*eff_pixel_size-
                tomo_ref_heights[1])/eff_pixel_size)/2)
            n2 = center_stack.shape[0]-n1
        logging.debug(f'n1 = {n1}, n2 = {n2} (n2-n1) = {(n2-n1)*eff_pixel_size:.3f} mm')
        if not self.test_mode and not self.galaxy_flag:
            tmp = center_stack[:,0,:]
            tmp_max = tmp.max()
            tmp[n1,:] = tmp_max
            tmp[n2-1,:] = tmp_max
            if is_index_range(center_rows, 0, tmp.shape[0]):
                tmp[center_rows[0],:] = tmp_max
                tmp[center_rows[1]-1,:] = tmp_max
            extent = [0, tmp.shape[1], tmp.shape[0], 0]
            quickImshow(tmp, title=f'center stack theta={theta_start}',
                    path=self.output_folder, extent=extent, save_fig=self.save_plots,
                    save_only=self.save_plots_only, aspect='auto')
            del tmp
            #extent = [0, center_stack.shape[2], n2, n1]
            #quickImshow(center_stack[n1:n2,0,:], title=f'center stack theta={theta_start}',
            #        path=self.output_folder, extent=extent, save_fig=self.save_plots,
            #        save_only=self.save_plots_only, show_grid=True, aspect='auto')

        # Get cross sectional diameter in mm
        cross_sectional_dim = center_stack.shape[2]*eff_pixel_size
        logging.debug(f'cross_sectional_dim = {cross_sectional_dim}')

        # Determine center offset at sample row boundaries
        logging.info('Determine center offset at sample row boundaries')

        # Lower row center
        use_row = False
        use_center = False
        row = center_rows[0]
        if self.test_mode or self.galaxy_flag:
            assert(is_int(row, n1, n2-2))
        if is_int(row, n1, n2-2):
            if self.test_mode or self.galaxy_flag:
                use_row = True
            else:
                quickImshow(center_stack[:,0,:], title=f'theta={theta_start}', aspect='auto')
                use_row = input_yesno('\nCurrent row index for lower center = '
                        f'{row}, use this value (y/n)?', 'y')
                if self.save_plots_only:
                    clearImshow(f'theta={theta_start}')
                if use_row:
                    center_offset = find_center.get('lower_center_offset')
                    if is_num(center_offset):
                        use_center = input_yesno('Current lower center offset = '+
                                f'{center_offset}, use this value (y/n)?', 'y')
        if not use_center:
            if not use_row:
                if not self.test_mode:
                    quickImshow(center_stack[:,0,:], title=f'theta={theta_start}',
                            aspect='auto')
                row = input_int('\nEnter row index to find lower center', n1, n2-2, n1)
                if row == '':
                    row = n1
                if self.save_plots_only:
                    clearImshow(f'theta={theta_start}')
            # center_stack order: row,theta,column
            center_offset = self._findCenterOnePlane(center_stack[row,:,:], row, thetas_deg,
                    eff_pixel_size, cross_sectional_dim, num_core=num_core,
                    galaxy_param=galaxy_param)
        logging.info(f'lower_center_offset = {center_offset:.2f} {type(center_offset)}')

        # Update config and save to file
        find_center['row_bounds'] = [n1, n2]
        find_center['lower_row'] = row
        find_center['lower_center_offset'] = center_offset
        self.cf.saveFile(self.config_out)
        lower_row = row

        # Upper row center
        use_row = False
        use_center = False
        row = center_rows[1]
        if self.test_mode or self.galaxy_flag:
            assert(is_int(row, lower_row+1, n2-1))
        if is_int(row, lower_row+1, n2-1):
            if self.test_mode or self.galaxy_flag:
                use_row = True
            else:
                quickImshow(center_stack[:,0,:], title=f'theta={theta_start}', aspect='auto')
                use_row = input_yesno('\nCurrent row index for upper center = '
                        f'{row}, use this value (y/n)?', 'y')
                if self.save_plots_only:
                    clearImshow(f'theta={theta_start}')
                if use_row:
                    center_offset = find_center.get('upper_center_offset')
                    if is_num(center_offset):
                        use_center = input_yesno('Current upper center offset = '+
                                f'{center_offset}, use this value (y/n)?', 'y')
        if not use_center:
            if not use_row:
                if not self.test_mode:
                    quickImshow(center_stack[:,0,:], title=f'theta={theta_start}',
                            aspect='auto')
                row = input_int('\nEnter row index to find upper center', lower_row+1, n2-1, n2-1)
                if row == '':
                    row = n2-1
                if self.save_plots_only:
                    clearImshow(f'theta={theta_start}')
            # center_stack order: row,theta,column
            center_offset = self._findCenterOnePlane(center_stack[row,:,:], row, thetas_deg,
                    eff_pixel_size, cross_sectional_dim, num_core=num_core,
                    galaxy_param=galaxy_param)
        logging.info(f'upper_center_offset = {center_offset:.2f}')
        del center_stack

        # Update config and save to file
        find_center['upper_row'] = row
        find_center['upper_center_offset'] = center_offset
        find_center['completed'] = True
        self.cf.saveFile(self.config_out)

    def checkCenters(self):
        """Check centers for the tomography stacks.
        """
        #RV TODO load all stacks and check at all stack boundaries
        return
        logging.debug('Check centers for tomography stacks')
        center_stack_index = self.config.get('center_stack_index')
        if center_stack_index is None:
            raise ValueError('Unable to read center_stack_index from config')
        center_stack_index = self.tomo_stacks[self.tomo_data_indices.index(center_stack_index)]
        lower_row = self.config.get('lower_row')
        if lower_row is None:
            raise ValueError('Unable to read lower_row from config')
        lower_center_offset = self.config.get('lower_center_offset')
        if lower_center_offset is None:
            raise ValueError('Unable to read lower_center_offset from config')
        upper_row = self.config.get('upper_row')
        if upper_row is None:
            raise ValueError('Unable to read upper_row from config')
        upper_center_offset = self.config.get('upper_center_offset')
        if upper_center_offset is None:
            raise ValueError('Unable to read upper_center_offset from config')
        center_slope = (upper_center_offset-lower_center_offset)/(upper_row-lower_row)
        shift = upper_center_offset-lower_center_offset
        if lower_row == 0:
            logging.warning(f'lower_row == 0: one row offset between both planes')
        else:
            lower_row -= 1
            lower_center_offset -= center_slope

        # stack order: stack,row,theta,column
        if center_stack_index:
            stack1 = self.tomo_stacks[center_stack_index-1]
            stack2 = self.tomo_stacks[center_stack_index]
            if not stack1.size:
                logging.error(f'Unable to load required tomography stack {stack1}')
            elif not stack2.size:
                logging.error(f'Unable to load required tomography stack {stack1}')
            else:
                assert(0 <= lower_row < stack2.shape[0])
                assert(0 <= upper_row < stack1.shape[0])
                plane1 = stack1[upper_row,:]
                plane2 = stack2[lower_row,:]
                for i in range(-2, 3):
                    shift_i = shift+2*i
                    plane1_shifted = spi.shift(plane2, [0, shift_i])
                    quickPlot((plane1[0,:],), (plane1_shifted[0,:],),
                            title=f'stacks {stack1} {stack2} shifted {2*i} theta={self.start_theta}',
                            path=self.output_folder, save_fig=self.save_plots,
                            save_only=self.save_plots_only)
        if center_stack_index < self.num_tomo_stacks-1:
            stack1 = self.tomo_stacks[center_stack_index]
            stack2 = self.tomo_stacks[center_stack_index+1]
            if not stack1.size:
                logging.error(f'Unable to load required tomography stack {stack1}')
            elif not stack2.size:
                logging.error(f'Unable to load required tomography stack {stack1}')
            else:
                assert(0 <= lower_row < stack2.shape[0])
                assert(0 <= upper_row < stack1.shape[0])
                plane1 = stack1[upper_row,:]
                plane2 = stack2[lower_row,:]
                for i in range(-2, 3):
                    shift_i = -shift+2*i
                    plane1_shifted = spi.shift(plane2, [0, shift_i])
                    quickPlot((plane1[0,:],), (plane1_shifted[0,:],), 
                            title=f'stacks {stack1} {stack2} shifted {2*i} theta={start_theta}',
                            path=self.output_folder, save_fig=self.save_plots,
                            save_only=self.save_plots_only)
        del plane1, plane2, plane1_shifted

        # Update config file
        self.config = update('config.txt', 'check_centers', True, 'find_centers')

    def reconstructTomoStacks(self, galaxy_param=None, num_core=None):
        """Reconstruct tomography stacks.
        """
        if num_core is None:
            num_core = self.num_core
        logging.info(f'num_core = {num_core}')
        if self.galaxy_flag:
            assert(galaxy_param)
            if not os.path.exists('center_slice_pngs'):
                os.mkdir('center_slice_pngs')
        logging.debug('Reconstruct tomography stacks')
        stacks = self.config['stack_info']['stacks']
        assert(len(self.tomo_stacks) == self.config['stack_info']['num'])
        assert(len(self.tomo_stacks) == len(stacks))
        assert(len(self.tomo_recon_stacks) == len(stacks))
        if self.galaxy_flag:
            assert(isinstance(galaxy_param, dict))
            # Get rotation axis centers
            center_offsets = galaxy_param['center_offsets']
            assert(isinstance(center_offsets, list) and len(center_offsets) == 2)
            lower_center_offset = center_offsets[0]
            assert(is_num(lower_center_offset))
            upper_center_offset = center_offsets[1]
            assert(is_num(upper_center_offset))
        else:
            if galaxy_param:
                logging.warning('Ignoring galaxy_param in reconstructTomoStacks (only for Galaxy)')
                galaxy_param = None
            lower_center_offset = None
            upper_center_offset = None

        # Get rotation axis rows and centers
        find_center = self.config['find_center']
        lower_row = find_center.get('lower_row')
        if lower_row is None:
            logging.error('Unable to read lower_row from config')
            return
        upper_row = find_center.get('upper_row')
        if upper_row is None:
            logging.error('Unable to read upper_row from config')
            return
        logging.debug(f'lower_row = {lower_row} upper_row = {upper_row}')
        assert(lower_row < upper_row)
        if lower_center_offset is None:
            lower_center_offset = find_center.get('lower_center_offset')
            if lower_center_offset is None:
                logging.error('Unable to read lower_center_offset from config')
                return
        if upper_center_offset is None:
            upper_center_offset = find_center.get('upper_center_offset')
            if upper_center_offset is None:
                logging.error('Unable to read upper_center_offset from config')
                return
        center_slope = (upper_center_offset-lower_center_offset)/(upper_row-lower_row)

        # Set thetas (in radians)
        theta_range = self.config['theta_range']
        theta_start = theta_range['start']
        theta_end = theta_range['end']
        num_theta = theta_range['num']
        num_theta_skip = self.config['preprocess'].get('num_theta_skip', 0)
        thetas = np.radians(np.linspace(theta_start, theta_end,
                int(num_theta/(num_theta_skip+1)), endpoint=False))

        # Reconstruct tomo stacks
        zoom_perc = self.config['preprocess'].get('zoom_perc', 100)
        if zoom_perc == 100:
            basetitle = 'recon stack fullres'
        else:
            basetitle = f'recon stack {zoom_perc}p'
        load_error = False
        for i,stack in enumerate(stacks):
            # Check if stack can be loaded
            # reconstructed stack order for each one in stack : row/z,x,y
            # preprocessed stack order for each one in stack: row,theta,column
            index = stack['index']
            if not self.galaxy_flag:
                available = False
                if stack.get('reconstructed', False):
                    self.tomo_recon_stacks[i], available = self._loadTomo('recon stack', index)
                if self.tomo_recon_stacks[i].size or available:
                    if self.tomo_stacks[i].size:
                        self.tomo_stacks[i] = np.array([])
                    assert(stack.get('preprocessed', False) == True)
                    assert(stack.get('reconstructed', False) == True)
                    continue
            stack['reconstructed'] = False
            if not self.tomo_stacks[i].size:
                self.tomo_stacks[i], available = self._loadTomo('red stack', index,
                        required=True)
            if not self.tomo_stacks[i].size:
                logging.error(f'Unable to load tomography stack {index} for reconstruction')
                stack[i]['preprocessed'] = False
                load_error = True
                continue
            assert(0 <= lower_row < upper_row < self.tomo_stacks[i].shape[0])
            center_offsets = [lower_center_offset-lower_row*center_slope,
                    upper_center_offset+(self.tomo_stacks[i].shape[0]-1-upper_row)*center_slope]
            t0 = time()
            logging.debug(f'running _reconstructOneTomoStack on {num_core} cores ...')
            self.tomo_recon_stacks[i]= self._reconstructOneTomoStack(self.tomo_stacks[i],
                    thetas, center_offsets=center_offsets, num_core=num_core,
                    algorithm='gridrec')
            logging.debug(f'... _reconstructOneTomoStack took {time()-t0:.2f} seconds!')
            logging.info(f'Reconstruction of stack {index} took {time()-t0:.2f} seconds!')
            if self.galaxy_flag:
                x_slice = int(self.tomo_stacks[i].shape[0]/2) 
                title = f'{basetitle} {index} xslice{x_slice}'
                quickImshow(self.tomo_recon_stacks[i][x_slice,:,:], title=title,
                        path='center_slice_pngs', save_fig=True, save_only=True)
                y_slice = int(self.tomo_stacks[i].shape[1]/2) 
                title = f'{basetitle} {index} yslice{y_slice}'
                quickImshow(self.tomo_recon_stacks[i][:,y_slice,:], title=title,
                        path='center_slice_pngs', save_fig=True, save_only=True)
                z_slice = int(self.tomo_stacks[i].shape[2]/2) 
                title = f'{basetitle} {index} zslice{z_slice}'
                quickImshow(self.tomo_recon_stacks[i][:,:,z_slice], title=title,
                        path='center_slice_pngs', save_fig=True, save_only=True)
            else:#elif not self.test_mode:
                x_slice = int(self.tomo_stacks[i].shape[0]/2) 
                title = f'{basetitle} {index} xslice{x_slice}'
                quickImshow(self.tomo_recon_stacks[i][x_slice,:,:], title=title,
                        path=self.output_folder, save_fig=self.save_plots,
                        save_only=self.save_plots_only)
                y_slice = int(self.tomo_stacks[i].shape[1]/2) 
                title = f'{basetitle} {index} yslice{y_slice}'
                quickImshow(self.tomo_recon_stacks[i][:,y_slice,:], title=title,
                        path=self.output_folder, save_fig=self.save_plots,
                        save_only=self.save_plots_only)
                z_slice = int(self.tomo_stacks[i].shape[2]/2) 
                title = f'{basetitle} {index} zslice{z_slice}'
                quickImshow(self.tomo_recon_stacks[i][:,:,z_slice], title=title,
                        path=self.output_folder, save_fig=self.save_plots,
                        save_only=self.save_plots_only)
#                quickPlot(self.tomo_recon_stacks[i]
#                        [x_slice,int(self.tomo_recon_stacks[i].shape[1]/2),:],
#                        title=f'{title} cut{int(self.tomo_recon_stacks[i].shape[1]/2)}',
#                        path=self.output_folder, save_fig=self.save_plots,
#                        save_only=self.save_plots_only)
                if not self.test_mode:
                    self._saveTomo('recon stack', self.tomo_recon_stacks[i], index)
            self.tomo_stacks[i] = np.array([])

            # Update config and save to file
            stack['reconstructed'] = True
            combine_stacks = self.config.get('combine_stacks')
            if combine_stacks and index in combine_stacks.get('stacks', []):
                combine_stacks['stacks'].remove(index)
            self.cf.saveFile(self.config_out)

        if self.galaxy_flag:
            # Save reconstructed tomography stack to file
            t0 = time()
            output_name = galaxy_param['output_name']
            logging.info(f'Saving reconstructed tomography stack to {output_name} ...')
            save_stacks = {f'set_{stack["index"]}':tomo_stack
                    for stack,tomo_stack in zip(stacks,self.tomo_recon_stacks)}
            np.savez(output_name, **save_stacks)
            logging.info(f'... done in {time()-t0:.2f} seconds!')

            # Create cross section profile in yz-plane
            tomosum = 0
            [tomosum := tomosum+np.sum(tomo_recon_stack, axis=(0,2)) for tomo_recon_stack in
                self.tomo_recon_stacks]
            quickPlot(tomosum, title='recon stack sum yz', path='center_slice_pngs',
                save_fig=True, save_only=True)

            # Create cross section profile in xz-plane
            tomosum = 0
            [tomosum := tomosum+np.sum(tomo_recon_stack, axis=(0,1)) for tomo_recon_stack in
                self.tomo_recon_stacks]
            quickPlot(tomosum, title='recon stack sum xz', path='center_slice_pngs',
                save_fig=True, save_only=True)

            # Create cross section profile in xy-plane
            num_tomo_stacks = len(stacks)
            row_bounds = self.config['find_center']['row_bounds']
            if not is_index_range(row_bounds, 0, self.tomo_recon_stacks[0].shape[0]):
                illegal_value(row_bounds, 'find_center:row_bounds', 'config file')
                return
            if num_tomo_stacks == 1:
                low_bound = row_bounds[0]
            else:
                low_bound = 0
            tomosum = np.sum(self.tomo_recon_stacks[0][low_bound:row_bounds[1],:,:], axis=(1,2))
            if num_tomo_stacks > 2:
                tomosum = np.concatenate([tomosum]+
                        [np.sum(self.tomo_recon_stacks[i][row_bounds[0]:row_bounds[1],:,:],
                        axis=(1,2)) for i in range(1, num_tomo_stacks-1)])
            if num_tomo_stacks > 1:
                tomosum = np.concatenate([tomosum,
                    np.sum(self.tomo_recon_stacks[num_tomo_stacks-1][row_bounds[0]:,:,:],
                    axis=(1,2))])
            quickPlot(tomosum, title='recon stack sum xy', path='center_slice_pngs',
                save_fig=True, save_only=True)

    def combineTomoStacks(self, galaxy_param=None):
        """Combine the reconstructed tomography stacks.
        """
        # stack order: stack,row(z),x,y
        if self.galaxy_flag:
            assert(galaxy_param)
            if not os.path.exists('combine_pngs'):
                os.mkdir('combine_pngs')
        logging.debug('Combine reconstructed tomography stacks')
        stack_info = self.config['stack_info']
        stacks = stack_info['stacks']
        assert(len(self.tomo_recon_stacks) == stack_info['num'])
        assert(len(self.tomo_recon_stacks) == len(stacks))
        if self.galaxy_flag:
            assert(isinstance(galaxy_param, dict))
            # Get image bounds
            x_bounds = galaxy_param['x_bounds']
            assert(isinstance(x_bounds, list) and len(x_bounds) == 2)
            y_bounds = galaxy_param['y_bounds']
            assert(isinstance(y_bounds, list) and len(y_bounds) == 2)
            z_bounds = galaxy_param['z_bounds']
            assert(isinstance(z_bounds, list) and len(z_bounds) == 2)
        else:
            if galaxy_param:
                logging.warning('Ignoring galaxy_param in reconstructTomoStacks (only for Galaxy)')
                galaxy_param = None

        # Load any unloaded reconstructed stacks
        for i,stack in enumerate(stacks):
            available = False
            if not self.tomo_recon_stacks[i].size and stack.get('reconstructed', False):
                self.tomo_recon_stacks[i], available = self._loadTomo('recon stack',
                        stack['index'], required=True)
            if not self.tomo_recon_stacks[i].size:
                logging.error(f'Unable to load reconstructed stack {stack["index"]}')
                stack['reconstructed'] = False
                return
            if i:
                if (self.tomo_recon_stacks[i].shape[1] != self.tomo_recon_stacks[0].shape[1] or
                        self.tomo_recon_stacks[i].shape[1] != self.tomo_recon_stacks[0].shape[1]):
                    logging.error('Incompatible reconstructed tomography stack dimensions')
                    return

        # Get center stack boundaries
        row_bounds = self.config['find_center']['row_bounds']
        if not is_index_range(row_bounds, 0, self.tomo_recon_stacks[0].shape[0]):
            illegal_value(row_bounds, 'find_center:row_bounds', 'config file')
            return

        # Selecting x bounds (in yz-plane)
        tomosum = 0
        #RV FIX :=
        [tomosum := tomosum+np.sum(tomo_recon_stack, axis=(0,2)) for tomo_recon_stack in
                self.tomo_recon_stacks]
        combine_stacks = self.config.get('combine_stacks')
        if self.galaxy_flag:
            if x_bounds[0] == -1:
                x_bounds[0] = 0
            if x_bounds[1] == -1:
                x_bounds[1] = tomosum.size
            if not is_index_range(x_bounds, 0, tomosum.size):
                illegal_value(x_bounds, 'x_bounds', 'galaxy input')
            tomosum_min = tomosum.min()
            tomosum_max = tomosum.max()
            quickPlot((range(tomosum.size), tomosum),
                    ([x_bounds[0], x_bounds[0]], [tomosum_min, tomosum_max], 'r-'),
                    ([x_bounds[1]-1, x_bounds[1]-1], [tomosum_min, tomosum_max], 'r-'),
                    title=f'recon stack sum yz', path='combine_pngs', save_fig=True, save_only=True)
        else:
            if combine_stacks and 'x_bounds' in combine_stacks:
                x_bounds = combine_stacks['x_bounds']
                if not is_index_range(x_bounds, 0, tomosum.size):
                    illegal_value(x_bounds, 'combine_stacks:x_bounds', 'config file')
                elif not self.test_mode:
                    accept = False
                    while not accept:
                        mask, x_bounds = draw_mask_1d(tomosum, current_index_ranges=[x_bounds],
                                title='select x data range', legend='recon stack sum yz')
                        while len(x_bounds) != 1:
                            print('Please select exactly one continuous range')
                            mask, x_bounds = draw_mask_1d(tomosum, title='select x data range',
                                    legend='recon stack sum yz')
                        x_bounds = list(x_bounds[0])
                        quickPlot(tomosum, vlines=x_bounds, title='recon stack sum yz')
                        print(f'x_bounds = {x_bounds} (lower bound inclusive, upper bound '+
                                'exclusive)')
                        accept = input_yesno('Accept these bounds (y/n)?', 'y')
                        if not accept:
                            x_bounds = None
            else:
                if not input_yesno('\nDo you want to change the image x-bounds (y/n)?', 'y'):
                    x_bounds = [0, tomosum.size]
                else:
                    accept = False
                    while not accept:
                        mask, x_bounds = draw_mask_1d(tomosum, title='select x data range',
                                legend='recon stack sum yz')
                        while len(x_bounds) != 1:
                            print('Please select exactly one continuous range')
                            mask, x_bounds = draw_mask_1d(tomosum, title='select x data range',
                                    legend='recon stack sum yz')
                        x_bounds = list(x_bounds[0])
                        quickPlot(tomosum, vlines=x_bounds, title='recon stack sum yz')
                        print(f'x_bounds = {x_bounds} (lower bound inclusive, upper bound '+
                                'exclusive)')
                        accept = input_yesno('Accept these bounds (y/n)?', 'y')
            if False and self.test_mode:
                np.savetxt(f'{self.output_folder}/recon_stack_sum_yz.txt',
                        tomosum[x_bounds[0]:x_bounds[1]], fmt='%.6e')
            if self.save_plots_only:
                clearPlot('recon stack sum yz')

        # Selecting y bounds (in xz-plane)
        tomosum = 0
        #RV FIX :=
        [tomosum := tomosum+np.sum(tomo_recon_stack, axis=(0,1)) for tomo_recon_stack in
                self.tomo_recon_stacks]
        if self.galaxy_flag:
            if y_bounds[0] == -1:
                y_bounds[0] = 0
            if y_bounds[1] == -1:
                y_bounds[1] = tomosum.size
            if not is_index_range(y_bounds, 0, tomosum.size):
                illegal_value(y_bounds, 'y_bounds', 'galaxy input')
            tomosum_min = tomosum.min()
            tomosum_max = tomosum.max()
            quickPlot((range(tomosum.size), tomosum),
                    ([y_bounds[0], y_bounds[0]], [tomosum_min, tomosum_max], 'r-'),
                    ([y_bounds[1]-1, y_bounds[1]-1], [tomosum_min, tomosum_max], 'r-'),
                    title=f'recon stack sum xz', path='combine_pngs', save_fig=True, save_only=True)
        else:
            if combine_stacks and 'y_bounds' in combine_stacks:
                y_bounds = combine_stacks['y_bounds']
                if not is_index_range(y_bounds, 0, tomosum.size):
                    illegal_value(y_bounds, 'combine_stacks:y_bounds', 'config file')
                elif not self.test_mode:
                    accept = False
                    while not accept:
                        mask, y_bounds = draw_mask_1d(tomosum, current_index_ranges=[y_bounds],
                                title='select y data range', legend='recon stack sum xz')
                        while len(y_bounds) != 1:
                            print('Please select exactly one continuous range')
                            mask, y_bounds = draw_mask_1d(tomosum, title='select y data range',
                                    legend='recon stack sum xz')
                        y_bounds = list(y_bounds[0])
                        quickPlot(tomosum, vlines=y_bounds, title='recon stack sum xz')
                        print(f'y_bounds = {y_bounds} (lower bound inclusive, upper bound '+
                                'exclusive)')
                        accept = input_yesno('Accept these bounds (y/n)?', 'y')
                        if not accept:
                            y_bounds = None
            else:
                if not input_yesno('\nDo you want to change the image y-bounds (y/n)?', 'y'):
                    y_bounds = [0, tomosum.size]
                else:
                    accept = False
                    while not accept:
                        mask, y_bounds = draw_mask_1d(tomosum, title='select y data range',
                                legend='recon stack sum xz')
                        while len(y_bounds) != 1:
                            print('Please select exactly one continuous range')
                            mask, y_bounds = draw_mask_1d(tomosum, title='select y data range',
                                    legend='recon stack sum xz')
                        y_bounds = list(y_bounds[0])
                        quickPlot(tomosum, vlines=y_bounds, title='recon stack sum xz')
                        print(f'y_bounds = {y_bounds} (lower bound inclusive, upper bound '+
                                'exclusive)')
                        accept = input_yesno('Accept these bounds (y/n)?', 'y')
            if False and self.test_mode:
                np.savetxt(f'{self.output_folder}/recon_stack_sum_xz.txt',
                        tomosum[y_bounds[0]:y_bounds[1]], fmt='%.6e')
            if self.save_plots_only:
                clearPlot('recon stack sum xz')

        # Combine reconstructed tomography stacks
        logging.info(f'Combining reconstructed stacks ...')
        t0 = time()
        num_tomo_stacks = len(stacks)
        if num_tomo_stacks == 1:
            low_bound = row_bounds[0]
        else:
            low_bound = 0
        tomo_recon_combined = self.tomo_recon_stacks[0][low_bound:row_bounds[1],
                x_bounds[0]:x_bounds[1],y_bounds[0]:y_bounds[1]]
        if num_tomo_stacks > 2:
            tomo_recon_combined = np.concatenate([tomo_recon_combined]+
                    [self.tomo_recon_stacks[i][row_bounds[0]:row_bounds[1],
                    x_bounds[0]:x_bounds[1],y_bounds[0]:y_bounds[1]]
                    for i in range(1, num_tomo_stacks-1)])
        if num_tomo_stacks > 1:
            tomo_recon_combined = np.concatenate([tomo_recon_combined]+
                    [self.tomo_recon_stacks[num_tomo_stacks-1][row_bounds[0]:,
                    x_bounds[0]:x_bounds[1],y_bounds[0]:y_bounds[1]]])
        logging.info(f'... done in {time()-t0:.2f} seconds!')
        combined_stacks = [stack['index'] for stack in stacks]

        # Wrap up if in test_mode
        tomosum = np.sum(tomo_recon_combined, axis=(1,2))
        if self.test_mode:
            zoom_perc = self.config['preprocess'].get('zoom_perc', 100)
            filename = 'recon combined sum xy'
            if zoom_perc is None or zoom_perc == 100:
                filename += ' fullres.dat'
            else:
                filename += f' {zoom_perc}p.dat'
            quickPlot(tomosum, title='recon combined sum xy',
                    path=self.output_folder, save_fig=self.save_plots,
                    save_only=self.save_plots_only)
            if False:
                np.savetxt(f'{self.output_folder}/recon_combined_sum_xy.txt',
                        tomosum, fmt='%.6e')
            np.savetxt(f'{self.output_folder}/recon_combined.txt',
                    tomo_recon_combined[int(tomosum.size/2),:,:], fmt='%.6e')

            # Update config and save to file
            if combine_stacks:
                combine_stacks['x_bounds'] = x_bounds
                combine_stacks['y_bounds'] = y_bounds
                combine_stacks['stacks'] = combined_stacks
            else:
                self.config['combine_stacks'] = {'x_bounds' : x_bounds, 'y_bounds' : y_bounds,
                        'stacks' : combined_stacks}
            self.cf.saveFile(self.config_out)
            return

        # Selecting z bounds (in xy-plane)
        if self.galaxy_flag:
            if z_bounds[0] == -1:
                z_bounds[0] = 0
            if z_bounds[1] == -1:
                z_bounds[1] = tomosum.size
            if not is_index_range(z_bounds, 0, tomosum.size):
                illegal_value(z_bounds, 'z_bounds', 'galaxy input')
            tomosum_min = tomosum.min()
            tomosum_max = tomosum.max()
            quickPlot((range(tomosum.size), tomosum),
                    ([z_bounds[0], z_bounds[0]], [tomosum_min, tomosum_max], 'r-'),
                    ([z_bounds[1]-1, z_bounds[1]-1], [tomosum_min, tomosum_max], 'r-'),
                    title=f'recon stack sum xy', path='combine_pngs', save_fig=True, save_only=True)
        else:
            if not input_yesno('\nDo you want to change the image z-bounds (y/n)?', 'n'):
                z_bounds = [0, tomosum.size]
            else:
                accept = False
                while not accept:
                    mask, z_bounds = draw_mask_1d(tomosum, title='select z data range',
                            legend='recon stack sum xy')
                    while len(z_bounds) != 1:
                        print('Please select exactly one continuous range')
                        mask, z_bounds = draw_mask_1d(tomosum, title='select z data range',
                                legend='recon stack sum xy')
                    z_bounds = list(z_bounds[0])
                    quickPlot(tomosum, vlines=z_bounds, title='recon stack sum xy')
                    print(f'z_bounds = {z_bounds} (lower bound inclusive, upper bound exclusive)')
                    accept = input_yesno('Accept these bounds (y/n)?', 'y')
            if self.save_plots_only:
                clearPlot('recon combined sum xy')
        if z_bounds[0] != 0 or z_bounds[1] != tomosum.size:
            tomo_recon_combined = tomo_recon_combined[z_bounds[0]:z_bounds[1],:,:]

        # Plot center slices
        if self.galaxy_flag:
            path = 'combine_pngs'
            save_fig = True
            save_only = True
        else:
            path = self.output_folder
            save_fig = self.save_plots
            save_only = self.save_plots_only
        quickImshow(tomo_recon_combined[int(tomo_recon_combined.shape[0]/2),:,:],
                title=f'recon combined xslice{int(tomo_recon_combined.shape[0]/2)}',
                path=path, save_fig=save_fig, save_only=save_only)
        quickImshow(tomo_recon_combined[:,int(tomo_recon_combined.shape[1]/2),:],
                title=f'recon combined yslice{int(tomo_recon_combined.shape[1]/2)}',
                path=path, save_fig=save_fig, save_only=save_only)
        quickImshow(tomo_recon_combined[:,:,int(tomo_recon_combined.shape[2]/2)],
                title=f'recon combined zslice{int(tomo_recon_combined.shape[2]/2)}',
                path=path, save_fig=save_fig, save_only=save_only)

        # Save combined reconstructed tomography stack to file
        if self.galaxy_flag:
            t0 = time()
            output_name = galaxy_param['output_name']
            logging.info(f'Saving combined reconstructed tomography stack to {output_name} ...')
            np.save(output_name, tomo_recon_combined)
            logging.info(f'... done in {time()-t0:.2f} seconds!')
        else:
            base_name = 'recon combined'
            for stack in stacks:
                base_name += f' {stack["index"]}'
            self._saveTomo(base_name, tomo_recon_combined)

        # Update config and save to file
        if combine_stacks:
            combine_stacks['x_bounds'] = x_bounds
            combine_stacks['y_bounds'] = y_bounds
            combine_stacks['z_bounds'] = z_bounds
            combine_stacks['stacks'] = combined_stacks
        else:
            self.config['combine_stacks'] = {'x_bounds' : x_bounds, 'y_bounds' : y_bounds,
                    'z_bounds' : z_bounds, 'stacks' : combined_stacks}
        self.cf.saveFile(self.config_out)

def runTomo(config_file=None, config_dict=None, output_folder='.', log_level='INFO',
        test_mode=False, num_core=-1):
    """Run a tomography analysis.
    """
    # Instantiate Tomo object
    tomo = Tomo(config_file=config_file, output_folder=output_folder, log_level=log_level,
            test_mode=test_mode, num_core=num_core)
    if not tomo.is_valid:
        raise ValueError('Invalid config and/or detector file provided.')

    # Preprocess the image files
    assert(tomo.config['stack_info'])
    num_tomo_stacks = tomo.config['stack_info']['num']
    assert(num_tomo_stacks == len(tomo.tomo_stacks))
    preprocessed_stacks = []
    if not tomo.test_mode:
        preprocess = tomo.config.get('preprocess', None)
        if preprocess:
            preprocessed_stacks = [stack['index'] for stack in tomo.config['stack_info']['stacks']
                    if stack.get('preprocessed', False)]
    if len(preprocessed_stacks) != num_tomo_stacks:
        tomo.genTomoStacks()
        if not tomo.is_valid:
            IOError('Unable to load all required image files.')
        tomo.cf.saveFile(tomo.config_out)

    # Find centers
    find_center = tomo.config.get('find_center')
    if find_center is None or not find_center.get('completed', False):
        tomo.findCenters()

    # Check centers
    #if num_tomo_stacks > 1 and not tomo.config.get('check_centers', False):
    #    tomo.checkCenters()

    # Reconstruct tomography stacks
    assert(tomo.config['stack_info']['stacks'])
    reconstructed_stacks = [stack['index'] for stack in tomo.config['stack_info']['stacks']
            if stack.get('reconstructed', False)]
    if len(reconstructed_stacks) != num_tomo_stacks:
        tomo.reconstructTomoStacks()

    # Combine reconstructed tomography stacks
    reconstructed_stacks = [stack['index'] for stack in tomo.config['stack_info']['stacks']
            if stack.get('reconstructed', False)]
    combine_stacks = tomo.config.get('combine_stacks')
    if len(reconstructed_stacks) and (combine_stacks is None or
            combine_stacks.get('stacks') != reconstructed_stacks):
        tomo.combineTomoStacks()

#%%============================================================================
if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Tomography reconstruction')
    parser.add_argument('-c', '--config',
            default=None,
            help='Input config')
    parser.add_argument('-o', '--output_folder',
            default='.',
            help='Output folder')
    parser.add_argument('-l', '--log_level',
            default='INFO',
            help='Log level')
    parser.add_argument('-t', '--test_mode',
            action='store_true',
            default=False,
            help='Test mode flag')
    parser.add_argument('--num_core',
            type=int,
            default=-1,
            help='Number of cores')
    args = parser.parse_args()

    if args.config is None:
        if os.path.isfile('config.yaml'):
            args.config = 'config.yaml'
        else:
            args.config = 'config.txt'

    # Set basic log configuration
    logging_format = '%(asctime)s : %(levelname)s - %(module)s : %(funcName)s - %(message)s'
    if not args.test_mode:
        level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(level, int):
            raise ValueError(f'Invalid log_level: {args.log_level}')
        logging.basicConfig(format=logging_format, level=level, force=True,
                handlers=[logging.StreamHandler()])

    logging.debug(f'config = {args.config}')
    logging.debug(f'output_folder = {args.output_folder}')
    logging.debug(f'log_level = {args.log_level}')
    logging.debug(f'test_mode = {args.test_mode}')
    logging.debug(f'num_core = {args.num_core}')

    # Run tomography analysis
    runTomo(config_file=args.config, output_folder=args.output_folder, log_level=args.log_level,
            test_mode=args.test_mode, num_core=args.num_core)

#%%============================================================================
#    input('Press any key to continue')
#%%============================================================================

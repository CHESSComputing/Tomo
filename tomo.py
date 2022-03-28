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
try:
    import pyinputplus as pyip
except:
    pass
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

class ConfigTomo(msnc.Config):
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
        if not msnc.is_int(self.num_tomo_stacks, 1):
            self.num_tomo_stacks = None
            msnc.illegal_value('num_tomo_stacks', self.num_tomo_stacks, 'config file')
            return False
        logging.info(f'num_tomo_stacks = {self.num_tomo_stacks}')

        # Find tomography images file/folders and stack parameters
        tomo_data_paths_indices = sorted({key:value for key,value in self.config.items()
            if 'tomo_data_path' in key}.items())
        if len(tomo_data_paths_indices) != self.num_tomo_stacks:
            logging.error(f'Incorrect number of tomography data path names in config file')
            is_valid = False
        self.tomo_data_paths = [tomo_data_paths_indices[i][1] for i in range(self.num_tomo_stacks)]
        self.tomo_data_indices = [msnc.get_trailing_int(tomo_data_paths_indices[i][0])
                if msnc.get_trailing_int(tomo_data_paths_indices[i][0]) else None
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
        if not msnc.is_num(self.start_theta, 0.):
            msnc.illegal_value('start_theta', self.start_theta, 'config file')
            is_valid = False
        logging.debug(f'start_theta = {self.start_theta}')
        self.end_theta = self.config.get('end_theta', 180.)
        if not msnc.is_num(self.end_theta, self.start_theta):
            msnc.illegal_value('end_theta', self.end_theta, 'config file')
            is_valid = False
        logging.debug(f'end_theta = {self.end_theta}')
        self.num_thetas = self.config.get('num_thetas')
        if not (self.num_thetas is None or msnc.is_int(self.num_thetas, 1)):
            msnc.illegal_value('num_thetas', self.num_thetas, 'config file')
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
        if not msnc.is_int(self.num_tomo_stacks, 1):
            self.num_tomo_stacks = None
            msnc.illegal_value('stack_info:num', self.num_tomo_stacks, 'config file')
            return False
        logging.info(f'num_tomo_stacks = {self.num_tomo_stacks}')

        # Find tomography images file/folders and stack parameters
        stacks = stack_info.get('stacks')
        if stacks is None or len(stacks) is not self.num_tomo_stacks:
            msnc.illegal_value('stack_info:stacks', stacks, 'config file')
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
            if not msnc.is_num(self.start_theta, 0.):
                msnc.illegal_value('theta_range:start', self.start_theta, 'config file')
                is_valid = False
            logging.debug(f'start_theta = {self.start_theta}')
            self.end_theta = theta_range.get('end', 180.)
            if not msnc.is_num(self.end_theta, self.start_theta):
                msnc.illegal_value('theta_range:end', self.end_theta, 'config file')
                is_valid = False
            logging.debug(f'end_theta = {self.end_theta}')
            self.num_thetas = theta_range.get('num')
            if self.num_thetas and not msnc.is_int(self.num_thetas, 1):
                msnc.illegal_value('theta_range:num', self.num_thetas, 'config file')
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
            msnc.illegal_value('work_folder', work_folder, 'config file')
            is_valid = False
        logging.info(f'work_folder: {work_folder}')

        # Check data filetype (shared by both file formats)
        self.data_filetype = self.config.get('data_filetype', 'tif')
        if not isinstance(self.data_filetype, str) or (self.data_filetype != 'tif' and
                self.data_filetype != 'h5'):
            msnc.illegal_value('data_filetype', self.data_filetype, 'config file')

        if self.suffix == '.yml' or self.suffix == '.yaml':
            is_valid = self._validate_yaml()
        elif self.suffix == '.txt':
            is_valid = self._validate_txt()
        else:
            logging.error(f'Undefined or illegal config file extension: {self.suffix}')

        # Find tomography bright field images file/folder
        if self.tdf_data_path:
            if self.data_filetype == 'h5':
                if isinstance(self.tdf_data_path, str):
                    if not os.path.isabs(self.tdf_data_path):
                        self.tdf_data_path = os.path.abspath(
                                f'{work_folder}/{self.tdf_data_path}')
                else:
                    msnc.illegal_value('tdf_data_path', tdf_data_fil, 'config file')
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
                    msnc.illegal_value('tdf_data_path', self.tdf_data_path, 'config file')
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
                    msnc.illegal_value('tbf_data_path', tbf_data_fil, 'config file')
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
                    msnc.illegal_value('tbf_data_path', self.tbf_data_path, 'config file')
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
                    msnc.illegal_value(f'stack_info:stacks:data_path', data_path, 'config file')
                    is_valid = False
                    data_path = None
            else:
                if isinstance(data_path, int):
                    data_path = os.path.abspath(f'{work_folder}/{data_path}/nf')
                elif isinstance(data_path, str):
                    if not os.path.isabs(data_path):
                        data_path = os.path.abspath(f'{work_folder}/{data_path}')
                else:
                    msnc.illegal_value(f'stack_info:stacks:data_path', data_path, 'config file')
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
                msnc.illegal_value(f'stack_info:stacks:index', index, 'config file')
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
            elif not msnc.is_num(ref_height):
                msnc.illegal_value(f'stack_info:stacks:ref_height', ref_height, 'config file')
                is_valid = False
                ref_height = None
            # Set reference heights relative to first stack
            if (len(tomo_ref_heights) and msnc.is_num(ref_height) and
                    msnc.is_num(tomo_ref_heights[0])):
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
            log_level='INFO', log_stream='tomo.log', galaxy_flag=False, test_mode=False):
        """Initialize with optional config input file or dictionary
        """
        self.ncore = mp.cpu_count()
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
        if config_out is not None:
            if isinstance(config_out, str):
                if isinstance(log_stream, str):
                    path = os.path.split(log_stream)[0]
                    if path and not os.path.isdir(path):
                        raise OSError(f'Invalid log_stream path')
            else:
                raise OSError(f'Invalid config_out input {config_out} {type(config_out)}')
        if not os.path.isdir(output_folder):
            raise OSError(f'Invalid output_folder input {output_folder} {type(output_folder)}')
        if isinstance(log_stream, str):
            path = os.path.split(log_stream)[0]
            if path and not os.path.isdir(path):
                raise OSError(f'Invalid log_stream path')
            if not os.path.isabs(path):
                log_stream = f'{output_folder}/{log_stream}'
        if not isinstance(galaxy_flag, bool):
            raise ValueError(f'Invalid galaxy_flag input {galaxy_flag} {type(galaxy_flag)}')
        if not isinstance(test_mode, bool):
            raise ValueError(f'Invalid test_mode input {test_mode} {type(test_mode)}')

        # Set log configuration
        logging_format = '%(asctime)s : %(levelname)s - %(module)s : %(funcName)s - %(message)s'
        if self.test_mode:
            self.save_plots_only = True
            if isinstance(log_stream, str):
                logging.basicConfig(filename=f'{log_stream}', filemode='w',
                        format=logging_format, level=logging.WARNING, force=True)
            elif isinstance(log_stream, io.TextIOWrapper):
                logging.basicConfig(filemode='w', format=logging_format, level=logging.WARNING,
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

        logging.info(f'ncore = {self.ncore}')
        logging.debug(f'config_file = {config_file}')
        logging.debug(f'config_dict = {config_dict}')
        logging.debug(f'config_out = {self.config_out}')
        logging.debug(f'output_folder = {self.output_folder}')
        logging.debug(f'log_stream = {log_stream}')
        logging.debug(f'log_level = {log_level}')
        logging.debug(f'galaxy_flag = {self.galaxy_flag}')
        logging.debug(f'test_mode = {self.test_mode}')

        # Create config object and load config file 
        self.cf = ConfigTomo(config_file, config_dict)
        if not self.cf.load_flag:
            self.is_valid = False
            return

        if self.galaxy_flag:
            self.ncore = 1 #RV can I set this? mp.cpu_count()
            assert(self.output_folder == '.')
            assert(self.test_mode is False)
            self.save_plots = True
            self.save_plots_only = True
        else:
            # Input validation is already performed during link_data_to_galaxy

            # Check config file parameters
            self.is_valid =  self.cf.validate()

            # Load detector info file
            df = msnc.Detector(self.cf.config['detector']['id'])

            # Check detector info file parameters
            if df.validate():
                pixel_size = df.getPixelSize()
                num_rows, num_columns = df.getDimensions()
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
            num_thetas = pyip.inputInt('\nEnter the number of thetas (>0): ', greaterThan=0)
        elif not msnc.is_int(num_thetas, 0):
            msnc.illegal_value('num_thetas', num_thetas, 'config file')
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
            img_start, img_offset, num_imgs = msnc.selectImageRange(img_start, img_offset,
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
            img_start, img_offset, num_imgs = msnc.selectImageRange(img_start, img_offset,
                num_imgs, 'bright field')
        if img_start < 0 or num_imgs < 1:
            logging.error('Unable to find suitable bright field images')
            if bright_field['data_path']:
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
                img_start, img_offset, num_imgs = msnc.selectImageRange(img_start, img_offset,
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
            available_stacks[i] = True

        # Safe updated config to file
        if self.is_valid:
            self.cf.saveFile(self.config_out)

        return

    def _genDark(self, tdf_files, dark_field_pngname):
        """Generate dark field.
        """
        # Load the dark field images
        logging.debug('Loading dark field...')
        dark_field = self.config['dark_field']
        tdf_stack = msnc.loadImageStack(tdf_files, self.config['data_filetype'],
                dark_field['img_offset'], dark_field['num'])

        # Take median
        self.tdf = np.median(tdf_stack, axis=0)
        del tdf_stack

        # RV make input of some kind (not always needed)
        tdf_cutoff = 21
        self.tdf[self.tdf > tdf_cutoff] = np.nan
        tdf_mean = np.nanmean(self.tdf)
        logging.debug(f'tdf_cutoff = {tdf_cutoff}')
        logging.debug(f'tdf_mean = {tdf_mean}')
        np.nan_to_num(self.tdf, copy=False, nan=tdf_mean, posinf=tdf_mean, neginf=0.)
        if not self.test_mode and not self.galaxy_flag:
            msnc.quickImshow(self.tdf, title='dark field', path=self.output_folder,
                    save_fig=self.save_plots, save_only=self.save_plots_only)
        elif self.galaxy_flag:
            msnc.quickImshow(self.tdf, title='dark field', name=dark_field_pngname,
                    save_fig=True, save_only=True)

    def _genBright(self, tbf_files, bright_field_pngname):
        """Generate bright field.
        """
        # Load the bright field images
        logging.debug('Loading bright field...')
        bright_field = self.config['bright_field']
        tbf_stack = msnc.loadImageStack(tbf_files, self.config['data_filetype'],
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
        if not self.test_mode and not self.galaxy_flag:
            msnc.quickImshow(self.tbf, title='bright field', path=self.output_folder,
                    save_fig=self.save_plots, save_only=self.save_plots_only)
        elif self.galaxy_flag:
            msnc.quickImshow(self.tbf, title='bright field', name=bright_field_pngname,
                    save_fig=True, save_only=True)

    def _setDetectorBounds(self, tomo_stack_files, tomo_field_pngname, detectorbounds_pngname):
        """Set vertical detector bounds for image stack.
        """
        preprocess = self.config.get('preprocess')
        if preprocess is None:
            img_x_bounds = [None, None]
        else:
            img_x_bounds = preprocess.get('img_x_bounds', [0, self.tbf.shape[0]])
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
            num_x_min = int(delta_z/pixel_size)+1
            logging.debug(f'num_x_min = {num_x_min}')
            if num_x_min > self.tbf.shape[0]:
                logging.warning('Image bounds and pixel size prevent seamless stacking')
                num_x_min = self.tbf.shape[0]

        # Select image bounds
        if self.galaxy_flag:
            if num_x_min is None or num_x_min >= self.tbf.shape[0]:
                img_x_bounds = [0, self.tbf.shape[0]]
            else:
                margin = int(num_x_min/10)
                offset = min(0, int((self.tbf.shape[0]-num_x_min)/2-margin))
                img_x_bounds = [offset, num_x_min+offset+2*margin]
            tomo_stack = msnc.loadImageStack(tomo_stack_files[0], self.config['data_filetype'],
                stacks[0]['img_offset'], 1)
            x_sum = np.sum(tomo_stack[0,:,:], 1)
            title = f'tomography image at theta={self.config["theta_range"]["start"]}'
            msnc.quickImshow(tomo_stack[0,:,:], title=title, name=tomo_field_pngname,
                    save_fig=True, save_only=True)
            msnc.quickPlot((range(x_sum.size), x_sum),
                    ([img_x_bounds[0], img_x_bounds[0]], [x_sum.min(), x_sum.max()], 'r-'),
                    ([img_x_bounds[1]-1, img_x_bounds[1]-1], [x_sum.min(), x_sum.max()], 'r-'),
                    title='sum over theta and y', name=detectorbounds_pngname,
                    save_fig=True, save_only=True)
            
            # Update config and save to file
            if preprocess is None:
                self.cf.config['preprocess'] = {'img_x_bounds' : img_x_bounds}
            else:
                preprocess['img_x_bounds'] = img_x_bounds
            self.cf.saveFile(self.config_out)
            return

        # For one tomography stack only: load the first image
        title = None
        msnc.quickImshow(self.tbf, title='bright field')
        if num_tomo_stacks == 1:
            tomo_stack = msnc.loadImageStack(tomo_stack_files[0], self.config['data_filetype'],
                stacks[0]['img_offset'], 1)
            title = f'tomography image at theta={self.config["theta_range"]["start"]}'
            msnc.quickImshow(tomo_stack[0,:,:], title=title)
            tomo_or_bright = pyip.inputNum('\nSelect image bounds from bright field (0) or '+
                    'from first tomography image (1): ', min=0, max=1)
        else:
            print('\nSelect image bounds from bright field')
            tomo_or_bright = 0
        if tomo_or_bright:
            x_sum = np.sum(tomo_stack[0,:,:], 1)
            use_bounds = 'no'
            if img_x_bounds[0] is not None and img_x_bounds[1] is not None:
                if img_x_bounds[0] < 0:
                    msnc.illegal_value('img_x_bounds[0]', img_x_bounds[0], 'config file')
                    img_x_bounds[0] = 0
                if not img_x_bounds[0] < img_x_bounds[1] <= x_sum.size:
                    msnc.illegal_value('img_x_bounds[1]', img_x_bounds[1], 'config file')
                    img_x_bounds[1] = x_sum.size
                tomo_tmp = tomo_stack[0,:,:]
                tomo_tmp[img_x_bounds[0],:] = tomo_stack[0,:,:].max()
                tomo_tmp[img_x_bounds[1]-1,:] = tomo_stack[0,:,:].max()
                title = f'tomography image at theta={self.config["theta_range"]["start"]}'
                msnc.quickImshow(tomo_stack[0,:,:], title=title)
                msnc.quickPlot((range(x_sum.size), x_sum),
                        ([img_x_bounds[0], img_x_bounds[0]], [x_sum.min(), x_sum.max()], 'r-'),
                        ([img_x_bounds[1]-1, img_x_bounds[1]-1], [x_sum.min(), x_sum.max()], 'r-'),
                        title='sum over theta and y')
                print(f'lower bound = {img_x_bounds[0]} (inclusive)\n'+
                        f'upper bound = {img_x_bounds[1]} (exclusive)]')
                use_bounds =  pyip.inputYesNo('Accept these bounds ([y]/n)?: ', blank=True)
            if use_bounds == 'no':
                img_x_bounds = msnc.selectImageBounds(tomo_stack[0,:,:], 0,
                        img_x_bounds[0], img_x_bounds[1], num_x_min,
                        f'tomography image at theta={self.config["theta_range"]["start"]}')
                if num_x_min is not None and img_x_bounds[1]-img_x_bounds[0]+1 < num_x_min:
                    logging.warning('Image bounds and pixel size prevent seamless stacking')
                tomo_tmp = tomo_stack[0,:,:]
                tomo_tmp[img_x_bounds[0],:] = tomo_stack[0,:,:].max()
                tomo_tmp[img_x_bounds[1]-1,:] = tomo_stack[0,:,:].max()
                title = f'tomography image at theta={self.config["theta_range"]["start"]}'
                msnc.quickImshow(tomo_stack[0,:,:], title=title, path=self.output_folder,
                        save_fig=self.save_plots, save_only=True)
                msnc.quickPlot(range(img_x_bounds[0], img_x_bounds[1]),
                        x_sum[img_x_bounds[0]:img_x_bounds[1]],
                        title='sum over theta and y', path=self.output_folder,
                        save_fig=self.save_plots, save_only=True)
        else:
            x_sum = np.sum(self.tbf, 1)
            use_bounds = 'no'
            if img_x_bounds[0] is not None and img_x_bounds[1] is not None:
                if img_x_bounds[0] < 0:
                    msnc.illegal_value('img_x_bounds[0]', img_x_bounds[0], 'config file')
                    img_x_bounds[0] = 0
                if not img_x_bounds[0] < img_x_bounds[1] <= x_sum.size:
                    msnc.illegal_value('img_x_bounds[1]', img_x_bounds[1], 'config file')
                    img_x_bounds[1] = x_sum.size
                msnc.quickPlot((range(x_sum.size), x_sum),
                        ([img_x_bounds[0], img_x_bounds[0]], [x_sum.min(), x_sum.max()], 'r-'),
                        ([img_x_bounds[1]-1, img_x_bounds[1]-1], [x_sum.min(), x_sum.max()], 'r-'),
                        title='sum over theta and y')
                print(f'lower bound = {img_x_bounds[0]} (inclusive)\n'+
                        f'upper bound = {img_x_bounds[1]} (exclusive)]')
                use_bounds =  pyip.inputYesNo('Accept these bounds ([y]/n)?: ', blank=True)
            if use_bounds == 'no':
                fit = msnc.fitStep(y=x_sum, model='rectangle', form='atan')
                x_low = fit.get('center1', None)
                x_upp = fit.get('center2', None)
                if (x_low is not None and x_low >= 0 and x_upp is not None and
                        x_low < x_upp < x_sum.size):
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
                if num_x_min is not None and img_x_bounds[1]-img_x_bounds[0]+1 < num_x_min:
                    logging.warning('Image bounds and pixel size prevent seamless stacking')
                msnc.quickPlot(range(img_x_bounds[0], img_x_bounds[1]),
                        x_sum[img_x_bounds[0]:img_x_bounds[1]],
                        title='sum over theta and y', path=self.output_folder,
                        save_fig=self.save_plots, save_only=True)
        logging.debug(f'img_x_bounds: {img_x_bounds}')

        if self.save_plots_only:
            msnc.clearFig('bright field')
            msnc.clearFig('sum over theta and y')
            if title:
                msnc.clearFig(title)

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
                if pyip.inputYesNo(
                        '\nDo you want to zoom in to reduce memory requirement (y/[n])? ',
                        blank=True) == 'yes':
                    zoom_perc = pyip.inputInt('    Enter zoom percentage [1, 100]: ',
                            min=1, max=100)
            else:
                zoom_perc = preprocess['zoom_perc']
                if msnc.is_num(zoom_perc, 1., 100.):
                    zoom_perc = int(zoom_perc)
                else:
                    msnc.illegal_value('zoom_perc', zoom_perc, 'config file')
                    zoom_perc = 100
        num_theta_skip = 0
        if not self.galaxy_flag:
            if preprocess is None or 'num_theta_skip' not in preprocess:
                if pyip.inputYesNo(
                        'Do you want to skip thetas to reduce memory requirement (y/[n])? ',
                        blank=True) == 'yes':
                    num_theta_skip = pyip.inputInt('    Enter the number skip theta interval'+
                            f' [0, {self.num_thetas-1}]: ', min=0, max=self.num_thetas-1)
            else:
                num_theta_skip = preprocess['num_theta_skip']
                if not msnc.is_int(num_theta_skip, 0):
                    msnc.illegal_value('num_theta_skip', num_theta_skip, 'config file')
                    num_theta_skip = 0
        logging.info(f'zoom_perc = {zoom_perc}')
        logging.info(f'num_theta_skip = {num_theta_skip}')

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

    def _genTomo(self, tomo_stack_files, available_stacks):
        """Generate tomography fields.
        """
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
            zoom_perc = preprocess['zoom_perc']
            num_theta_skip = preprocess['num_theta_skip']
        else:
            img_x_bounds = preprocess.get('img_x_bounds', [0, self.tbf.shape[0]])
            img_y_bounds = preprocess.get('img_y_bounds', [0, self.tbf.shape[1]])
            zoom_perc = 100
            num_theta_skip = 0

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
            t0 = time()
            tomo_stack = msnc.loadImageStack(tomo_stack_files[i], self.config['data_filetype'],
                    stack['img_offset'], self.config['theta_range']['num'], num_theta_skip,
                    img_x_bounds, img_y_bounds)
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
            if not self.galaxy_flag:
                index = stack['index']
                title = f'red stack fullres {index}'
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
                if not self.galaxy_flag:
                    title = f'red stack {zoom_perc}p {index}'
                    if not self.test_mode:
                        msnc.quickImshow(tomo_stack[0,:,:], title=title, path=self.output_folder,
                                save_fig=self.save_plots, save_only=self.save_plots_only)
    
            # Convert tomography stack from theta,row,column to row,theta,column
            tomo_stack = np.swapaxes(tomo_stack, 0, 1)

            # Save tomography stack to file
            if not self.galaxy_flag:
                if not self.test_mode:
                    self._saveTomo('red stack', tomo_stack, index)
                else:
                    np.savetxt(self.output_folder+f'red_stack_{index}.txt',
                            tomo_stack[0,:,:], fmt='%.6e')
                
            # Combine stacks
            t0 = time()
            self.tomo_stacks[i] = tomo_stack
            logging.debug(f'combining nstack took {time()-t0:.2f} seconds!')

            # Update config and save to file
            stack['preprocessed'] = True
            self.cf.saveFile(self.config_out)

        if self.tdf.size:
            del tdf
        del tbf

    def _reconstructOnePlane(self, tomo_plane_T, center, thetas_deg, eff_pixel_size,
            cross_sectional_dim, plot_sinogram=True):
        """Invert the sinogram for a single tomography plane.
        """
        # tomo_plane_T index order: column,theta
        assert(0 <= center < tomo_plane_T.shape[0])
        center_offset = center-tomo_plane_T.shape[0]/2
        two_offset = 2*int(np.round(center_offset))
        two_offset_abs = np.abs(two_offset)
        max_rad = int(0.5*(cross_sectional_dim/eff_pixel_size)*1.1) # 10% slack to avoid edge effects
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
        recon_sinogram = iradon(sinogram, theta=thetas_deg, circle=True)
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

    def _findCenterOnePlane(self, sinogram, row, thetas_deg, eff_pixel_size, cross_sectional_dim,
            tol=0.1):
        """Find center for a single tomography plane.
        """
        # sinogram index order: theta,column
        # need index order column,theta for iradon, so take transpose
        sinogram_T = sinogram.T
        center = sinogram.shape[1]/2

        # try automatic center finding routines for initial value
        tomo_center = tomopy.find_center_vo(sinogram)
        center_offset_vo = tomo_center-center
        print(f'Center at row {row} using Nghia Vo’s method = {center_offset_vo:.2f}')
        recon_plane = self._reconstructOnePlane(sinogram_T, tomo_center, thetas_deg,
                eff_pixel_size, cross_sectional_dim, False)
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
            recon_plane = self._reconstructOnePlane(sinogram_T, tomo_center, thetas_deg,
                    eff_pixel_size, cross_sectional_dim, False)
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
            return float(center_offset)

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
                recon_plane = self._reconstructOnePlane(sinogram_T, center_offset+center,
                        thetas_deg, eff_pixel_size, cross_sectional_dim, False)
                base_name=f'edges row{row} center_offset{center_offset}'
                self._plotEdgesOnePlane(recon_plane, base_name)
            if pyip.inputInt('\nContinue (0) or end the search (1): ', min=0, max=1):
                break

        del sinogram_T
        del recon_plane
        center_offset = pyip.inputNum(f'    Enter chosen center offset '+
                f'[{-int(center)}, {int(center)}]: ', min=-int(center), max=int(center))
        return float(center_offset)

    def _reconstructOneTomoStack(self, tomo_stack, thetas, row_bounds=None,
            center_offsets=[], sigma=0.1, rwidth=30, ncore=1, algorithm='gridrec',
            run_secondary_sirt=False, secondary_iter=100):
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
            if not (0 <= row_bounds[0] <= row_bounds[1] <= tomo_stack.shape[0]):
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

    def findImageFiles(self):
        """Find all available image files.
        """
        self.is_valid = True

        # Find dark field images
        dark_field = self.config['dark_field']
        img_start, num_imgs, dark_files = msnc.findImageFiles(
                dark_field['data_path'], self.config['data_filetype'], 'dark field')
        if img_start < 0 or num_imgs < 1:
            logging.error('Unable to find suitable dark field images')
            if dark_field['data_path']:
                self.is_valid = False
        dark_field['num'] = num_imgs
        dark_field['img_start'] = img_start
        logging.info(f'Number of dark field images = {dark_field["num"]}')
        logging.info(f'Dark field image start index = {dark_field["img_start"]}')

        # Find bright field images
        bright_field = self.config['bright_field']
        img_start, num_imgs, bright_files  = msnc.findImageFiles(
                bright_field['data_path'], self.config['data_filetype'], 'bright field')
        if img_start < 0 or num_imgs < 1:
            logging.error('Unable to find suitable bright field images')
            self.is_valid = False
        bright_field['num'] = num_imgs
        bright_field['img_start'] = img_start
        logging.info(f'Number of bright field images = {bright_field["num"]}')
        logging.info(f'Bright field image start index = {bright_field["img_start"]}')

        # Find tomography images
        tomo_stack_files = []
        for stack in self.config['stack_info']['stacks']:
            index = stack['index']
            img_start, num_imgs, tomo_files = msnc.findImageFiles(
                    stack['data_path'], self.config['data_filetype'], f'tomography set {index}')
            if img_start < 0 or num_imgs < 1:
                logging.error('Unable to find suitable tomography images')
                self.is_valid = False
            stack['num'] = num_imgs
            stack['img_start'] = img_start
            logging.info(f'Number of tomography images for set {index} = {stack["num"]}')
            logging.info(f'Tomography set {index} image start index = {stack["img_start"]}')
            tomo_stack_files.append(tomo_files)
            del tomo_files

        # Safe updated config
        if self.is_valid:
            self.cf.saveFile(self.config_out)

        return dark_files, bright_files, tomo_stack_files

    def genTomoStacks(self, tdf_files=None, tbf_files=None, tomo_stack_files=None,
            dark_field_pngname=None, bright_field_pngname=None, tomo_field_pngname=None,
            detectorbounds_pngname=None, output_name=None):
        """Preprocess tomography images.
        """
        # Try loading any already preprocessed stacks (skip in Galaxy)
        # preprocessed stack order for each one in stack: row,theta,column
        stack_info = self.config['stack_info']
        stacks = stack_info['stacks']
        num_tomo_stacks = stack_info['num']
        assert(num_tomo_stacks == len(self.tomo_stacks))
        available_stacks = [False]*num_tomo_stacks
        if self.galaxy_flag:
            assert(tdf_files is None or isinstance(tdf_files, list))
            assert(isinstance(tbf_files, list))
            assert(isinstance(tomo_stack_files, list))
            assert(num_tomo_stacks == len(tomo_stack_files))
            assert(isinstance(dark_field_pngname, str))
            assert(isinstance(bright_field_pngname, str))
            assert(isinstance(tomo_field_pngname, str))
            assert(isinstance(detectorbounds_pngname, str))
            assert(isinstance(output_name, str))
        else:
            if tdf_files:
                logging.warning('Ignoring tdf_files in genTomoStacks (only for Galaxy)')
            if tbf_files:
                logging.warning('Ignoring tbf_files in genTomoStacks (only for Galaxy)')
            if tomo_stack_files:
                logging.warning('Ignoring tomo_stack_files in genTomoStacks (only for Galaxy)')
            tdf_files, tbf_files, tomo_stack_files = self.findImageFiles()
            if not self.is_valid:
                return
            for i,stack in enumerate(stacks):
                if not self.tomo_stacks[i].size and stack.get('preprocessed', False):
                    self.tomo_stacks[i], available_stacks[i] = \
                            self._loadTomo('red stack', stack['index'])
            if dark_field_pngname:
                logging.warning('Ignoring dark_field_pngname in genTomoStacks (only for Galaxy)')
            if bright_field_pngname:
                logging.warning('Ignoring bright_field_pngname in genTomoStacks (only for Galaxy)')
            if tomo_field_pngname:
                logging.warning('Ignoring tomo_field_pngname in genTomoStacks (only for Galaxy)')
            if detectorbounds_pngname:
                logging.warning('Ignoring detectorbounds_pngname in genTomoStacks '+
                    '(only used in Galaxy)')
            if output_name:
                logging.warning('Ignoring output_name in genTomoStacks '+
                    '(only used in Galaxy)')

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
                self._genDark(tdf_files, dark_field_pngname)

            # Generate bright field
            self._genBright(tbf_files, bright_field_pngname)

            # Set vertical detector bounds for image stack
            self._setDetectorBounds(tomo_stack_files, tomo_field_pngname, detectorbounds_pngname)

            # Set zoom and/or theta skip to reduce memory the requirement
            self._setZoomOrSkip()

            # Generate tomography fields
            self._genTomo(tomo_stack_files, available_stacks)

            # Save tomography stack to file
            if self.galaxy_flag:
                t0 = time()
                logging.info(f'Saving preprocessed tomography stack to file ...')
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

    def findCenters(self):
        """Find rotation axis centers for the tomography stacks.
        """
        logging.debug('Find centers for tomography stacks')
        stacks = self.config['stack_info']['stacks']
        available_stacks = [stack['index'] for stack in stacks if stack.get('preprocessed', False)]
        logging.debug('Avaliable stacks: {available_stacks}')

        # Check for valid available center stack index
        find_center = self.config.get('find_center')
        if find_center and 'center_stack_index' in find_center:
            center_stack_index = find_center['center_stack_index']
            if (not isinstance(center_stack_index, int) or
                    center_stack_index not in available_stacks):
                msnc.illegal_value('center_stack_index', center_stack_index, 'config file')
            else:
                if self.test_mode:
                    find_center['completed'] = True
                    self.cf.saveFile(self.config_out)
                    return
                print('\nFound calibration center offset info for stack '+
                        f'{center_stack_index}')
                if pyip.inputYesNo('Do you want to use this again (y/n)? ') == 'yes':
                    find_center['completed'] = True
                    self.cf.saveFile(self.config_out)
                    return

        # Load the required preprocessed stack
        # preprocessed stack order: row,theta,column
        num_tomo_stacks = self.config['stack_info']['num']
        assert(len(stacks) == num_tomo_stacks)
        assert(len(self.tomo_stacks) == num_tomo_stacks)
        if num_tomo_stacks == 1:
            center_stack_index = stacks[0]['index']
            if not self.tomo_stacks[0].size:
                self.tomo_stacks[0], available = self._loadTomo('red stack', center_stack_index,
                    required=True)
            center_stack = self.tomo_stacks[0]
            if not center_stack.size:
                logging.error('Unable to load the required preprocessed tomography stack')
                return
            assert(stacks[0].get('preprocessed', False) == True)
        else:
            while True:
                center_stack_index = pyip.inputInt('\nEnter tomography stack index to get '
                        f'rotation axis centers {available_stacks}: ')
                while center_stack_index not in available_stacks:
                    center_stack_index = pyip.inputInt('\nEnter tomography stack index to get '
                            f'rotation axis centers {available_stacks}: ')
                tomo_stack_index = available_stacks.index(center_stack_index)
                if not self.tomo_stacks[tomo_stack_index].size:
                    self.tomo_stacks[tomo_stack_index], available = self._loadTomo(
                            'red stack', center_stack_index, required=True)
                center_stack = self.tomo_stacks[tomo_stack_index]
                if not center_stack.size:
                    logging.error(f'Unable to load the {center_stack_index}th '+
                        'preprocessed tomography stack, pick another one')
                else:
                    break
                assert(stacks[tomo_stack_index].get('preprocessed', False) == True)
        if find_center is None:
            self.config['find_center'] = {'center_stack_index' : center_stack_index}
        find_center = self.config['find_center']

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
        tomo_ref_heights = [stack['ref_height'] for stack in stacks]
        if num_tomo_stacks == 1:
            n1 = 0
            height = center_stack.shape[0]*eff_pixel_size
            if pyip.inputYesNo('\nDo you want to reconstruct the full samply height '+
                    f'({height:.3f} mm) (y/n)? ') == 'no':
                height = pyip.inputNum('\nEnter the desired reconstructed sample height '+
                        f'in mm [0, {height:.3f}]: ', min=0, max=height)
                n1 = int(0.5*(center_stack.shape[0]-height/eff_pixel_size))
        else:
            n1 = int((1.+(tomo_ref_heights[0]+center_stack.shape[0]*eff_pixel_size-
                tomo_ref_heights[1])/eff_pixel_size)/2)
        n2 = center_stack.shape[0]-n1
        logging.info(f'n1 = {n1}, n2 = {n2} (n2-n1) = {(n2-n1)*eff_pixel_size:.3f} mm')
        if not center_stack.size:
            RuntimeError('Center stack not loaded')
        msnc.quickImshow(center_stack[:,0,:], title=f'center stack theta={theta_start}',
                path=self.output_folder, save_fig=self.save_plots, save_only=self.save_plots_only)

        # Get cross sectional diameter in mm
        cross_sectional_dim = center_stack.shape[2]*eff_pixel_size
        logging.debug(f'cross_sectional_dim = {cross_sectional_dim}')

        # Determine center offset at sample row boundaries
        logging.info('Determine center offset at sample row boundaries')

        # Lower row center
        use_row = False
        use_center = False
        row = find_center.get('lower_row')
        if msnc.is_int(row, n1, n2-2):
            msnc.quickImshow(center_stack[:,0,:], title=f'theta={theta_start}', aspect='auto')
            use_row = pyip.inputYesNo('\nCurrent row index for lower center = '
                    f'{row}, use this value (y/n)? ')
            if self.save_plots_only:
                msnc.clearFig(f'theta={theta_start}')
            if use_row:
                center_offset = find_center.get('lower_center_offset')
                if msnc.is_num(center_offset):
                    use_center = pyip.inputYesNo('Current lower center offset = '+
                            f'{center_offset}, use this value (y/n)? ')
        if not use_center:
            if not use_row:
                msnc.quickImshow(center_stack[:,0,:], title=f'theta={theta_start}', aspect='auto')
                row = pyip.inputInt('\nEnter row index to find lower center '+
                        f'[[{n1}], {n2-2}]: ', min=n1, max=n2-2, blank=True)
                if row == '':
                    row = n1
                if self.save_plots_only:
                    msnc.clearFig(f'theta={theta_start}')
            # center_stack order: row,theta,column
            center_offset = self._findCenterOnePlane(center_stack[row,:,:], row, thetas_deg,
                    eff_pixel_size, cross_sectional_dim)
        logging.info(f'Lower center offset = {center_offset}')

        # Update config and save to file
        find_center['row_bounds'] = [n1, n2]
        find_center['lower_row'] = row
        find_center['lower_center_offset'] = center_offset
        self.cf.saveFile(self.config_out)
        lower_row = row

        # Upper row center
        use_row = False
        use_center = False
        row = find_center.get('upper_row')
        if msnc.is_int(row, lower_row+1, n2-1):
            msnc.quickImshow(center_stack[:,0,:], title=f'theta={theta_start}', aspect='auto')
            use_row = pyip.inputYesNo('\nCurrent row index for upper center = '
                    f'{row}, use this value (y/n)? ')
            if self.save_plots_only:
                msnc.clearFig(f'theta={theta_start}')
            if use_row:
                center_offset = find_center.get('upper_center_offset')
                if msnc.is_num(center_offset):
                    use_center = pyip.inputYesNo('Current upper center offset = '+
                            f'{center_offset}, use this value (y/n)? ')
        if not use_center:
            if not use_row:
                msnc.quickImshow(center_stack[:,0,:], title=f'theta={theta_start}', aspect='auto')
                row = pyip.inputInt('\nEnter row index to find upper center '+
                        f'[{lower_row+1}, [{n2-1}]]: ', min=lower_row+1, max=n2-1, blank=True)
                if row == '':
                    row = n2-1
                if self.save_plots_only:
                    msnc.clearFig(f'theta={theta_start}')
            # center_stack order: row,theta,column
            center_offset = self._findCenterOnePlane(center_stack[row,:,:], row, thetas_deg,
                    eff_pixel_size, cross_sectional_dim)
        logging.info(f'upper_center_offset = {center_offset}')
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
                    msnc.quickPlot((plane1[0,:],), (plane1_shifted[0,:],),
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
                    msnc.quickPlot((plane1[0,:],), (plane1_shifted[0,:],), 
                            title=f'stacks {stack1} {stack2} shifted {2*i} theta={start_theta}',
                            path=self.output_folder, save_fig=self.save_plots,
                            save_only=self.save_plots_only)
        del plane1, plane2, plane1_shifted

        # Update config file
        self.config = msnc.update('config.txt', 'check_centers', True, 'find_centers')

    def reconstructTomoStacks(self):
        """Reconstruct tomography stacks.
        """
        logging.debug('Reconstruct tomography stacks')

        # Get rotation axis rows and centers
        find_center = self.config['find_center']
        lower_row = find_center.get('lower_row')
        if lower_row is None:
            logging.error('Unable to read lower_row from config')
            return
        lower_center_offset = find_center.get('lower_center_offset')
        if lower_center_offset is None:
            logging.error('Unable to read lower_center_offset from config')
            return
        upper_row = find_center.get('upper_row')
        if upper_row is None:
            logging.error('Unable to read upper_row from config')
            return
        upper_center_offset = find_center.get('upper_center_offset')
        if upper_center_offset is None:
            logging.error('Unable to read upper_center_offset from config')
            return
        logging.debug(f'lower_row = {lower_row} upper_row = {upper_row}')
        assert(lower_row < upper_row)
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
            basetitle = 'recon stack full'
        else:
            basetitle = f'recon stack {zoom_perc}p'
        load_error = False
        stacks = self.config['stack_info']['stacks']
        for i,stack in enumerate(stacks):
            # Check if stack can be loaded
            # reconstructed stack order for each one in stack : row/z,x,y
            # preprocessed stack order for each one in stack: row,theta,column
            index = stack['index']
            available = False
            if stack.get('reconstructed', False):
                self.tomo_recon_stacks[i], available = self._loadTomo('recon stack', index)
            if self.tomo_recon_stacks[i].size or available:
                if self.tomo_stacks[i].size:
                    self.tomo_stacks[i] = np.array([])
                assert(stack.get('reconstructed', False) == True)
                continue
            else:
                stack['reconstructed'] = False
                if not self.tomo_stacks[i].size:
                    self.tomo_stacks[i], available = self._loadTomo('red stack', index,
                            required=True)
                if not self.tomo_stacks[i].size:
                    logging.error(f'Unable to load tomography stack {index} for reconstruction')
                    load_error = True
                    continue
                assert(0 <= lower_row < upper_row < self.tomo_stacks[i].shape[0])
                center_offsets = [lower_center_offset-lower_row*center_slope,
                        upper_center_offset+(self.tomo_stacks[i].shape[0]-1-upper_row)*center_slope]
                t0 = time()
                self.tomo_recon_stacks[i]= self._reconstructOneTomoStack(self.tomo_stacks[i],
                        thetas, center_offsets=center_offsets, sigma=0.1, ncore=self.ncore,
                        algorithm='gridrec', run_secondary_sirt=True, secondary_iter=25)
                logging.info(f'Reconstruction of stack {index} took {time()-t0:.2f} seconds!')
                if not self.test_mode:
                    row_slice = int(self.tomo_stacks[i].shape[0]/2) 
                    title = f'{basetitle} {index} slice{row_slice}'
                    msnc.quickImshow(self.tomo_recon_stacks[i][row_slice,:,:], title=title,
                            path=self.output_folder, save_fig=self.save_plots,
                            save_only=self.save_plots_only)
                    msnc.quickPlot(self.tomo_recon_stacks[i]
                            [row_slice,int(self.tomo_recon_stacks[i].shape[2]/2),:],
                            title=f'{title} cut{int(self.tomo_recon_stacks[i].shape[2]/2)}',
                            path=self.output_folder, save_fig=self.save_plots,
                            save_only=self.save_plots_only)
                    self._saveTomo('recon stack', self.tomo_recon_stacks[i], index)
#                else:
#                    np.savetxt(self.output_folder+f'recon_stack_{index}.txt',
#                            self.tomo_recon_stacks[i][row_slice,:,:], fmt='%.6e')
                self.tomo_stacks[i] = np.array([])

                # Update config and save to file
                stack['reconstructed'] = True
                self.cf.saveFile(self.config_out)

    def combineTomoStacks(self):
        """Combine the reconstructed tomography stacks.
        """
        # stack order: stack,row(z),x,y
        logging.debug('Combine reconstructed tomography stacks')
        # Load any unloaded reconstructed stacks
        stacks = self.config['stack_info']['stacks']
        for i,stack in enumerate(stacks):
            if not self.tomo_recon_stacks[i].size and stack.get('reconstructed', False):
                self.tomo_recon_stacks[i], available = self._loadTomo('recon stack',
                        stack['index'], required=True)
            if not self.tomo_recon_stacks[i].size or not available:
                logging.error(f'Unable to load reconstructed stack {stack["index"]}')
                return
            if i:
                if (self.tomo_recon_stacks[i].shape[1] != self.tomo_recon_stacks[0].shape[1] or
                        self.tomo_recon_stacks[i].shape[1] != self.tomo_recon_stacks[0].shape[1]):
                    logging.error('Incompatible reconstructed tomography stack dimensions')
                    return

        # Get center stack boundaries
        row_bounds = self.config['find_center']['row_bounds']
        if not msnc.is_index_range(row_bounds, 0, self.tomo_recon_stacks[0].shape[0]):
            msnc.illegal_value('row_bounds', row_bounds, 'config file')
            return

        # Selecting xy bounds
        tomosum = 0
        #RV FIX :=
        #[tomosum := tomosum+np.sum(tomo_recon_stack, axis=(0,2)) for tomo_recon_stack in
        #        self.tomo_recon_stacks]
        combine_stacks =self.config.get('combine_stacks')
        if combine_stacks and 'x_bounds' in combine_stacks:
            x_bounds = combine_stacks['x_bounds']
            if not msnc.is_index_range(x_bounds, 0, self.tomo_recon_stacks[0].shape[1]):
                msnc.illegal_value('x_bounds', x_bounds, 'config file')
            elif not self.test_mode:
                msnc.quickPlot(tomosum, title='recon stack sum yz')
                if pyip.inputYesNo('\nCurrent image x-bounds: '+
                        f'[{x_bounds[0]}, {x_bounds[1]}], use these values ([y]/n)? ',
                        blank=True) == 'no':
                    if pyip.inputYesNo(
                            'Do you want to change the image x-bounds ([y]/n)? ',
                            blank=True) == 'no':
                        x_bounds = [0, self.tomo_recon_stacks[0].shape[1]]
                    else:
                        x_bounds = msnc.selectArrayBounds(tomosum, title='recon stack sum yz')
        else:
            msnc.quickPlot(tomosum, title='recon stack sum yz')
            if pyip.inputYesNo('\nDo you want to change the image x-bounds (y/n)? ') == 'no':
                x_bounds = [0, self.tomo_recon_stacks[0].shape[1]]
            else:
                x_bounds = msnc.selectArrayBounds(tomosum, title='recon stack sum yz')
        if False and self.test_mode:
            np.savetxt(self.output_folder+'recon_stack_sum_yz.txt',
                    tomosum[x_bounds[0]:x_bounds[1]], fmt='%.6e')
        if self.save_plots_only:
            msnc.clearFig('recon stack sum yz')
        tomosum = 0
        #RV FIX :=
        #[tomosum := tomosum+np.sum(tomo_recon_stack, axis=(0,1)) for tomo_recon_stack in
        #        self.tomo_recon_stacks]
        if combine_stacks and 'y_bounds' in combine_stacks:
            y_bounds = combine_stacks['y_bounds']
            if not msnc.is_index_range(x_bounds, 0, self.tomo_recon_stacks[0].shape[1]):
                msnc.illegal_value('y_bounds', y_bounds, 'config file')
            elif not self.test_mode:
                msnc.quickPlot(tomosum, title='recon stack sum xz')
                if pyip.inputYesNo('\nCurrent image y-bounds: '+
                        f'[{y_bounds[0]}, {y_bounds[1]}], use these values ([y]/n)? ',
                        blank=True) == 'no':
                    if pyip.inputYesNo(
                            'Do you want to change the image y-bounds ([y]/n)? ',
                            blank=True) == 'no':
                        y_bounds = [0, self.tomo_recon_stacks[0].shape[1]]
                    else:
                        y_bounds = msnc.selectArrayBounds(tomosum, title='recon stack sum yz')
        else:
            msnc.quickPlot(tomosum, title='recon stack sum xz')
            if pyip.inputYesNo('\nDo you want to change the image y-bounds (y/n)? ') == 'no':
                y_bounds = [0, self.tomo_recon_stacks[0].shape[1]]
            else:
                y_bounds = msnc.selectArrayBounds(tomosum, title='recon stack sum xz')
        if False and self.test_mode:
            np.savetxt(self.output_folder+'recon_stack_sum_xz.txt',
                    tomosum[y_bounds[0]:y_bounds[1]], fmt='%.6e')
        if self.save_plots_only:
            msnc.clearFig('recon stack sum xz')

        # Combine reconstructed tomography stacks
        logging.info(f'Combining reconstructed stacks ...')
        t0 = time()
        num_tomo_stacks = self.config['stack_info']['num']
        if num_tomo_stacks == 1:
            low_bound = row_bounds[0]
        else:
            low_bound = 0
        tomo_recon_combined = self.tomo_recon_stacks[0][low_bound:row_bounds[1]:,
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
        tomosum = np.sum(tomo_recon_combined, axis=(1,2))
        if self.test_mode:
            zoom_perc = self.config['preprocess'].get('zoom_perc', 100)
            filename = 'recon combined sum xy'
            if zoom_perc is None or zoom_perc == 100:
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
            combine_stacks =self.config.get('combine_stacks')

            # Update config and save to file
            if combine_stacks:
                combine_stacks['x_bounds'] = x_bounds
                combine_stacks['y_bounds'] = y_bounds
            else:
                self.config['combine_stacks'] = {'x_bounds' : x_bounds, 'y_bounds' : y_bounds}
            self.cf.saveFile(self.config_out)
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

        # Save combined reconstructed tomo stacks
        base_name = 'recon combined'
        combined_stacks = []
        for stack in stacks:
            base_name += f' {stack["index"]}'
            combined_stacks.append(stack['index'])
        self._saveTomo(base_name, tomo_recon_combined)

        # Update config and save to file
        if combine_stacks:
            combine_stacks['x_bounds'] = x_bounds
            combine_stacks['y_bounds'] = y_bounds
            combine_stacks['stacks'] = combined_stacks
        else:
            self.config['combine_stacks'] = {'x_bounds' : x_bounds, 'y_bounds' : y_bounds,
                    'stacks' : combined_stacks}
        self.cf.saveFile(self.config_out)

def runTomo(config_file=None, config_dict=None, output_folder='.', log_level='INFO',
        test_mode=False):
    """Run a tomography analysis.
    """
    # Instantiate Tomo object
    tomo = Tomo(config_file=config_file, output_folder=output_folder, log_level=log_level,
            test_mode=test_mode)
    if not tomo.is_valid:
        raise ValueError('Invalid config and/or detector file provided.')

    # Preprocess the image files
    num_tomo_stacks = tomo.config['stack_info']['num']
    assert(num_tomo_stacks == len(tomo.tomo_stacks))
    preprocess = tomo.config.get('preprocess', None)
    preprocessed_stacks = []
    if preprocess:
        preprocessed_stacks = [stack['index'] for stack in tomo.config['stack_info']['stacks']
            if stack.get('preprocessed', False)]
    if not len(preprocessed_stacks):
        tomo.genTomoStacks()
        if not tomo.is_valid:
            IOError('Unable to load all required image files.')
        find_center = tomo.config.get('find_center')
        if find_center and find_center.get('completed', False):
            center_stack_index = find_center['center_stack_index']
            if not center_stack_index in preprocessed_stacks:
                find_center['completed'] = False
#RV FIX
#        tomo.config.pop('check_center', 'check_center not found')
#        combined_stacks = tomo.config.get('combined_stacks')
#        if combined_stacks:
#            combined_stacks['completed'] = False
        tomo.cf.saveFile(tomo.config_out)

    # Find centers
    find_center = tomo.config.get('find_center')
    if find_center is None or not find_center.get('completed', False):
        tomo.findCenters()

    # Check centers
    #if num_tomo_stacks > 1 and not tomo.config.get('check_centers', False):
    #    tomo.checkCenters()

    # Reconstruct tomography stacks
    if len(tomo.config.get('reconstructed_stacks', [])) != tomo.config['stack_info']['num']:
        tomo.reconstructTomoStacks()

    # Combine reconstructed tomography stacks
    combined_stacks = tomo.config.get('combined_stacks')
    if combined_stacks is None or not combined_stacks.get('completed', False):
        tomo.combineTomoStacks()

#%%============================================================================
if __name__ == '__main__':
    # Parse command line arguments
    arguments = sys.argv[1:]
    config_file = None
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
    if config_file is None:
        if os.path.isfile('config.yaml'):
            config_file = 'config.yaml'
        else:
            config_file = 'config.txt'

    # Set basic log configuration
    logging_format = '%(asctime)s : %(levelname)s - %(module)s : %(funcName)s - %(message)s'
    if not test_mode:
        level = getattr(logging, log_level.upper(), None)
        if not isinstance(level, int):
            raise ValueError(f'Invalid log_level: {log_level}')
        logging.basicConfig(format=logging_format, level=level, force=True,
                handlers=[logging.StreamHandler()])

    logging.debug(f'config_file = {config_file}')
    logging.debug(f'output_folder = {output_folder}')
    logging.debug(f'log_level = {log_level}')
    logging.debug(f'test_mode = {test_mode}')

    # Run tomography analysis
    runTomo(config_file=config_file, output_folder=output_folder, log_level=log_level,
            test_mode=test_mode)

#%%============================================================================
    input('Press any key to continue')
#%%============================================================================

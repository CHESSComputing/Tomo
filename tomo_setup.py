#!/usr/bin/env python3

import logging

import os
import sys
import re
import yaml
import argparse
import numpy as np
import tracemalloc

from tomo import Tomo
from general import get_trailing_int

#from memory_profiler import profile
#@profile
def __main__():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Setup tomography reconstruction')
    parser.add_argument('--inputconfig',
            default='inputconfig.txt',
            help='Input config from tool form')
    parser.add_argument('--inputfiles',
            default='inputfiles.txt',
            help='Input file collections')
    parser.add_argument('-c', '--config',
            help='Input config file')
    parser.add_argument('--detector',
            help='Detector info (number of rows and columns, and pixel size)')
    parser.add_argument('--num_theta',
            help='Number of theta angles')
    parser.add_argument('--theta_range',
            help='Theta range (lower bound, upper bound)')
    parser.add_argument('--output_config',
            help='Output config')
    parser.add_argument('--output_data',
            help='Preprocessed tomography data')
    parser.add_argument('-l', '--log', 
            type=argparse.FileType('w'),
            default=sys.stdout,
            help='Log file')
    args = parser.parse_args()

    # Starting memory monitoring
    tracemalloc.start()

    # Set basic log configuration
    logging_format = '%(asctime)s : %(levelname)s - %(module)s : %(funcName)s - %(message)s'
    log_level = 'INFO'
    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f'Invalid log_level: {log_level}')
    logging.basicConfig(format=logging_format, level=level, force=True,
            handlers=[logging.StreamHandler()])

    # Check command line arguments
    logging.info(f'config = {args.config}')
    if args.detector is None:
        logging.info(f'detector = {args.detector}')
    else:
        logging.info(f'detector = {args.detector.split()}')
    logging.info(f'num_theta = {args.num_theta}')
    if args.theta_range is None:
        logging.info(f'theta_range = {args.theta_range}')
    else:
        logging.info(f'theta_range = {args.theta_range.split()}')
    logging.info(f'output_config = {args.output_config}')
    logging.info(f'output_data = {args.output_data}')
    logging.info(f'log = {args.log}')
    logging.debug(f'is log stdout? {args.log is sys.stdout}')
    if args.detector is not None and len(args.detector.split()) != 3:
        raise ValueError(f'Invalid detector: {args.detector}')
    if args.num_theta is None or int(args.num_theta) < 1:
        raise ValueError(f'Invalid num_theta: {args.num_theta}')
    if args.theta_range is not None and len(args.theta_range.split()) != 2:
        raise ValueError(f'Invalid theta_range: {args.theta_range}')
    num_theta = int(args.num_theta)

    # Read and check tool config input
    inputconfig = []
    with open(args.inputconfig) as f:
        inputconfig = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    assert(len(inputconfig) >= 6)
    config_type = inputconfig[0]
    input_type = inputconfig[1]
    num_stack = int(inputconfig[2])
    stack_types = [x.strip() for x in inputconfig[3].split()]
    num_imgs = [int(x.strip()) for x in inputconfig[4].split()]
    img_offsets = [int(x.strip()) for x in inputconfig[5].split()]
    if config_type == 'config_manual':
        assert(len(inputconfig) == 7)
        ref_heights = [float(x.strip()) for x in inputconfig[6].split()]
        assert(args.detector is not None)
        assert(args.theta_range is not None)
    else:
        ref_heights = None
    logging.info(f'config_type = {config_type} {type(config_type)}')
    logging.info(f'input_type = {input_type} {type(input_type)}')
    logging.info(f'num_stack = {num_stack} {type(num_stack)}')
    logging.info(f'stack_types = {stack_types} {type(stack_types)}')
    logging.info(f'num_imgs = {num_imgs} {type(num_imgs)}')
    logging.info(f'img_offsets = {img_offsets} {type(img_offsets)}')
    logging.info(f'ref_heights = {ref_heights} {type(ref_heights)}')
    if config_type != 'config_file' and config_type != 'config_manual':
        raise ValueError('Invalid input config provided.')
    if input_type != 'collections' and input_type != 'files':
        raise ValueError('Invalid input config provided.')
    if len(stack_types) != num_stack:
        raise ValueError('Invalid input config provided.')
    if len(num_imgs) != num_stack:
        raise ValueError('Invalid input config provided.')
    if len(img_offsets) != num_stack:
        raise ValueError('Invalid input config provided.')
    if ref_heights is not None and len(ref_heights) != num_stack:
        raise ValueError('Invalid input config provided.')

    # Read input files and collect data files info
    datasets = []
    with open(args.inputfiles) as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            fields = [x.strip() for x in line.split('\t')]
            filepath = fields[0]
            element_identifier = fields[1] if len(fields) > 1 else fields[0].split('/')[-1]
            datasets.append({'element_identifier' : element_identifier, 'filepath' : filepath})
    logging.debug(f'datasets:\n{datasets}')
    if input_type == 'files' and len(datasets) != num_stack:
        raise ValueError('Inconsistent number of input files provided.')

    # Read and sort data files
    collections = []
    stack_index = 1
    for i, dataset in enumerate(datasets):
        if input_type == 'collections':
            element_identifier = [x.strip() for x in dataset['element_identifier'].split('_')]
            if len(element_identifier) > 1:
                name = element_identifier[0]
            else:
                name = 'other'
        else:
            if stack_types[i] == 'tdf' or stack_types[i] == 'tbf':
                name = stack_types[i]
            elif stack_types[i] == 'data':
                name = f'set{stack_index}'
                stack_index += 1
            else:
                raise ValueError('Invalid input config provided.')
        filepath = dataset['filepath']
        if not len(collections):
            collections = [{'name' : name, 'filepaths' : [filepath]}]
        else:
            collection = [c for c in collections if c['name'] == name]
            if len(collection):
                collection[0]['filepaths'].append(filepath)
            else:
                collection = {'name' : name, 'filepaths' : [filepath]}
                collections.append(collection)
    logging.debug(f'collections:\n{collections}')

    # Instantiate Tomo object
    tomo = Tomo(config_file=args.config, config_out=args.output_config, log_level=log_level,
            log_stream=args.log, galaxy_flag=True)
    if config_type == 'config_file':
        if not tomo.is_valid:
            raise ValueError('Invalid config file provided.')
    else:
        assert(tomo.config is None)
        tomo.config = {}
    logging.debug(f'config:\n{tomo.config}')

    # Set detector inputs
    if config_type == 'config_manual':
        detector = args.detector.split()
        tomo.config['detector'] = {'rows' : int(detector[0]),
                'columns' : int(detector[1]), 'pixel_size' : float(detector[2])}

    # Set theta inputs
    config_theta_range = tomo.config.get('theta_range')
    if config_theta_range is None:
        tomo.config['theta_range'] = {'num' : num_theta}
        config_theta_range = tomo.config['theta_range']
    else:
        config_theta_range['num'] = num_theta
    if config_type == 'config_manual':
        theta_range = args.theta_range.split()
        config_theta_range['start'] = float(theta_range[0])
        config_theta_range['end'] = float(theta_range[1])

    # Find dark field files
    dark_field = tomo.config.get('dark_field')
    tdf_files = [c['filepaths'] for c in collections if c['name'] == 'tdf']
    if len(tdf_files) != 1 or len(tdf_files[0]) < 1:
        logging.warning('Unable to obtain dark field files')
        if config_type == 'config_file':
            assert(dark_field is not None)
            assert(dark_field['data_path'] is None)
            assert(dark_field['img_start'] == -1)
            assert(not dark_field['num'])
        else:
            tomo.config['dark_field'] = {'data_path' : None, 'img_start' : -1, 'num' : 0}
        tdf_files = [None]
        num_collections = 0
    else:
        if config_type == 'config_file':
            assert(dark_field is not None)
            assert(dark_field['data_path'] is not None)
            assert(dark_field.get('img_start') is not None)
        else:
            tomo.config['dark_field'] = {'data_path' : tdf_files[0], 'img_start' : 0}
            dark_field = tomo.config['dark_field']
        tdf_index = [i for i,c in enumerate(collections) if c['name'] == 'tdf']
        tdf_index_check = [i for i,s in enumerate(stack_types) if s == 'tdf']
        if tdf_index != tdf_index_check:
            raise ValueError(f'Inconsistent tdf_index ({tdf_index} vs. {tdf_index_check}).')
        tdf_index = tdf_index[0]
        dark_field['img_offset'] = img_offsets[tdf_index]
        dark_field['num'] = num_imgs[tdf_index]
        num_collections = 1

    # Find bright field files
    bright_field = tomo.config.get('bright_field')
    tbf_files = [c['filepaths'] for c in collections if c['name'] == 'tbf']
    if len(tbf_files) != 1 or len(tbf_files[0]) < 1:
        exit('Unable to obtain bright field files')
    if config_type == 'config_file':
        assert(bright_field is not None)
        assert(bright_field['data_path'] is not None)
        assert(bright_field.get('img_start') is not None)
    else:
        tomo.config['bright_field'] = {'data_path' : tbf_files[0], 'img_start' : 0}
        bright_field = tomo.config['bright_field']
    tbf_index = [i for i,c in enumerate(collections) if c['name'] == 'tbf']
    tbf_index_check = [i for i,s in enumerate(stack_types) if s == 'tbf']
    if tbf_index != tbf_index_check:
        raise ValueError(f'Inconsistent tbf_index ({tbf_index} vs. {tbf_index_check}).')
    tbf_index = tbf_index[0]
    bright_field['img_offset'] = img_offsets[tbf_index]
    bright_field['num'] = num_imgs[tbf_index]
    num_collections += 1

    # Find tomography files
    stack_info = tomo.config.get('stack_info')
    if config_type == 'config_file':
        assert(stack_info is not None)
        if stack_info['num'] != len(collections) - num_collections:
            raise ValueError('Inconsistent number of tomography data image sets')
        assert(stack_info.get('stacks') is not None)
        for stack in stack_info['stacks']:
            assert(stack['data_path'] is not None)
            assert(stack.get('img_start') is not None)
            assert(stack.get('index') is not None)
            assert(stack.get('ref_height') is not None)
    else:
        tomo.config['stack_info'] = {'num' : len(collections) - num_collections, 'stacks' : []}
        stack_info = tomo.config['stack_info']
        for i in range(stack_info['num']):
            stack_info['stacks'].append({'img_start' : 0, 'index' : i+1})
    tomo_stack_files = []
    for stack in stack_info['stacks']:
        index = stack['index']
        tomo_files = [c['filepaths'] for c in collections if c['name'] == f'set{index}']
        if len(tomo_files) != 1 or len(tomo_files[0]) < 1:
            exit(f'Unable to obtain tomography images for set {index}')
        tomo_index = [i for i,c in enumerate(collections) if c['name'] == f'set{index}']
        if len(tomo_index) != 1:
            raise ValueError(f'Illegal tomo_index ({tomo_index}).')
        tomo_index = tomo_index[0]
        stack['img_offset'] = img_offsets[tomo_index]
        assert(num_imgs[tomo_index] == -1)
        stack['num'] = num_theta
        if config_type == 'config_manual':
            if len(tomo_files) == 1:
                stack['data_path'] = tomo_files[0]
            stack['ref_height'] = ref_heights[tomo_index]
        tomo_stack_files.append(tomo_files[0])
        num_collections += 1
    if num_collections != num_stack:
        raise ValueError('Inconsistent number of data image sets')

    # Preprocess the image files
    galaxy_param = {'tdf_files' : tdf_files[0], 'tbf_files' : tbf_files[0],
            'tomo_stack_files' : tomo_stack_files, 'output_name' : args.output_data}
    tomo.genTomoStacks(galaxy_param)
    if not tomo.is_valid:
        IOError('Unable to load all required image files.')

    # Displaying memory usage
    logging.info(f'Memory usage: {tracemalloc.get_traced_memory()}')
 
    # stopping memory monitoring
    tracemalloc.stop()

if __name__ == "__main__":
    __main__()


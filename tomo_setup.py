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

#from memory_profiler import profile
#@profile
def __main__():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Setup tomography reconstruction')
    parser.add_argument('-i', '--inputfiles',
            default='inputfiles.txt',
            help='Input file collections')
    parser.add_argument('-c', '--config',
            help='Input config')
    parser.add_argument('--theta_range',
            help='Theta range (lower bound, upper bound, number of angles)')
    parser.add_argument('--output_config',
            help='Output config')
    parser.add_argument('--output_data',
            help='Preprocessed tomography data')
    parser.add_argument('-l', '--log', 
            type=argparse.FileType('w'),
            default=sys.stdout,
            help='Log file')
    parser.add_argument('tomo_ranges', metavar='N', type=int, nargs='+')
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

    logging.debug(f'config = {args.config}')
    logging.debug(f'theta_range = {args.theta_range.split()}')
    logging.debug(f'output_config = {args.output_config}')
    logging.debug(f'output_data = {args.output_data}')
    logging.debug(f'log = {args.log}')
    logging.debug(f'is log stdout? {args.log is sys.stdout}')
    logging.debug(f'tomoranges = {args.tomo_ranges}')

    # Read input files and collect data files info
    datasets = []
    with open(args.inputfiles) as cf:
        for line in cf:
            if not line.strip() or line.startswith('#'):
                continue
            fields = [x.strip() for x in line.split('\t')]
            filepath = fields[0]
            element_identifier = fields[1] if len(fields) > 1 else fields[0].split('/')[-1]
            datasets.append({'element_identifier' : fields[1], 'filepath' : filepath})
    logging.debug(f'datasets:\n{datasets}')

    # Read and sort data files
    collections = []
    for dataset in datasets:
        element_identifier = [x.strip() for x in dataset['element_identifier'].split('_')]
        if len(element_identifier) > 1:
            name = element_identifier[0]
        else:
            name = 'other'
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
    if len(args.tomo_ranges) != 2*len(collections):
        raise ValueError('Inconsistent tomo ranges size.')

    # Instantiate Tomo object
    tomo = Tomo(config_file=args.config, config_out=args.output_config, log_level=log_level,
            log_stream=args.log, galaxy_flag=True)
    if not tomo.is_valid:
        raise ValueError('Invalid config file provided.')
    logging.debug(f'config:\n{tomo.config}')

    # Set theta inputs
    theta_range = args.theta_range.split()
    config_theta_range = tomo.config.get('theta_range')
    if config_theta_range is None:
        config_tomo.config['theta_range'] = {'start' : float(theta_range[0]),
            'end' : float(theta_range[1]), 'num' : int(theta_range[2])}
    else:
        config_theta_range['start'] = float(theta_range[0])
        config_theta_range['end'] = float(theta_range[1])
        config_theta_range['num'] = int(theta_range[2])

    # Find dark field files
    dark_field = tomo.config['dark_field']
    tdf_files = [c['filepaths'] for c in collections if c['name'] == 'tdf']
    if len(tdf_files) != 1 or len(tdf_files[0]) < 1:
        logging.warning('Unable to obtain dark field files')
        assert(dark_field['data_path'] is None)
        assert(dark_field['img_start'] == -1)
        assert(not dark_field['num'])
        tdf_files = [None]
        num_collections = 0
    else:
        dark_field['img_offset'] = args.tomo_ranges[0]
        dark_field['num'] = args.tomo_ranges[1]
        num_collections = 1

    # Find bright field files
    bright_field = tomo.config['bright_field']
    bright_field['img_offset'] = args.tomo_ranges[2*num_collections]
    bright_field['num'] = args.tomo_ranges[2*num_collections+1]
    tbf_files = [c['filepaths'] for c in collections if c['name'] == 'tbf']
    if len(tbf_files) != 1 or len(tbf_files[0]) < 1:
        exit('Unable to obtain bright field files')
    num_collections += 1

    # Find tomography files
    stack_info = tomo.config['stack_info']
    if stack_info['num'] != len(collections) - num_collections:
        raise ValueError('Inconsistent number of tomography data image sets')
    tomo_stack_files = []
    for stack in stack_info['stacks']:
        stack['img_offset'] = args.tomo_ranges[2*num_collections]
        stack['num'] = args.tomo_ranges[2*num_collections+1]
        tomo_files = [c['filepaths'] for c in collections if c['name'] == f'set{stack["index"]}']
        if len(tomo_files) != 1 or len(tomo_files[0]) < 1:
            exit(f'Unable to obtain tomography images for set {stack["index"]}')
        tomo_stack_files.append(tomo_files[0])
        num_collections += 1

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


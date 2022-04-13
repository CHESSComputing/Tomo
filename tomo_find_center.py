#!/usr/bin/env python3

import logging

import sys
import argparse
import tracemalloc

from tomo import Tomo

def __main__():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Find the center axis for a tomography reconstruction')
    parser.add_argument('-i', '--input_stacks',
            required=True, help='Preprocessed image file stacks')
    parser.add_argument('-c', '--config',
            required=True, help='Input config')
    parser.add_argument('--row_bounds',
            required=True, nargs=2, type=int, help='Reconstruction row bounds')
    parser.add_argument('--center_rows',
            required=True, nargs=2, type=int, help='Center finding rows')
    parser.add_argument('--center_type_selector',
            help='Reconstruct slices for a set of center positions?')
    parser.add_argument('--set_center',
            type=int, help='Set center ')
    parser.add_argument('--set_range',
            type=float, help='Set range')
    parser.add_argument('--set_step',
            type=float, help='Set step')
    parser.add_argument('--output_config',
            required=True, help='Output config')
    parser.add_argument('-l', '--log', 
            type=argparse.FileType('w'), default=sys.stdout, help='Log file')
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

    logging.debug(f'input_stacks = {args.input_stacks}')
    logging.debug(f'config = {args.config}')
    logging.debug(f'row_bounds = {args.row_bounds} {type(args.row_bounds)}')
    logging.debug(f'center_rows = {args.center_rows} {type(args.center_rows)}')
    logging.debug(f'center_type_selector = {args.center_type_selector}')
    logging.debug(f'set_center = {args.set_center}')
    logging.debug(f'set_range = {args.set_range}')
    logging.debug(f'set_step = {args.set_step}')
    logging.debug(f'output_config = {args.output_config}')
    logging.debug(f'log = {args.log}')
    logging.debug(f'is log stdout? {args.log is sys.stdout}')

    # Instantiate Tomo object
    tomo = Tomo(config_file=args.config, config_out=args.output_config, log_level=log_level,
            log_stream=args.log, galaxy_flag=True)
    if not tomo.is_valid:
        raise ValueError('Invalid config file provided.')
    logging.debug(f'config:\n{tomo.config}')

    # Load preprocessed image files
    tomo.loadTomoStacks(args.input_stacks)

    # Find centers
    galaxy_param = {'row_bounds' : args.row_bounds, 'center_rows' : args.center_rows,
            'center_type_selector' : args.center_type_selector, 'set_center' : args.set_center, 
            'set_range' : args.set_range, 'set_step' : args.set_step}
    tomo.findCenters(galaxy_param)

    # Displaying memory usage
    logging.info(f'Memory usage: {tracemalloc.get_traced_memory()}')

    # stopping memory monitoring
    tracemalloc.stop()

if __name__ == "__main__":
    __main__()


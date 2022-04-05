#!/usr/bin/env python3

import logging

import sys
import argparse

from tomo import Tomo

def __main__():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Perfrom a tomography reconstruction')
    parser.add_argument('-i', '--input_stacks',
            help='Preprocessed image file stacks')
    parser.add_argument('-c', '--config',
            help='Input config')
    parser.add_argument('--output_config',
            help='Output config')
    parser.add_argument('--output_data',
            help='Reconstructed tomography data')
    parser.add_argument('-l', '--log', 
            type=argparse.FileType('w'),
            default=sys.stdout,
            help='Log file')
    args = parser.parse_args()

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
    logging.debug(f'output_config = {args.output_config}')
    logging.debug(f'output_data = {args.output_data}')
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

    # Reconstruct tomography stacks
    tomo.reconstructTomoStacks(args.output_data)

if __name__ == "__main__":
    __main__()


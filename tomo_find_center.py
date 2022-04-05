#!/usr/bin/env python3

import logging

#import os
import sys
import re
#import yaml
import argparse
#import numpy as np

from tomo import Tomo
#import msnc_tools as msnc

def __main__():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Setup tomography reconstruction')
    parser.add_argument('-i', '--input_stacks',
            help='Preprocessed image file stacks')
    parser.add_argument('-c', '--config',
            help='Input config')
    parser.add_argument('--row_bounds',
            help='Reconstruction row bounds')
    parser.add_argument('--center_rows',
            help='Center finding rows')
    parser.add_argument('--output_config',
            help='Output config')
    parser.add_argument('--recon_center_low',
            help='Lower reconstructed slice center')
    parser.add_argument('--recon_center_upp',
            help='Upper reconstructed slice center')
    parser.add_argument('-l', '--log', 
            type=argparse.FileType('w'),
            default=sys.stdout,
            help='Log file')
    args = parser.parse_args()

    indexRegex = re.compile(r'\d+')
    row_bounds = [int(i) for i in indexRegex.findall(args.row_bounds)]
    center_rows = [int(i) for i in indexRegex.findall(args.center_rows)]

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
    logging.debug(f'row_bounds = {args.row_bounds} {row_bounds}')
    logging.debug(f'center_rows = {args.center_rows} {center_rows}')
    logging.debug(f'output_config = {args.output_config}')
    logging.debug(f'recon_center_low = {args.recon_center_low}')
    logging.debug(f'recon_center_uppm = {args.recon_center_upp}')
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
    tomo.findCenters(row_bounds, center_rows, args.recon_center_low, args.recon_center_upp)

if __name__ == "__main__":
    __main__()


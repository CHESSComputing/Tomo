#!/usr/bin/env python3

import logging

import sys
import argparse
import numpy as np

def __main__():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Read a reconstructed image')
    parser.add_argument('-i', '--input_image',
            help='Reconstructed image file')
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

    logging.info(f'input_image = {args.input_image}')
    logging.debug(f'log = {args.log}')
    logging.debug(f'is log stdout? {args.log is sys.stdout}')

    # Load image
    f = np.load(args.input_image)
    logging.info(f'f shape = {f.shape}')

if __name__ == "__main__":
    __main__()


#!/usr/bin/env python3

import logging

import argparse
import pathlib
import sys
#import tracemalloc

from workflow.run_tomo import Tomo

#from memory_profiler import profile
#@profile
def __main__():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Reduce tomography data')
    parser.add_argument('-i', '--input_file',
            required=True,
            type=pathlib.Path,
            help='''Full or relative path to the input file (in Nexus format).''')
    parser.add_argument('-o', '--output_file',
            required=False,
            type=pathlib.Path,
            help='''Full or relative path to the output file (in yaml format).''')
    parser.add_argument('--center_rows',
            nargs=2,
            type=int,
            help='''Center finding rows.''')
    parser.add_argument('--galaxy_flag',
            action='store_true',
            help='''Use this flag to run the scripts as a galaxy tool.''')
    parser.add_argument('-l', '--log',
#            type=argparse.FileType('w'),
            default=sys.stdout,
            help='Logging stream or filename')
    parser.add_argument('--log_level',
            choices=logging._nameToLevel.keys(),
            default='INFO',
            help='''Specify a preferred logging level.''')
    args = parser.parse_args()

    # Set log configuration
    # When logging to file, the stdout log level defaults to WARNING
    logging_format = '%(asctime)s : %(levelname)s - %(module)s : %(funcName)s - %(message)s'
    level = logging.getLevelName(args.log_level)
    if args.log is sys.stdout:
        logging.basicConfig(format=logging_format, level=level, force=True,
                handlers=[logging.StreamHandler()])
    else:
        if isinstance(args.log, str):
            logging.basicConfig(filename=f'{args.log}', filemode='w',
                    format=logging_format, level=level, force=True)
        elif isinstance(args.log, io.TextIOWrapper):
            logging.basicConfig(filemode='w', format=logging_format, level=level,
                    stream=args.log, force=True)
        else:
            raise(ValueError(f'Invalid argument --log: {args.log}'))
        stream_handler = logging.StreamHandler()
        logging.getLogger().addHandler(stream_handler)
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(logging.Formatter(logging_format))

    # Starting memory monitoring
#    tracemalloc.start()

    # Log command line arguments
    logging.info(f'input_file = {args.input_file}')
    logging.info(f'output_file = {args.output_file}')
    logging.info(f'center_rows = {args.center_rows}')
    logging.info(f'galaxy_flag = {args.galaxy_flag}')
    logging.debug(f'log = {args.log}')
    logging.debug(f'is log stdout? {args.log is sys.stdout}')
    logging.debug(f'log_level = {args.log_level}')

    # Instantiate Tomo object
    tomo = Tomo(galaxy_flag=args.galaxy_flag)

    # Read input file
    data = tomo.read(args.input_file)

    # Find the calibrated center axis info
    data = tomo.find_centers(data, center_rows=args.center_rows)

    # Write output file
    data = tomo.write(data, args.output_file)

    # Displaying memory usage
#    logging.info(f'Memory usage: {tracemalloc.get_traced_memory()}')
 
    # stopping memory monitoring
#    tracemalloc.stop()

if __name__ == "__main__":
    __main__()

#!/usr/bin/env python3

import logging
logging.getLogger(__name__)

import argparse
import pathlib
import sys

from .models import TomoWorkflow as Workflow
try:
    from deepdiff import DeepDiff
except:
    pass

parser = argparse.ArgumentParser(description='''Operate on representations of
        Tomo data workflows saved to files.''')
parser.add_argument('-l', '--log',
#        type=argparse.FileType('w'),
        default=sys.stdout,
        help='Logging stream or filename')
parser.add_argument('--log_level',
        choices=logging._nameToLevel.keys(),
        default='INFO',
        help='''Specify a preferred logging level.''')
subparsers = parser.add_subparsers(title='subcommands', required=True)#, dest='command')


# CONSTRUCT
def construct(args:list) -> None:
    if args.template_file is not None:
        wf = Workflow.construct_from_file(args.template_file)
        wf.cli()
    else:
        wf = Workflow.construct_from_cli()
    wf.write_to_file(args.output_file, force_overwrite=args.force_overwrite)

construct_parser = subparsers.add_parser('construct', help='''Construct a valid Tomo
        workflow representation on the command line and save it to a file. Optionally use
        an existing file as a template and/or preform the reconstruction or transfer to Galaxy.''')
construct_parser.set_defaults(func=construct)
construct_parser.add_argument('-t', '--template_file',
        type=pathlib.Path,
        required=False,
        help='''Full or relative template file path for the constructed workflow.''')
construct_parser.add_argument('-f', '--force_overwrite',
        action='store_true',
        help='''Use this flag to overwrite the output file if it already exists.''')
construct_parser.add_argument('-o', '--output_file',
        type=pathlib.Path,
        help='''Full or relative file path to which the constructed workflow will be written.''')


# VALIDATE
def validate(args:list) -> bool:
    try:
        wf = Workflow.construct_from_file(args.input_file)
        logger.info(f'Success: {args.input_file} represents a valid Tomo workflow configuration.')
        return(True)
    except BaseException as e:
        logger.error(f'{e.__class__.__name__}: {str(e)}')
        logger.info(f'''Failure: {args.input_file} does not represent a valid Tomo workflow
                configuration.''')
        return(False)

validate_parser = subparsers.add_parser('validate',
        help='''Validate a file as a representation of a Tomo workflow (this is most useful
                after a .yaml file has been manually edited).''')
validate_parser.set_defaults(func=validate)
validate_parser.add_argument('input_file',
        type=pathlib.Path,
        help='''Full or relative file path to validate as a Tomo workflow.''')


# CONVERT
def convert(args:list) -> None:
    wf = Workflow.construct_from_file(args.input_file)
    wf.write_to_file(args.output_file, force_overwrite=args.force_overwrite)

convert_parser = subparsers.add_parser('convert', help='''Convert one Tomo workflow
        representation to another. File format of both input and output files will be
        automatically determined from the files' extensions.''')
convert_parser.set_defaults(func=convert)
convert_parser.add_argument('-f', '--force_overwrite',
        action='store_true',
        help='''Use this flag to overwrite the output file if it already exists.''')
convert_parser.add_argument('-i', '--input_file',
        type=pathlib.Path,
        required=True,
        help='''Full or relative input file path to be converted.''')
convert_parser.add_argument('-o', '--output_file',
        type=pathlib.Path,
        required=True,
        help='''Full or relative file path to which the converted input will be written.''')


# DIFF / COMPARE
def diff(args:list) -> bool:
    raise ValueError('diff not tested')
#    wf1 = Workflow.construct_from_file(args.file1).dict_for_yaml()
#    wf2 = Workflow.construct_from_file(args.file2).dict_for_yaml()
#    diff = DeepDiff(wf1,wf2,
#                    ignore_order_func=lambda level:'independent_dimensions' not in level.path(),
#                    report_repetition=True,
#                    ignore_string_type_changes=True,
#                    ignore_numeric_type_changes=True)
    diff_report = diff.pretty()
    if len(diff_report) > 0:
        logger.info(f'The configurations in {args.file1} and {args.file2} are not identical.')
        print(diff_report)
        return(True)
    else:
        logger.info(f'The configurations in {args.file1} and {args.file2} are identical.')
        return(False)

diff_parser = subparsers.add_parser('diff', aliases=['compare'], help='''Print a comparison of 
        two Tomo workflow representations stored in files. The files may have different formats.''')
diff_parser.set_defaults(func=diff)
diff_parser.add_argument('file1',
        type=pathlib.Path,
        help='''Full or relative path to the first file for comparison.''')
diff_parser.add_argument('file2',
        type=pathlib.Path,
        help='''Full or relative path to the second file for comparison.''')


# LINK TO GALAXY
def link_to_galaxy(args:list) -> None:
    from .link_to_galaxy import link_to_galaxy
    link_to_galaxy(args.input_file, galaxy=args.galaxy, user=args.user,
            password=args.password, api_key=args.api_key)

link_parser = subparsers.add_parser('link_to_galaxy', help='''Construct a Galaxy history and link
        to an existing Tomo workflow representations in a NeXus file.''')
link_parser.set_defaults(func=link_to_galaxy)
link_parser.add_argument('-i', '--input_file',
        type=pathlib.Path,
        required=True,
        help='''Full or relative input file path to the existing Tomo workflow representations as 
                a NeXus file.''')
link_parser.add_argument('-g', '--galaxy',
        required=True,
        help='Target Galaxy instance URL/IP address')
link_parser.add_argument('-u', '--user',
        default=None,
        help='Galaxy user email address')
link_parser.add_argument('-p', '--password',
        default=None,
        help='Password for the Galaxy user')
link_parser.add_argument('-a', '--api_key',
        default=None,
        help='Galaxy admin user API key (required if not defined in the tools list file)')


# RUN THE RECONSTRUCTION
def run_tomo(args:list) -> None:
    from .run_tomo import run_tomo
    run_tomo(args.input_file, args.output_file, args.modes, center_file=args.center_file,
            num_core=args.num_core, output_folder=args.output_folder, save_figs=args.save_figs)

tomo_parser = subparsers.add_parser('run_tomo', help='''Construct and add reconstructed tomography
        data to an existing Tomo workflow representations in a NeXus file.''')
tomo_parser.set_defaults(func=run_tomo)
tomo_parser.add_argument('-i', '--input_file',
        required=True,
        type=pathlib.Path,
        help='''Full or relative input file path containing raw and/or reduced data.''')
tomo_parser.add_argument('-o', '--output_file',
        required=True,
        type=pathlib.Path,
        help='''Full or relative input file path containing raw and/or reduced data.''')
tomo_parser.add_argument('-c', '--center_file',
        type=pathlib.Path,
        help='''Full or relative input file path containing the rotation axis centers info.''')
#tomo_parser.add_argument('-f', '--force_overwrite',
#        action='store_true',
#        help='''Use this flag to overwrite any existing reduced data.''')
tomo_parser.add_argument('-n', '--num_core',
        type=int,
        default=-1,
        help='''Specify the number of processors to use.''')
tomo_parser.add_argument('--output_folder',
        type=pathlib.Path,
        default='.',
        help='Full or relative path to an output folder')
tomo_parser.add_argument('-s', '--save_figs',
        choices=['yes', 'no', 'only'],
        default='no',
        help='''Specify weather to display ('yes' or 'no'), save ('yes'), or only save ('only').''')
tomo_parser.add_argument('--reduce_data',
        dest='modes',
        const='reduce_data',
        action='append_const',
        help='''Use this flag to create and add reduced data to the input file.''')
tomo_parser.add_argument('--find_center',
        dest='modes',
        const='find_center',
        action='append_const',
        help='''Use this flag to find and add the calibrated center axis info to the input file.''')
tomo_parser.add_argument('--reconstruct_data',
        dest='modes',
        const='reconstruct_data',
        action='append_const',
        help='''Use this flag to create and add reconstructed data data to the input file.''')
tomo_parser.add_argument('--combine_data',
        dest='modes',
        const='combine_data',
        action='append_const',
        help='''Use this flag to combine reconstructed data data and add to the input file.''')


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])

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
            raise ValueError(f'Invalid argument --log: {args.log}')
        stream_handler = logging.StreamHandler()
        logging.getLogger().addHandler(stream_handler)
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(logging.Formatter(logging_format))

    args.func(args)

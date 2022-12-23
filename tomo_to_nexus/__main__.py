import argparse
import pathlib
import sys
import logging
from .models import TOMOWorkflow as Workflow
try:
    from deepdiff import DeepDiff
except:
    pass

parser = argparse.ArgumentParser(description='''Operate on representations of
        TOMO data workflows saved to files.''')
parser.add_argument('-l', '--log',
        choices=logging._nameToLevel.keys(),
        default='INFO',
        help='''Specify a preferred logging level.''')
subparsers = parser.add_subparsers(title='subcommands', required=True)#, dest='command')


# CONSTRUCT
def construct(args:list, logger=logging.getLogger(__name__)) -> None:
    if args.template_file is not None:
        wf = Workflow.construct_from_file(args.template_file, logger=logger)
        wf.cli()
    else:
        wf = Workflow.construct_from_cli()
    wf.write_to_file(args.output_file, force_overwrite=args.force_overwrite, logger=logger)

construct_parser = subparsers.add_parser('construct', help='''Construct a valid TOMO
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
def validate(args:list, logger=logging.getLogger(__name__)) -> bool:
    try:
        wf = Workflow.construct_from_file(args.input_file, logger=logger)
        logger.info(f'Success: {args.input_file} represents a valid TOMO workflow configuration.')
        return(True)
    except BaseException as e:
        logger.error(f'{e.__class__.__name__}: {str(e)}')
        logger.info(f'''Failure: {args.input_file} does not represent a valid TOMO workflow
                configuration.''')
        return(False)

validate_parser = subparsers.add_parser('validate',
        help='''Validate a file as a representation of a TOMO workflow (this is most useful
                after a .yaml file has been manually edited).''')
validate_parser.set_defaults(func=validate)
validate_parser.add_argument('input_file',
        type=pathlib.Path,
        help='''Full or relative file path to validate as a TOMO workflow.''')


# CONVERT
def convert(args:list, logger=logging.getLogger(__name__)) -> None:
    wf = Workflow.construct_from_file(args.input_file, logger=logger)
    wf.write_to_file(args.output_file, force_overwrite=args.force_overwrite, logger=logger)

convert_parser = subparsers.add_parser('convert', help='''Convert one TOMO workflow
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
def diff(args:list, logger=logging.getLogger(__name__)) -> bool:
    raise(ValueError('diff not tested'))
#    wf1 = Workflow.construct_from_file(args.file1, logger=logger).dict_for_yaml()
#    wf2 = Workflow.construct_from_file(args.file2, logger=logger).dict_for_yaml()
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
        two TOMO workflow representations stored in files. The files may have different formats.''')
diff_parser.set_defaults(func=diff)
diff_parser.add_argument('file1',
        type=pathlib.Path,
        help='''Full or relative path to the first file for comparison.''')
diff_parser.add_argument('file2',
        type=pathlib.Path,
        help='''Full or relative path to the second file for comparison.''')


# RUN THE RECONSTRUCTION
def run_tomo(args:list, logger=logging.getLogger(__name__)) -> None:
    from .run_tomo import run_tomo
    run_tomo(args.input_file, args.modes, force_overwrite=args.force_overwrite, logger=logger,
        num_core=args.num_core)

tomo_parser = subparsers.add_parser('run_tomo', help='''Construct and add reconstructed tomography
        data to an exising TOMO workflow representations in a NeXus file.''')
tomo_parser.set_defaults(func=run_tomo)
tomo_parser.add_argument('-i', '--input_file',
        type=pathlib.Path,
        required=True,
        help='''Full or relative input file path containing raw and/or reduced data.''')
tomo_parser.add_argument('-f', '--force_overwrite',
        action='store_true',
        help='''Use this flag to overwrite any existing reduced data.''')
tomo_parser.add_argument('-l', '--log',
        choices=logging._nameToLevel.keys(),
        default='INFO',
        help='''Specify a preferred logging level.''')
tomo_parser.add_argument('-n', '--num_core',
        type=int,
        default=-1,
        help='''Specify the number of processors to use.''')
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
tomo_parser.add_argument('--reconstruct_image',
        dest='modes',
        const='reconstruct_image',
        action='append_const',
        help='''Use this flag to create and add reconstructed image data to the input file.''')
tomo_parser.add_argument('--combine_images',
        dest='modes',
        const='combine_images',
        action='append_const',
        help='''Use this flag to combine reconstructed image data and add to the input file.''')


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])

    log_level = getattr(logging, args.log)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    args.func(args, logger=logger)


import yaml

import logging

import os
import numpy as np
try:
    import h5py
except:
    pass
from functools import cache
from pathlib import PosixPath
from pydantic import validator, ValidationError, conint, confloat, constr, \
        conlist, FilePath, PrivateAttr
from pydantic import BaseModel as PydanticBaseModel
from nexusformat.nexus import *
from time import time
from typing import Optional, Literal
from typing_extensions import TypedDict
from pyspec.file.spec import FileSpec

from msnctools.general import is_int, is_num, input_int, input_int_list, input_num, input_yesno, \
        input_menu, index_nearest, string_to_list, file_exists_and_readable, findImageFiles, \
        selectImageRange


def import_scanparser(station):
    if station in ('smb', 'fast', 'id1a3', 'id3a'):
        from scanparsers import SMBRotationScanParser
#        from msnctools.scanparsers import SMBRotationScanParser
        globals()['ScanParser'] = SMBRotationScanParser
    elif station in ('fmb', 'id3b'):
        from scanparsers import FMBRotationScanParser
#        from msnctools.scanparsers import FMBRotationScanParser
        globals()['ScanParser'] = FMBRotationScanParser
    else:
        raise(RuntimeError(f'Invalid station: {station}'))

@cache
def get_available_scan_numbers(spec_file:str):
    scans = FileSpec(spec_file).scans
    scan_numbers = list(scans.keys())
    return(scan_numbers)

@cache
def get_scanparser(spec_file:str, scan_number:int):
    if scan_number not in get_available_scan_numbers(spec_file):
        return(None)
    else:
        return(ScanParser(spec_file, scan_number))


class BaseModel(PydanticBaseModel):
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    @classmethod
    def construct_from_cli(cls):
        obj = cls.construct()
        obj.cli()
        return(obj)

    @classmethod
    def construct_from_yaml(cls, filename):
        try:
            with open(filename, 'r') as infile:
                indict = yaml.load(infile, Loader=yaml.CLoader)
        except:
            raise(ValueError(f'Could not load a dictionary from {filename}'))
        else:
            obj = cls(**indict)
            return(obj)

    @classmethod
    def construct_from_file(cls, filename, logger=logging.getLogger(__name__)):
        file_exists_and_readable(filename)
        filename = os.path.abspath(filename)
        fileformat = os.path.splitext(filename)[1]
        yaml_extensions = ('.yaml','.yml')
        nexus_extensions = ('.nxs','.nx5','.h5','.hdf5')
        t0 = time()
        if fileformat.lower() in yaml_extensions:
            obj = cls.construct_from_yaml(filename)
            logger.info(f'Constructed a model from {filename} in {time()-t0:.2f} seconds.')
            return(obj)
        elif fileformat.lower() in nexus_extensions:
            obj = cls.construct_from_nexus(filename)
            logger.info(f'Constructed a model from {filename} in {time()-t0:.2f} seconds.')
            return(obj)
        else:
            logger.error(f'Unsupported file extension for constructing a model: {fileformat}')
            raise(TypeError(f'Unrecognized file extension: {fileformat}'))

    def dict_for_yaml(self, exclude_fields=[]):
        yaml_dict = {}
        for field_name in self.__fields__:
            if field_name in exclude_fields:
                continue
            else:
                field_value = getattr(self, field_name, None)
                if field_value is not None:
                    if isinstance(field_value, BaseModel):
                        yaml_dict[field_name] = field_value.dict_for_yaml()
                    elif isinstance(field_value,list) and all(isinstance(item,BaseModel)
                            for item in field_value):
                        yaml_dict[field_name] = [item.dict_for_yaml() for item in field_value]
                    elif isinstance(field_value, PosixPath):
                        yaml_dict[field_name] = str(field_value)
                    else:
                        yaml_dict[field_name] = field_value
                else:
                    continue
        return(yaml_dict)

    def write_to_yaml(self, filename=None, logger=logging.getLogger(__name__)):
        yaml_dict = self.dict_for_yaml()
        if filename is None:
            logger.info('Printing yaml representation here:\n'+
                    f'{yaml.dump(yaml_dict, sort_keys=False)}')
        else:
            try:
                with open(filename, 'w') as outfile:
                    yaml.dump(yaml_dict, outfile, sort_keys=False)
                logger.info(f'Sucessfully wrote this model to {filename}')
            except:
                logger.error(f'Unknown error -- could not write to {filename} in yaml format.')
                logger.info('Printing yaml representation here:\n'+
                        f'{yaml.dump(yaml_dict, sort_keys=False)}')

    def write_to_file(self, filename, force_overwrite=False, logger=logging.getLogger(__name__)):
        file_writeable, fileformat = self.output_file_valid(filename,
                force_overwrite=force_overwrite, logger=logger)
        if fileformat == 'yaml':
            if file_writeable:
                self.write_to_yaml(filename=filename, logger=logger)
            else:
                self.write_to_yaml(logger=logger)
        elif fileformat == 'nexus':
            if file_writeable:
                self.write_to_nexus(filename=filename, logger=logger)

    def output_file_valid(self, filename, force_overwrite=False,
            logger=logging.getLogger(__name__)):
        filename = os.path.abspath(filename)
        fileformat = os.path.splitext(filename)[1]
        yaml_extensions = ('.yaml','.yml')
        nexus_extensions = ('.nxs','.nx5','.h5','.hdf5')
        if fileformat.lower() not in (*yaml_extensions, *nexus_extensions):
            return(False, None) # Only yaml and NeXus files allowed for output now.
        elif fileformat.lower() in yaml_extensions:
            fileformat = 'yaml'
        elif fileformat.lower() in nexus_extensions:
            fileformat = 'nexus'
        if os.path.isfile(filename):
            if os.access(filename, os.W_OK):
                if not force_overwrite:
                    logger.error(f'{filename} will not be overwritten.')
                    return(False, fileformat)
            else:
                logger.error(f'Cannot access {filename} for writing.')
                return(False, fileformat)
        if os.path.isdir(os.path.dirname(filename)):
            if os.access(os.path.dirname(filename), os.W_OK):
                return(True, fileformat)
            else:
                logger.error(f'Cannot access {os.path.dirname(filename)} for writing.')
                return(False, fileformat)
        else:
            try:
                os.makedirs(os.path.dirname(filename))
                return(True, fileformat)
            except:
                logger.error(f'Cannot create {os.path.dirname(filename)} for output.')
                return(False, fileformat)

    def set_single_attr_cli(self, attr_name, attr_desc='unknown attribute', list_flag=False,
            **cli_kwargs):
        if cli_kwargs.get('chain_attr_desc', False):
            cli_kwargs['attr_desc'] = attr_desc
        try:
            attr = getattr(self, attr_name, None)
            if attr is None:
                attr = self.__fields__[attr_name].type_.construct()
            if cli_kwargs.get('chain_attr_desc', False):
                cli_kwargs['attr_desc'] = attr_desc
            input_accepted = False
            while not input_accepted:
                try:
                    attr.cli(**cli_kwargs)
                except ValidationError as e:
                    print(e)
                    print(f'Removing {attr_desc} configuration')
                    attr = self.__fields__[attr_name].type_.construct()
                    continue
                except KeyboardInterrupt as e:
                    raise(e)
                except BaseException as e:
                    print(f'{type(e).__name__}: {e}')
                    print(f'Removing {attr_desc} configuration')
                    attr = self.__fields__[attr_name].type_.construct()
                    continue
                try:
                    setattr(self, attr_name, attr)
                except ValidationError as e:
                    print(e)
                except KeyboardInterrupt as e:
                    raise(e)
                except BaseException as e:
                    print(f'{type(e).__name__}: {e}')
                else:
                    input_accepted = True
        except:
            input_accepted = False
            while not input_accepted:
                attr = getattr(self, attr_name, None)
                if attr is None:
                    input_value = input(f'Type and enter a value for {attr_desc}: ')
                else:
                    input_value = input(f'Type and enter a new value for {attr_desc} or press '+
                            f'enter to keep the current one ({attr}): ')
                if list_flag:
                    input_value = string_to_list(input_value, remove_duplicates=False, sort=False)
                if len(input_value) == 0:
                    input_value = getattr(self, attr_name, None)
                try:
                    setattr(self, attr_name, input_value)
                except ValidationError as e:
                    print(e)
                except KeyboardInterrupt as e:
                    raise(e)
                except BaseException as e:
                    print(f'Unexpected {type(e).__name__}: {e}')
                else:
                    input_accepted = True

    def set_list_attr_cli(self, attr_name, attr_desc='unknown attribute', **cli_kwargs):
        if cli_kwargs.get('chain_attr_desc', False):
            cli_kwargs['attr_desc'] = attr_desc
        attr = getattr(self, attr_name, None)
        if attr is not None:
            # Check existing items
            for item in attr:
                item_accepted = False
                while not item_accepted:
                    item.cli(**cli_kwargs)
                    try:
                        setattr(self, attr_name, attr)
                    except ValidationError as e:
                        print(e)
                    except KeyboardInterrupt as e:
                        raise(e)
                    except BaseException as e:
                        print(f'{type(e).__name__}: {e}')
                    else:
                        item_accepted = True
        else:
            # Initialize list for new attribute & starting item
            attr = []
            item = self.__fields__[attr_name].type_.construct()
        # Append (optional) additional items
        append = input_yesno(f'Add a {attr_desc} configuration? (y/n)', 'n')
        while append:
            attr.append(item.__class__.construct_from_cli())
            try:
                setattr(self, attr_name, attr)
            except ValidationError as e:
                print(e)
                print(f'Removing last {attr_desc} configuration from the list')
                attr.pop()
            except KeyboardInterrupt as e:
                raise(e)
            except BaseException as e:
                print(f'{type(e).__name__}: {e}')
                print(f'Removing last {attr_desc} configuration from the list')
                attr.pop()
            else:
                append = input_yesno(f'Add another {attr_desc} configuration? (y/n)', 'n')


class Detector(BaseModel):
    prefix: constr(strip_whitespace=True, min_length=1)
    rows: conint(gt=0)
    columns: conint(gt=0)
    pixel_size: conlist(item_type=confloat(gt=0), min_items=1, max_items=2)
    lens_magnification: confloat(gt=0) = 1.0

    @property
    def get_pixel_size(self):
        return(list(np.asarray(self.pixel_size)/self.lens_magnification))

    def construct_from_yaml(self, filename):
        try:
            with open(filename, 'r') as infile:
                indict = yaml.load(infile, Loader=yaml.CLoader)
            detector = indict['detector']
            self.prefix = detector['id']
            pixels = detector['pixels']
            self.rows = pixels['rows']
            self.columns = pixels['columns']
            self.pixel_size = pixels['size']
            self.lens_magnification = indict['lens_magnification']
        except:
            logging.warning(f'Could not load a dictionary from {filename}')
            return(False)
        else:
            return(True)

    def cli(self):
        print('\n -- Configure the detector -- ')
        self.set_single_attr_cli('prefix', 'detector ID')
        self.set_single_attr_cli('rows', 'number of pixel rows')
        self.set_single_attr_cli('columns', 'number of pixel columns')
        self.set_single_attr_cli('pixel_size', 'pixel size in mm (enter either a single value for '+
                'square pixels or a pair of values for the size in the respective row and column '+
                'directions)', list_flag=True)
        self.set_single_attr_cli('lens_magnification', 'lens magnification')

    def construct_nxdetector(self):
        nxdetector = NXdetector()
        nxdetector.local_name = self.prefix
        pixel_size = self.get_pixel_size
        if len(pixel_size) == 1:
            nxdetector.x_pixel_size = pixel_size[0]
            nxdetector.y_pixel_size = pixel_size[0]
        else:
            nxdetector.x_pixel_size = pixel_size[0]
            nxdetector.y_pixel_size = pixel_size[1]
        nxdetector.x_pixel_size.attrs['units'] = 'mm'
        nxdetector.y_pixel_size.attrs['units'] = 'mm'
        return(nxdetector)


class ScanInfo(TypedDict):
    scan_number: int
    starting_image_offset: conint(ge=0)
    num_image: conint(gt=0)
    ref_height: float


class SpecScans(BaseModel):
    spec_file: FilePath
    scan_numbers: conlist(item_type=conint(gt=0), min_items=1)
    stack_info: conlist(item_type=ScanInfo, min_items=1) = []

    @validator('spec_file')
    def validate_spec_file(cls, spec_file):
        try:
            spec_file = os.path.abspath(spec_file)
            sspec_file = FileSpec(spec_file)
        except:
            raise(ValueError(f'Invalid SPEC file {spec_file}'))
        else:
            return(spec_file)

    @validator('scan_numbers')
    def validate_scan_numbers(cls, scan_numbers, values):
        spec_file = values.get('spec_file')
        if spec_file is not None:
            spec_scans = FileSpec(spec_file)
            for scan_number in scan_numbers:
                scan = spec_scans.get_scan_by_number(scan_number)
                if scan is None:
                    raise(ValueError(f'There is no scan number {scan_number} in {spec_file}'))
        return(scan_numbers)

    @validator('stack_info')
    def validate_stack_info(cls, stack_info, values):
        scan_numbers = values.get('scan_numbers')
        assert(len(scan_numbers) == len(stack_info))
        for scan_info in stack_info:
            assert(scan_info['scan_number'] in scan_numbers)
            is_int(scan_info['starting_image_offset'], ge=0, lt=scan_info['num_image'],
                    raise_error=True)
        return(stack_info)

    @classmethod
    def construct_from_nxcollection(cls, nxcollection:NXcollection):
        config = {}
        config['spec_file'] = nxcollection.attrs['spec_file']
        scan_numbers = []
        stack_info = []
        for nxsubentry_name, nxsubentry in nxcollection.items():
            scan_number = int(nxsubentry_name.split('_')[-1])
            scan_numbers.append(scan_number)
            stack_info.append({'scan_number': scan_number,
                    'starting_image_offset': int(nxsubentry.instrument.detector.frame_start_number),
                    'num_image': len(nxsubentry.independent_dimensions.rotation_angle),
                    'ref_height': float(nxsubentry.sample.z_translation)})
        config['scan_numbers']= sorted(scan_numbers)
        config['stack_info']= stack_info
        return(cls(**config))

    @property
    def available_scan_numbers(self):
        return(get_available_scan_numbers(self.spec_file))

    def get_scan_index(self, scan_number):
        scan_index = [scan_index for scan_index, scan_info in enumerate(self.stack_info)
                if scan_info['scan_number'] == scan_number]
        if len(scan_index) > 1:
            raise(ValueError('Duplicate scan_numbers in image stack'))
        elif len(scan_index) == 1:
            return(scan_index[0])
        else:
            return(None)

    def get_detector_data(self, detector_prefix, scan_number, scan_step_index):
        parser = get_scanparser(self.spec_file, scan_number)
        return(parser.get_detector_data(detector_prefix, scan_step_index))

    def get_scanparser(self, scan_number):
        return(get_scanparser(self.spec_file, scan_number))

    def scan_numbers_cli(self, attr_desc):
        if len(self.available_scan_numbers) == 1:
            input_mode = 1
        else:
            if hasattr(self, 'scan_numbers'):
                print(f'Currently selected {attr_desc}scan numbers are: {self.scan_numbers}')
                menu_options = [f'Select a subset of the available {attr_desc}scan numbers',
                                f'Use all available {attr_desc}scan numbers in {self.spec_file}',
                                f'Keep the currently selected {attr_desc}scan numbers']
            else:
                menu_options = [f'Select a subset of the available {attr_desc}scan numbers',
                                f'Use all available {attr_desc}scan numbers in {self.spec_file}']
            print(f'Available scan numbers in {self.spec_file} are: {self.available_scan_numbers}')
            input_mode = input_menu(menu_options, header='Choose one of the following options '+
                    'for selecting scan numbers')
        if input_mode == 0:
            accept_scan_numbers = False
            while not accept_scan_numbers:
                try:
                    self.scan_numbers = input_int_list(f'Enter a series of {attr_desc}scan numbers')
                except ValidationError as e:
                    print(e)
                except KeyboardInterrupt as e:
                    raise(e)
                except BaseException as e:
                    print(f'Unexpected {type(e).__name__}: {e}')
                else:
                    accept_scan_numbers = True
        elif input_mode == 1:
            self.scan_numbers = self.available_scan_numbers
        elif input_mode == 2:
            pass

    def cli(self, **cli_kwargs):
        if cli_kwargs.get('attr_desc') is not None:
            attr_desc = f'{cli_kwargs["attr_desc"]} '
        else:
            attr_desc = ''
        print(f'\n -- Configure which scans to use from a single {attr_desc}SPEC file')
        self.set_single_attr_cli('spec_file', attr_desc+'SPEC file path')
        self.scan_numbers_cli(attr_desc)

    def construct_nxcollection(self, field_name, thetas, detector):
        nxcollection = NXcollection()
        nxcollection.attrs['spec_file'] = str(self.spec_file)
        parser = self.get_scanparser(self.scan_numbers[0])
        nxcollection.attrs['date'] = parser.spec_scan.file_date
        for scan_number in self.scan_numbers:
            # Get scan info
            scan_info = self.stack_info[self.get_scan_index(scan_number)]
            # Add an NXsubentry to the NXcollection for each scan
            entry_name = f'scan_{scan_number}'
            nxsubentry = NXsubentry()
            nxcollection[entry_name] = nxsubentry
            parser = self.get_scanparser(scan_number)
            nxsubentry.start_time = parser.spec_scan.date
            nxsubentry.spec_command = parser.spec_command
            # Add an NXdata for independent dimensions to the scan's NXsubentry
            num_image = scan_info['num_image']
            if thetas is None:
                thetas = num_image*[0.0]
            else:
                assert(num_image == len(thetas))
            nxsubentry.independent_dimensions = NXdata()
            nxsubentry.independent_dimensions.rotation_angle = thetas
            nxsubentry.independent_dimensions.rotation_angle.units = 'degrees'
            # Add an NXinstrument to the scan's NXsubentry
            nxsubentry.instrument = NXinstrument()
            # Add an NXdetector to the NXinstrument to the scan's NXsubentry
            nxsubentry.instrument.detector = detector.construct_nxdetector()
            nxsubentry.instrument.detector.frame_start_number = scan_info['starting_image_offset']
            nxsubentry.instrument.detector.image_key = field_name
            # Add an NXsample to the scan's NXsubentry
            nxsubentry.sample = NXsample()
            nxsubentry.sample.z_translation = scan_info['ref_height']
            nxsubentry.sample.z_translation.units = 'mm'
        return(nxcollection)


class FlatField(SpecScans):

    def image_range_cli(self, attr_desc, detector_prefix):
        stack_info = self.stack_info
        for scan_number in self.scan_numbers:
            # Parse the available image range
            parser = self.get_scanparser(scan_number)
            image_offset = parser.starting_image_offset
            num_image = parser.get_num_image(detector_prefix.upper())
            scan_index = self.get_scan_index(scan_number)

            # Select the image set
            print(f'Available good image set index range: [{image_offset}, {num_image-1}]')
            image_set_approved = False
            if scan_index is not None:
                scan_info = stack_info[scan_index]
                print(f'Current starting image offset and number of images: '+
                        f'{scan_info["starting_offset_index"]} and {scan_info["num_image"]}')
                image_set_approved = input_yesno(f'Accept these values (y/n)?', 'y')
            if not image_set_approved:
                while not image_set_approved:
                    offset = input_int(f'Enter the starting image offset', ge=image_offset,
                            le=num_image-1, default=image_offset)
                    num = input_int(f'Enter the number of images', ge=1, le=num_image-offset,
                            default=num_image-offset)
                    print(f'Current starting image offset and number of images: {offset} and {num}')
                    image_set_approved = input_yesno(f'Accept these values (y/n)?', 'y')
                if scan_index is not None:
                    scan_info['starting_offset_index'] = offset
                    scan_info['num_image'] = num
                    scan_info['ref_height'] = parser.vertical_shift
                else:
                    stack_info.append({'scan_number': scan_number, 'starting_offset_index': offset,
                            'num_image': num, 'ref_height': parser.vertical_shift})
        self.stack_info = stack_info

    def cli(self, **cli_kwargs):
        if cli_kwargs.get('attr_desc') is not None:
            attr_desc = f'{cli_kwargs["attr_desc"]} '
        else:
            attr_desc = ''
        detector = cli_kwargs.get('detector')
        print(f'\n -- Configure the location of the {attr_desc}scan data -- ')
        self.set_single_attr_cli('spec_file', attr_desc+'SPEC file path')
        self.scan_numbers_cli(attr_desc)
        self.image_range_cli(attr_desc, detector.prefix)


class TomoField(SpecScans):
    theta_range: dict = {}

    @validator('theta_range')
    def validate_theta_range(cls, theta_range):
        if len(theta_range) != 4:
            raise(ValueError(f'Invalid theta range {theta_range}'))
        is_num(theta_range['start'], raise_error=True)
        is_num(theta_range['end'], raise_error=True)
        is_int(theta_range['num'], raise_error=True)
        is_int(theta_range['start_index'], raise_error=True)
        if (theta_range['end'] <= theta_range['start'] or theta_range['num'] <= 0 or 
                theta_range['start_index'] < 0):
            raise(ValueError(f'Invalid theta range {theta_range}'))
        return(theta_range)

    def theta_range_cli(self, scan_number, attr_desc, station):
        # Parse the available theta range
        parser = self.get_scanparser(scan_number)
        theta_vals = parser.theta_vals
        spec_theta_start = theta_vals.get('start')
        spec_theta_end = theta_vals.get('end')
        spec_num_theta = theta_vals.get('num')
        thetas = np.linspace(spec_theta_start, spec_theta_end, spec_num_theta)
        delta_theta = thetas[1]-thetas[0]

        # Check for consistency of theta ranges between scans
        if scan_number != self.scan_numbers[0]:
            parser = self.get_scanparser(scan_number)
            if (parser.theta_vals.get('start') != spec_theta_start or
                    parser.theta_vals.get('end') != spec_theta_end or
                    parser.theta_vals.get('num') != spec_num_theta):
                raise(ValueError(f'Incompatible theta ranges between {attr_desc}scans:'+
                        f'\n\tScan {self.scan_numbers[0]}: {theta_vals}'+
                        f'\n\tScan {scan_number}: {parser.theta_vals}'))
            return

        # Select the theta range for the tomo reconstruction from the first scan
        theta_start = self.theta_range.get('start', None)
        theta_end = self.theta_range.get('end', None)
        num_theta = self.theta_range.get('num', None)
        theta_index_start = self.theta_range.get('start_index', None)
        if (theta_start is not None and theta_end is not None and num_theta is not None and
                theta_index_start is not None):
            if len(self.theta_range) != 4:
                logging.warning(f'Illegal value for theta range {self.theta_range}')
                self.theta_range = {}
            if theta_start < spec_theta_start or theta_start not in thetas:
                logging.warning('theta start value is incompatible with SPEC file '+
                        f'({theta_start} < {spec_theta_start}): ignore input theta range')
                self.theta_range = {}
            if theta_end > spec_theta_end or theta_end not in thetas:
                logging.warning('theta end value is incompatible with SPEC file '+
                        f'({theta_end} > {spec_theta_end}): ignore input theta range')
                self.theta_range = {}
            if num_theta > thetas.size or num_theta != int((theta_end-theta_start)/delta_theta):
                logging.warning('number of theta value is incompatible with SPEC file '+
                        f'({num_theta} > {thetas.size}): ignore input theta range')
                self.theta_range = {}
        elif len(self.theta_range) and (len(self.theta_range) != 4 or theta_start is not None or
                theta_end is not None or num_theta is not None or theta_index_start is not None):
            logging.warning(f'Illegal value for theta range {self.theta_range}')
            self.theta_range = {}
        theta_range_approved = False
        if len(self.theta_range) == 4:
            print(f'Current theta range: [{theta_start}, {theta_start+delta_theta}, ..., '+
                    f'{theta_end})')
            print(f'Number of theta values: {num_theta}')
            theta_range_approved = input_yesno(f'Accept this theta range (y/n)?', 'y')
        if theta_range_approved:
            theta_index_start = index_nearest(thetas, theta_start)
            assert(theta_start == thetas[theta_index_start])
            theta_index_end = index_nearest(thetas, theta_end)
            assert(theta_end == thetas[theta_index_end])
        else:
            print(f'Theta range obtained from SPEC data: [{spec_theta_start}, {spec_theta_end})')
            print(f'Theta step size = {delta_theta}')
            print(f'Number of theta values: {spec_num_theta-1}')
            default_start = None
            default_end = None
            if station in ('smb', 'fast', 'id1a3', 'id3a'):
                theta_range_approved = input_yesno(f'Accept this theta range (y/n)?', 'y')
                if theta_range_approved:
                    theta_start = spec_theta_start
                    theta_end = spec_theta_end
                    num_theta = spec_num_theta-1
                    theta_index_start = 0
            elif station in ('fmb', 'id3b'):
                if spec_theta_start <= 0.0 and spec_theta_end >= 180.0:
                    default_start = 0
                    default_end = 180
                elif spec_theta_end-spec_theta_start == 180:
                    default_start = spec_theta_start
                    default_end = spec_theta_end
            while not theta_range_approved:
                theta_start = input_num(f'Enter the first theta (included)', ge=spec_theta_start,
                        lt=spec_theta_end, default=default_start)
                theta_index_start = index_nearest(thetas, theta_start)
                theta_start = thetas[theta_index_start]
                theta_end = input_num(f'Enter the last theta (excluded)',
                        ge=theta_start+delta_theta, le=spec_theta_end, default=default_end)
                theta_index_end = index_nearest(thetas, theta_end)
                theta_end = thetas[theta_index_end]
                num_theta = theta_index_end-theta_index_start
                print(f'Selected theta range: [{theta_start}, {theta_start+delta_theta}, ..., '+
                        f'{theta_end})')
                print(f'Number of theta values: {num_theta}')
                theta_range_approved = input_yesno(f'Accept this theta range (y/n)?', 'y')
            self.theta_range = {'start': float(theta_start), 'end': float(theta_end),
                    'num': num_theta, 'start_index': theta_index_start}

    def image_range_cli(self, attr_desc, detector_prefix):
        stack_info = self.stack_info
        for scan_number in self.scan_numbers:
            # Parse the available image range
            parser = self.get_scanparser(scan_number)
            image_offset = parser.starting_image_offset
            num_image = parser.get_num_image(detector_prefix.upper())
            scan_index = self.get_scan_index(scan_number)

            # Select the image set matching the theta range
            num_theta = self.theta_range['num']
            theta_index_start = self.theta_range['start_index']
            if num_theta > num_image-theta_index_start:
                raise(ValueError(f'Available {attr_desc}image indices incompatible with thetas:'+
                        f'\n\tNumber of thetas and offset = {num_theta} and {theta_index_start}'+
                        f'\n\tNumber of available images {num_image}'))
            if scan_index is not None:
                scan_info = stack_info[scan_index]
                scan_info['starting_offset_index'] = image_offset+theta_index_start
                scan_info['num_image'] = num_theta
                scan_info['ref_height'] = parser.vertical_shift
            else:
                stack_info.append({'scan_number': scan_number,
                        'starting_offset_index': image_offset+theta_index_start,
                        'num_image': num_theta, 'ref_height': parser.vertical_shift})
        self.stack_info = stack_info

    def cli(self, **cli_kwargs):
        if cli_kwargs.get('attr_desc') is not None:
            attr_desc = f'{cli_kwargs["attr_desc"]} '
        else:
            attr_desc = ''
        station = cli_kwargs.get('station')
        detector = cli_kwargs.get('detector')
        print(f'\n -- Configure the location of the {attr_desc}scan data -- ')
        self.set_single_attr_cli('spec_file', attr_desc+'SPEC file path')
        self.scan_numbers_cli(attr_desc)
        for scan_number in self.scan_numbers:
            self.theta_range_cli(scan_number, attr_desc, station)
        self.image_range_cli(attr_desc, detector.prefix)


class Sample(BaseModel):
    name: constr(min_length=1)
    description: Optional[str]
    rotation_angles: Optional[list]
    z_translations: Optional[list]

    @classmethod
    def construct_from_nxsample(cls, nxsample:NXsample):
        config = {}
        config['name'] = nxsample.name.nxdata
        if 'description' in nxsample:
            config['description'] = nxsample.description.nxdata
        if 'rotation_angle' in nxsample:
            config['rotation_angle'] = nxsample.rotation_angle.nxdata
        if 'z_translation' in nxsample:
            config['z_translation'] = nxsample.z_translation.nxdata
        return(cls(**config))

    def cli(self):
        print('\n -- Configure the sample metadata -- ')
        self.set_single_attr_cli('name', 'the sample name')
        self.set_single_attr_cli('description', 'a description of the sample (optional)')


class MapConfig(BaseModel):
    title: constr(strip_whitespace=True, min_length=1)
    station: Literal['smb', 'fmb', 'fast', 'id1a3', 'id3a', 'id3b'] = None
    sample: Sample
    detector: Detector = Detector.construct()
    dark_field: Optional[FlatField]
    bright_field: FlatField
    tomo_fields: TomoField
    _thetas: list[float] = PrivateAttr()
    _field_types = ({'name': 'dark_field', 'image_key': 2}, {'name': 'bright_field',
            'image_key': 1}, {'name': 'tomo_fields', 'image_key': 0})

    @classmethod
    def construct_from_nxentry(cls, nxentry:NXentry):
        config = {}
        config['title'] = nxentry.nxname
        config['station'] = nxentry.instrument.source.attrs['station']
        config['sample'] = Sample.construct_from_nxsample(nxentry['sample'])
        for nxobject_name, nxobject in nxentry.spec_scans.items():
            if isinstance(nxobject, NXcollection):
                config[nxobject_name] = SpecScans.construct_from_nxcollection(nxobject)
        return(cls(**config))

#FIX cache?
    @property
    def thetas(self):
        try:
            return(self._thetas)
        except:
            theta_range = self.tomo_fields.theta_range
            self._thetas = list(np.linspace(theta_range['start'], theta_range['end'],
                    theta_range['num']))
            return(self._thetas)

    def cli(self):
        print('\n -- Configure a map from a set of SPEC scans (dark, bright, and tomo), '+
                'and / or detector data -- ')
        self.set_single_attr_cli('title', 'title for the map entry')
        self.set_single_attr_cli('station', 'name of the station at which scans were collected '+
                '(examples: id1a3, id3a, id3b, smb, or fmb)')
        import_scanparser(self.station)
        self.set_single_attr_cli('sample')
        use_detector_config = False
        if hasattr(self.detector, 'prefix') and len(self.detector.prefix):
            use_detector_config = input_yesno(f'Current detector settings:\n{self.detector}\n'+
                    f'Keep these settings? (y/n)')
        if not use_detector_config:
            have_detector_config = input_yesno(f'Is a detector configuration file available? (y/n)')
            if have_detector_config:
                detector_config_file = input(f'Enter detector configuration file name: ')
                have_detector_config = self.detector.construct_from_yaml(detector_config_file)
            if not have_detector_config:
                self.set_single_attr_cli('detector', 'detector')
        have_dark_field = input_yesno(f'Are Dark field images available? (y/n)')
        if have_dark_field:
            self.set_single_attr_cli('dark_field', 'Dark field', chain_attr_desc=True,
                    station=self.station, detector=self.detector)
        self.set_single_attr_cli('bright_field', 'Bright field', chain_attr_desc=True,
                station=self.station, detector=self.detector)
        self.set_single_attr_cli('tomo_fields', 'Tomo field', chain_attr_desc=True,
                station=self.station, detector=self.detector)

    def construct_nxentry(self, nxroot, logger=logging.getLogger(__name__)):
        # Construct base NXentry
        nxentry = NXentry()

        # Add an NXentry to the NXroot
        nxroot[self.title] = nxentry
        nxroot.attrs['default'] = self.title
        nxentry.definition = 'NXtomo'
#        nxentry.attrs['default'] = 'data'

        # Add an NXinstrument to the NXentry
        nxinstrument = NXinstrument()
        nxentry.instrument = nxinstrument

        # Add an NXsource to the NXinstrument
        nxsource = NXsource()
        nxinstrument.source = nxsource
        nxsource.type = 'Synchrotron X-ray Source'
        nxsource.name = 'CHESS'
        nxsource.probe = 'x-ray'
 
        # Tag the NXsource with the station (as an attribute)
        nxsource.attrs['station'] = self.station

        # Add an NXdetector to the NXinstrument (don't fill in data fields yet)
        nxinstrument.detector = self.detector.construct_nxdetector()

        # Add an NXsample to NXentry (don't fill in data fields yet)
        nxsample = NXsample()
        nxentry.sample = nxsample
        nxsample.name = self.sample.name
        nxsample.description = self.sample.description

        # Add an NXcollection to the base NXentry to hold metadata about the spec scans in the map
        # Also obtain the data fields in NXsample and NXdetector
        nxspec_scans = NXcollection()
        nxentry.spec_scans = nxspec_scans
        image_keys = []
        sequence_numbers = []
        image_stacks = []
        rotation_angles = []
        z_translations = []
        for field_type in self._field_types:
            field_name = field_type['name']
            field = getattr(self, field_name)
            if field is None:
                continue
            image_key = field_type['image_key']
            if field_type['name'] == 'tomo_fields':
                thetas = self.thetas
            else:
                thetas = None
            # Add the scans in a single spec file
            parser = field.get_scanparser(field.scan_numbers[0])
            nxspec_scans[field_name] = field.construct_nxcollection(image_key, thetas,
                    self.detector)
            for scan_number in field.scan_numbers:
                parser = field.get_scanparser(scan_number)
                scan_info = field.stack_info[field.get_scan_index(scan_number)]
                image_offset = scan_info['starting_image_offset']
                num_image = scan_info['num_image']
                image_keys += num_image*[image_key]
                sequence_numbers += [i for i in range(num_image)]
                for scan_step_index in range(image_offset, image_offset+num_image):
                    image_stacks.append(field.get_detector_data(self.detector.prefix, scan_number,
                            scan_step_index))
                if thetas is None:
                    rotation_angles += scan_info['num_image']*[0.0]
                else:
                    assert(num_image == len(thetas))
                    rotation_angles += thetas
                z_translations += scan_info['num_image']*[scan_info['ref_height']]
        # Add image data to NXdetector
        nxinstrument.detector.image_key = image_keys
        nxinstrument.detector.sequence_number = sequence_numbers
        nxinstrument.detector.data = np.stack([image for image in image_stacks])
        # Add image data to NXsample
        nxsample.rotation_angle = rotation_angles
        nxsample.rotation_angle.attrs['units'] = 'degrees'
        nxsample.z_translation = z_translations
        nxsample.z_translation.attrs['units'] = 'mm'

        # Add an NXdata to NXentry
        nxdata = NXdata()
        nxentry.data = nxdata
        nxdata.makelink(nxentry.instrument.detector.data, name='data')
        nxdata.makelink(nxentry.instrument.detector.image_key)
        nxdata.makelink(nxentry.sample.rotation_angle)
        nxdata.makelink(nxentry.sample.z_translation)
#        nxdata.attrs['axes'] = ['field', 'row', 'column']
#        nxdata.attrs['field_indices'] = 0
#        nxdata.attrs['row_indices'] = 1
#        nxdata.attrs['column_indices'] = 2


class TOMOWorkflow(BaseModel):
    sample_maps: conlist(item_type=MapConfig, min_items=1) = [MapConfig.construct()]

    @classmethod
    def construct_from_nexus(cls, filename):
        nxroot = nxload(filename)
        sample_maps = []
        config = {'sample_maps':sample_maps}
        for nxentry_name, nxentry in nxroot.items():
            sample_maps.append(MapConfig.construct_from_nxentry(nxentry))
        return(cls(**config))

    def cli(self):
        print('\n -- Configure a map -- ')
        self.set_list_attr_cli('sample_maps', 'sample map')

    def construct_nxfile(self, filename, mode='w-', logger=logging.getLogger(__name__)):
        nxroot = NXroot()
        for sample_map in self.sample_maps:
            import_scanparser(sample_map.station)
            sample_map.construct_nxentry(nxroot, logger=logger)
        nxroot.save(filename, mode=mode)

    def write_to_nexus(self, filename, logger=logging.getLogger(__name__)):
        t0 = time()
        self.construct_nxfile(filename, mode='w', logger=logger)
        logger.info(f'Saved all sample maps to {filename} in {time()-t0:.2f} seconds.')


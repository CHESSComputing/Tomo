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
from pyspec.file.spec import FileSpec

from msnctools.general import is_int, is_num, input_int, input_int_list, input_num, input_yesno, \
        input_menu, index_nearest, string_to_list, file_exists_and_readable, findImageFiles, \
        selectImageRange


def import_scanparser(station):
    if station in ('smb', 'fast', 'id1a3', 'id3a'):
        from msnctools.scanparsers import SMBRotationScanParser
        globals()['ScanParser'] = SMBRotationScanParser
    elif station in ('fmb', 'id3b'):
        from msnctools.scanparsers import FMBRotationScanParser
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
        try:
            attr = getattr(self, attr_name)
            if cli_kwargs.get('chain_attr_desc', False):
                cli_kwargs['attr_desc'] = attr_desc
            attr.cli(**cli_kwargs)
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
    id: constr(strip_whitespace=True, min_length=1)
    rows: conint(gt=0)
    columns: conint(gt=0)
    pixel_size: conlist(item_type=confloat(gt=0), min_items=1, max_items=2)
    lens_magnification: confloat(gt=0) = 1.0

    def get_pixel_size(self):
        if hasattr(self, 'pixel_size'):
            return(self.pixel_size/self.lens_magnification)
        else:
            return(None)

    def construct_from_yaml(self, filename):
        file_exists_and_readable(filename)
        try:
            with open(filename, 'r') as infile:
                indict = yaml.load(infile, Loader=yaml.CLoader)
        except:
            raise(ValueError(f'Could not load a dictionary from {filename}'))
        detector = indict['detector']
        self.id = detector['id']
        pixels = detector['pixels']
        self.rows = pixels['rows']
        self.columns = pixels['columns']
        self.pixel_size = pixels['size']
        self.lens_magnification = indict['lens_magnification']

    def cli(self):
        print(' -- Configure the detector -- ')
        self.set_single_attr_cli('id', 'detector ID')
        self.set_single_attr_cli('rows', 'number of pixel rows')
        self.set_single_attr_cli('columns', 'number of pixel columns')
        self.set_single_attr_cli('pixel_size', 'pixel size in mm (enter either a single value for '+
                'square pixels or a pair of values for the size in the respective row and column '+
                'directions)', list_flag=True)
        self.set_single_attr_cli('lens_magnification', 'lens magnification')

    def construct_nxdetector(self):
        nxdetector = NXdetector()
        nxdetector.local_name = self.id
#        nxdetector.rows = self.rows
#        nxdetector.rows.attrs['units'] = 'pixels'
#        nxdetector.columns = self.columns
#        nxdetector.columns.attrs['units'] = 'pixels'
        if len(self.pixel_size) ==1:
            nxdetector.x_pixel_size = self.pixel_size[0]
            nxdetector.y_pixel_size = self.pixel_size[0]
        else:
            nxdetector.x_pixel_size = self.pixel_size[0]
            nxdetector.y_pixel_size = self.pixel_size[1]
        nxdetector.x_pixel_size.attrs['units'] = 'mm'
        nxdetector.y_pixel_size.attrs['units'] = 'mm'
        return(nxdetector)


class ScanData(BaseModel):
    spec_file: FilePath
    scan_number: conint(gt=0)

    @validator('spec_file')
    @classmethod
    def validate_spec_file(cls, spec_file):
        try:
            sspec_file = FileSpec(spec_file)
        except:
            raise(ValueError(f'Invalid SPEC file {spec_file}'))
        else:
            return(spec_file)

    @validator('scan_number')
    @classmethod
    def validate_scan_number(cls, scan_number, values):
        spec_file = values.get('spec_file')
        if spec_file is not None:
            spec_scan = FileSpec(spec_file)
            scan = spec_scan.get_scan_by_number(scan_number)
            if scan is not None:
                return(scan_number)
            else:
                raise(ValueError(f'There is no scan number {scan_number} in {spec_file}'))

    @property
    def available_scan_numbers(self):
        return(get_available_scan_numbers(self.spec_file))

    @property
    def scanparser(self):
        return(get_scanparser(self.spec_file, self.scan_number))

    def scan_number_cli(self, attr_desc):
        if len(self.available_scan_numbers) == 1:
            self.scan_number = self.available_scan_numbers[0]
        else:
            if hasattr(self, 'scan_number'):
                print(f'Currently selected {attr_desc}scan number is: {self.scan_number}')
                menu_options = [f'Select one of the available {attr_desc}scan numbers',
                                f'Keep the currently selected {attr_desc}scan number']
                input_mode = input_menu(menu_options, header='Choose one of the following options '+
                        'for selecting a scan number')
            else:
                input_mode = 0
            if input_mode == 0:
                print(f'Available {attr_desc}scan numbers in {self.spec_file} are: '+
                        f'{self.available_scan_numbers}')
                accept_scan_numbers = False
                while not accept_scan_numbers:
                    try:
                        self.scan_number = input_int(f'Enter a {attr_desc}scan number')
                    except ValidationError as e:
                        print(e)
                    except KeyboardInterrupt as e:
                        raise(e)
                    except BaseException as e:
                        print(f'Unexpected {type(e).__name__}: {e}')
                    else:
                        accept_scan_numbers = True
            elif input_mode == 1:
                pass

    def cli(self, **cli_kwargs):
        if cli_kwargs.get('attr_desc') is not None:
            attr_desc = f'{cli_kwargs["attr_desc"]} '
        else:
            attr_desc = ''
        print(f' -- Configure the location of the {attr_desc}scan data -- ')
        self.set_single_attr_cli('spec_file', attr_desc+'SPEC file path')
        self.scan_number_cli(attr_desc)


class SpecScans(BaseModel):
    spec_file: FilePath
    scan_numbers: conlist(item_type=conint(gt=0), min_items=1)

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

    @property
    def available_scan_numbers(self):
        return(get_available_scan_numbers(self.spec_file))

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
        print(f' -- Configure which scans to use from a single {attr_desc}SPEC file')
        self.set_single_attr_cli('spec_file', attr_desc+'SPEC file path')
        self.scan_numbers_cli(attr_desc)

#    def get_nxentry_name(self, scan_number:int):
#        return(f'{self.get_scanparser(scan_number).scan_name}/scan_{scan_number}')
#    def get_detector_data(self, detectors:list[Detector], scan_number:int, scan_step_index:int):
#        return(get_detector_data(tuple([detector.prefix for detector in detectors]), self.spec_file, scan_number, scan_step_index))
#    def get_mapping_motor_value(self, scan_number:int, scan_step_index:int, mapping_motor_mnemonic:str):
#        return(get_mapping_motor_value(self.spec_file, scan_number, scan_step_index, mapping_motor_mnemonic))
#    def get_index_in_map(self, scan_number:int, scan_step_index:int, map_coordinates:dict):
#        map_coordinates = tuple([(coordinate_name,tuple(coordinate_values)) for coordinate_name,coordinate_values in map_coordinates.items()])
#        return(get_index_in_map(self.spec_file, scan_number, scan_step_index, map_coordinates))
#    def construct_nxobject(self, nxgroup, mapping_motors=[], detector_prefixes=[]) -> list[NXentry]:
#        if 'spec_scans' not in nxgroup:
#            nxgroup.spec_scans = NXcollection()
#        scanparser = self.get_scanparser(self.scan_numbers[0])
#        nxgroup.spec_scans[scanparser.scan_name] = NXcollection()
#        nxgroup.spec_scans[scanparser.scan_name].attrs['spec_file'] = self.spec_file
#        nxgroup.spec_scans[scanparser.scan_name].attrs['date'] = scanparser.spec_scan.file_date
#        for scan_number in self.scan_numbers:
#            scanparser = self.get_scanparser(scan_number)
#            entry_name = self.get_nxentry_name(scan_number)
#            nxgroup.spec_scans[entry_name] = NXentry()
#            nxgroup.spec_scans[entry_name].start_time = scanparser.spec_scan.date
#            nxgroup.spec_scans[entry_name].instrument = NXinstrument()
#            for mapping_motor in mapping_motors:
#                nxgroup.spec_scans[entry_name].instrument[mapping_motor.mnemonic] = NXpositioner()
#                motor_value = self.get_mapping_motor_value(scan_number, -1, mapping_motor.mnemonic)
#                nxgroup.spec_scans[entry_name].instrument[mapping_motor.mnemonic].value = motor_value
#                nxgroup.spec_scans[entry_name].instrument[mapping_motor.mnemonic].value.units = mapping_motor.units
#            for detector_prefix in detector_prefixes:
#                nxgroup.spec_scans[entry_name].instrument[detector_prefix] = NXdetector()
#        return(nxgroup)

#@cache
#def get_mapping_motor_value(spec_file:str, scan_number:int, scan_step_index:int, mapping_motor_mnemonic:str):
#    scanparser = get_scanparser(spec_file, scan_number)
#    if mapping_motor_mnemonic in scanparser.spec_scan_motor_mnes:
#        motor_i = scanparser.spec_scan_motor_mnes.index(mapping_motor_mnemonic)
#        if scan_step_index >= 0:
#            scan_step = np.unravel_index(scan_step_index, scanparser.spec_scan_shape, order='F')
#            motor_value = scanparser.spec_scan_motor_vals[motor_i][scan_step[motor_i]]
#        else:
#            motor_value = scanparser.spec_scan_motor_vals[motor_i]
#    else:
#        motor_value = [scanparser.get_spec_positioner_value(mapping_motor_mnemonic)]
#    return(motor_value)
#@cache
#def get_index_in_map(spec_file:str, scan_number:int, scan_step_index:int, map_coordinates:tuple):
#    map_coordinates = dict(map_coordinates)
#    map_index = ()
#    scanparser = get_scanparser(spec_file, scan_number)
#    for mnemonic,values in map_coordinates.items():
#        coord_val = get_mapping_motor_value(spec_file, scan_number, scan_step_index, mnemonic)
#        coord_index = values.index(coord_val)
#        map_index = (coord_index, *map_index)
#    return(map_index)
#@lru_cache(maxsize=10)
#def get_detector_data(detector_prefixes:tuple, spec_file:str, scan_number:int, scan_step_index:int):
#    detector_data = []
#    scanparser = get_scanparser(spec_file, scan_number)
#    for prefix in detector_prefixes:
#        image_data = scanparser.get_detector_data(prefix, scan_step_index)
#        detector_data.append(image_data)
#    return(detector_data)


class FlatField(ScanData):
#    data_path: constr(strip_whitespace=True, min_length=1)
    stack_info: conlist(item_type=dict, min_items=1, max_items=1) = []

#    @validator('data_path')
#    @classmethod
#    def validate_data_path(cls, data_path):
#        if os.path.splitext(data_path)[1] == '.h5':
#            file_exists_and_readable(data_path)
#            try:
#                if not h5py.is_hdf5(data_path):
#                    raise(ValueError(f'Invalid h5 file {data_path}'))
#            except BaseException as e:
#                print(f'Unexpected {type(e).__name__}: {e}')
#            else:
#                return(data_path)
#        else:
#            if not os.path.isdir(data_path):
#                raise(ValueError(f'Invalid file path {data_path}'))
#            return(data_path)

    def cli(self, **cli_kwargs):
        if cli_kwargs.get('attr_desc') is not None:
            attr_desc = f'{cli_kwargs["attr_desc"]} '
        else:
            attr_desc = ''
        detector = cli_kwargs.get('detector')
        print(f' -- Configure the location of the {attr_desc}scan data -- ')
        self.set_single_attr_cli('spec_file', attr_desc+'SPEC file path')
        self.scan_number_cli(attr_desc)
        use_image_range = False
        if len(self.stack_info):
            use_image_range = input_yesno(f'Current {attr_desc}image range:\n{self.stack_info}\n'+
                    f'Keep these settings? (y/n)')
        if not use_image_range:
            parser = self.scanparser
            first_offset = parser.starting_image_offset
            num_image = parser.get_num_image(detector.id.upper())
            start_index = parser.starting_image_index-parser.starting_image_offset
            num_img = input_int(f'Enter the number of {attr_desc}images', ge=1, le=num_image,
                default=num_image)
            self.stack_info = [{'scan_number': self.scan_number, 'start_index': start_index,
                    'offset_index': first_offset, 'num': num_img,
                    'ref_height': parser.vertical_shift}]


class TomoField(SpecScans):
#    data_path: constr(strip_whitespace=True, min_length=1)
    theta_range: dict = {}
    stack_info: conlist(item_type=dict, min_items=1) = []

#    @validator('data_path')
#    def validate_data_path(cls, data_path):
#        if os.path.splitext(data_path)[1] == '.h5':
#            file_exists_and_readable(data_path)
#            try:
#                if not h5py.is_hdf5(data_path):
#                    raise(ValueError(f'Invalid h5 file {data_path}'))
#            except BaseException as e:
#                print(f'Unexpected {type(e).__name__}: {e}')
#            else:
#                return(data_path)
#        else:
#            if not os.path.isdir(data_path):
#                raise(ValueError(f'Invalid file path {data_path}'))
#            return(data_path)

    @validator('theta_range')
    def validate_theta_range(cls, theta_range):
        if len(theta_range) != 4:
            raise(ValueError(f'Invalid theta range {theta_range}'))
        is_num(theta_range['start'])
        is_num(theta_range['end'])
        is_int(theta_range['num'])
        is_int(theta_range['start_index'])
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

    def image_range_cli(self, scan_number, attr_desc, detector_id):
        # Parse the available image range
        parser = self.get_scanparser(scan_number)
        first_offset = parser.starting_image_offset
        num_image = parser.get_num_image(detector_id.upper())

        # Select the image set matching the theta range
        num_theta = self.theta_range['num']
        theta_index_start = self.theta_range['start_index']
        if num_theta > num_image-theta_index_start:
            raise(ValueError(f'Available {attr_desc}image indices incompatible with theta range:'+
                    f'\n\tNumber of thetas and offset = {num_theta} and {theta_index_start}'+
                    f'\n\tNumber of available images and offset {num_image}'))
        start_index = parser.starting_image_index-parser.starting_image_offset
        have_stack = False
        for stack in self.stack_info:
            if stack.get('scan_number', None) is not None and stack['scan_number'] == scan_number:
                stack['start_index'] = start_index
                stack['offset_index'] = first_offset+theta_index_start
                stack['num'] = num_theta
                stack['ref_height'] = parser.vertical_shift
                have_stack = True
        if not have_stack:
            self.stack_info.append({'scan_number': scan_number, 'start_index': start_index,
                    'offset_index': first_offset+theta_index_start, 'num': num_theta,
                    'ref_height': parser.vertical_shift})

    def cli(self, **cli_kwargs):
        if cli_kwargs.get('attr_desc') is not None:
            attr_desc = f'{cli_kwargs["attr_desc"]} '
        else:
            attr_desc = ''
        station = cli_kwargs.get('station')
        detector = cli_kwargs.get('detector')
        print(f' -- Configure the location of the {attr_desc}scan data -- ')
        self.set_single_attr_cli('spec_file', attr_desc+'SPEC file path')
        self.scan_numbers_cli(attr_desc)
#        if station in ('smb', 'fast', 'id1a3', 'id3a'):
#            self.data_path = f'{os.path.dirname(self.spec_file)}/{self.scan_numbers[0]}/nf'
#        elif station in ('fmb', 'id3b'):
#            self.data_path = f'{self.spec_file}_{detector.id.upper()}_{self.scan_numbers[0]:03d}.h5'
        for i, scan_number in enumerate(self.scan_numbers):
            self.theta_range_cli(scan_number, attr_desc, station)
            self.image_range_cli(scan_number, attr_desc, detector.id)


class MapConfig(BaseModel):
    title: constr(strip_whitespace=True, min_length=1)
    station: Literal['smb', 'fmb', 'fast', 'id1a3', 'id3a', 'id3b'] = None
    detector: Detector = Detector.construct()
    dark_field: Optional[FlatField]
    bright_field: FlatField = FlatField.construct()
    tomo_fields: TomoField = TomoField.construct()
    _thetas: list[float] = PrivateAttr()

    @validator('station', pre=True)
    @classmethod
    def validate_station(cls, station):
        return(station.lower())

#    @property
#    def bright_field(self):
#        return(self.bright_field)

#    @property
#    def tomo_fields(self):
#        return(self.tomo_fields)

#FIX cache?
    @property
    def thetas(self):
        try:
            return(self._thetas)
        except:
            theta_range = self.tomo_fields.theta_range
            self._thetas = np.linspace(theta_range['start'], theta_range['end'],
                    theta_range['num'])
            return(self._thetas)

    def cli(self):
        print('\n\n -- Configure a map from a set of SPEC scans (dark, bright, and tomo), '+
                'and / or detector data -- ')
        self.set_single_attr_cli('title', 'title for the map entry')
        self.set_single_attr_cli('station', 'name of the station at which scans were collected '+
                '(examples: id1a3, id3a, id3b, smb, or fmb)')
        import_scanparser(self.station)
        use_detector_config = False
        if hasattr(self.detector, 'id') and len(self.detector.id):
            use_detector_config = input_yesno(f'Current detector settings:\n{self.detector}\n'+
                    f'Keep these settings? (y/n)')
        if not use_detector_config:
            have_detector_config = input_yesno(f'Is a detector configuration file available? (y/n)')
            if have_detector_config:
                detector_config_file = input(f'Enter detector configuration file name: ')
                self.detector.construct_from_yaml(detector_config_file)
            else:
                self.set_single_attr_cli('detector', 'detector')
        have_dark_field = input_yesno(f'Are Dark field images available? (y/n)')
        if have_dark_field:
            attr = getattr(self, 'dark_field', None)
            if attr is None:
                self.dark_field = FlatField.construct()
            self.set_single_attr_cli('dark_field', 'Dark field', chain_attr_desc=True,
                    station=self.station, detector=self.detector)
        self.set_single_attr_cli('bright_field', 'Bright field', chain_attr_desc=True,
                station=self.station, detector=self.detector)
        self.set_single_attr_cli('tomo_fields', 'Tomo field', chain_attr_desc=True,
                station=self.station, detector=self.detector)

    def construct_nxentry(self, nxroot, logger=logging.getLogger(__name__)):
        t0 = time()
        # Construct base NXentry
        nxentry = NXentry()

        # Add NXentry to the NXroot
        nxroot[self.title] = nxentry

        # Add NXsample to NXentry
#FIX        nxentry.sample = self.sample.construct_nxobject()

        # Add NXinstrument to NXentry
        nxentry.instrument = NXinstrument()

        # Add an NXdetector to the NXinstrument
        nxentry.instrument[self.detector.id] = self.detector.construct_nxdetector()

        # Tag the NXentry with the station (as an attribute)
        nxentry.attrs['station'] = self.station

        # Add independent dimensions & their coordinate values to NXentry as an NXdata
        nxentry.independent_dimensions = NXdata()
        nxentry.independent_dimensions.attrs['axes'] = ['theta']
        nxentry.independent_dimensions['theta'] = NXfield(self.thetas)
        nxentry.independent_dimensions['theta'].units = 'degrees'
        nxentry.independent_dimensions['theta'].attrs['long_name'] = f'theta (degrees)'

        # Since independent dimensions are reused for all signals in this map,
        # define a convenience function for linking to their values.
        def link_independent_dimensions(nxdata_path):
            for indep_dim in self.independent_dimensions:
                nxroot[nxdata_path].makelink(
                        nxentry.independent_dimensions[indep_dim.label])
                nxroot[nxdata_path].attrs[f'{indep_dim.label}_indices'] = \
                        nxentry.independent_dimensions.attrs[f'{indep_dim.label}_indices']

        # Add a NXcollection to the base NXentry to hold metadata about the spec scans in the map
        nxentry.spec_scans = NXcollection()

        # Add the dark field
        if hasattr(self, 'bright_field'):
            print(f'bright_field:\n{self.bright_field}')
            return(nxentry)
            # Add a NXcollection to the NXcollection for all the scans in a single spec file
            scanparser = bright_field.scanparser(bright_field.scan_number)
            nxentry.spec_scans[scanparser.scan_name] = bright_field.construct_nxcollection(detector)
                t1 = time()
                entry_name = f'{scanparser.scan_name}'
                # Use multiple threads to fill in arrays set up to contain point-by-point scan data
                def fill_data(scan_step_index:int):
                    index = scans.get_index(scan_number, scan_step_index, self)
                    logger.debug(f'Adding data to nexus arrays from scan {scanparser.scan_title} step {scan_step_index}')
                    # Fill in the scalar values arrays
                    if self.signals.scalar_values is not None:
                        for scalar_values in self.signals.scalar_values:
                            nxentry[scalar_values.label][index] = scalar_values.get_value(scans, scan_number, scan_step_index)
                    # Fill in processed data
                    for scatter_type in ('saxs', 'waxs'):
                        scatter_config = getattr(self.signals, scatter_type, None)
                        if scatter_config is not None:
                            # Fill in paths to raw detector data
                            for detector in scatter_config.detectors:
                                nxentry.spec_scans[entry_name].instrument[detector.prefix].raw_data_files[scan_step_index] = scanparser.get_detector_data_file(detector.prefix, scan_step_index)
                            # Fill in the integrated data arrays
                            for integration_type in scatter_config.integration.configured_integration_types:
                                integration_config = getattr(scatter_config.integration, integration_type)
                                nxentry[scatter_type][integration_type].data.I[index] = integration_config.get_integrated_data(scatter_config.detectors, scans, scan_number, scan_step_index)
                # Create a pool of threads that run fill_data for each scan point.
                # Run the 0th scan step first so threads can access certain integrators, etc.
                # through the cache.
                fill_data(0)
                with ThreadPool(processes=4) as pool:
                    pool.map(fill_data, range(1, scanparser.spec_scan_npts))
                logger.info(f'Filled nexus data arrays for scan {scanparser.scan_title} in {time()-t1:.2f} seconds.')
        logger.info(f'Constructed an NXentry for {self.title}: {time()-t0:.2f} seconds')
        return(nxentry)


class TOMOWorkflow(BaseModel):
    sample_maps: conlist(item_type=MapConfig, min_items=1) = [MapConfig.construct()]
    def cli(self):
        print(' -- Configure a map -- ')
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


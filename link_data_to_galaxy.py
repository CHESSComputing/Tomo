#!/usr/bin/env python3

import logging

import os
import sys
import pwd
import h5py
import argparse
from ast import literal_eval
from bioblend.galaxy import GalaxyInstance

from tomo import Tomo
import msnc_tools as msnc

def get_folder_id(gi, path):
    library_id = None
    folder_id = None
    folder_names = path[1:] if len(path) > 1 else []
    new_folders = folder_names
    libs = gi.libraries.get_libraries(name=path[0])
    if libs:
        for lib in libs:
            library_id = lib['id']
            folders = gi.libraries.get_folders(library_id, folder_id=None, name=None)
            for i, folder in enumerate(folders):
                fid = folder['id']
                details = gi.libraries.show_folder(library_id, fid)
                library_path = details['library_path']
                match = library_path == folder_names
                if library_path == folder_names:
                    return (library_id, fid, [])
                elif len(library_path) < len(folder_names):
                    if library_path == folder_names[:len(library_path)]:
                        nf = folder_names[len(library_path):]
                        if len(nf) < len(new_folders):
                            folder_id = fid
                            new_folders = nf
    return (library_id, folder_id, new_folders)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='Create Galaxy Shared Data Library')
    parser.add_argument('-c', '--config',
            default='config.txt',
            help='plain config txt file')
    parser.add_argument('-g', '--galaxy',
            help='Target Galaxy instance URL/IP address')
    parser.add_argument('-u', '--user',
            default=None,
            help='Galaxy user email address')
    parser.add_argument('-p', '--password',
            help='Password for the Galaxy user')
    parser.add_argument('-a', '--api_key',
            help='Galaxy admin user API key (required if not defined in the tools list file)')
    parser.add_argument('-l', '--log',
            type=argparse.FileType('w'),
            default=sys.stdout,
            help='log file')
    parser.add_argument('-d', '--debug',
            action='store_true',
            help='Debug')
    args = parser.parse_args()

    # Set basic log configuration
    logging_format = '%(asctime)s : %(levelname)s - %(module)s : %(funcName)s - %(message)s'
    if args.debug:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'
    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f'Invalid log_level: {log_level}')
    logging.basicConfig(format=logging_format, level=level, force=True,
            handlers=[logging.StreamHandler()])

    logging.debug(f'log = {args.log}')
    logging.debug(f'is log stdout? {args.log == sys.stdout}')
    logging.debug(f'debug = {args.debug}')

    # Instantiate Tomo object
    tomo = Tomo(config_file=args.config, log_level=log_level, log_stream=args.log)

    # Check config file
    if not tomo.is_valid:
        raise ValueError('Invalid config and/or detector file provided.')
    config = tomo.cf.config

    # Find all available image files
    dark_files, bright_files, tomo_files = tomo.findImageFiles()
    if not tomo.is_valid:
        raise ValueError('Unable to find available image files')

    # Save a copy of config file in workdir to link to Galaxy
    work_folder = config['work_folder']
    config_name = f'{work_folder}/config.yaml'
    tomo.cf.saveFile(config_name)

    # Initialize collections
    collections = [{'name' : 'config', 'files' : [config_name]}]
    collections.append({'name' : 'tdf', 'files' : dark_files})
    collections.append( {'name' : 'tbf', 'files' : bright_files})
    if len(tomo_files) != config['stack_info']['num']:
        raise ValueError('Inconsistent number of tomography image sets')
    for i,stack in enumerate(config['stack_info']['stacks']):
        collections.append({'name' : f'set{stack["index"]}', 'files' : tomo_files[i]})
    logging.debug(f'collections:\n{collections}')

    # Get Galaxy instance
    if args.user and args.password:
        gi = GalaxyInstance(url=args.galaxy, email=args.user, password=args.password)
    elif args.api_key:
        gi = GalaxyInstance(url=args.galaxy, key=args.api_key)
    else:
        exit('Please specify either a valid Galaxy username/password or an API key.')

    # Create Galaxy work library/folder
    # Combine the user ID and the top level directory name as the base library name
    userid = pwd.getpwuid(os.getuid()).pw_name
    lib_path = [p.strip() for p in f'{userid}/{os.path.split(work_folder)[1]}'.split('/')]
    (library_id, folder_id, folder_names) = get_folder_id(gi, lib_path)
    if not library_id:
        library = gi.libraries.create_library(lib_path[0], description=None, synopsis=None)
        library_id = library['id']
#        if args.user:
#            gi.libraries.set_library_permissions(library_id, access_ids=args.user,
#                    manage_ids=args.user, modify_ids=args.user)
        logging.info(f'Created Library:\n{library}')
    for folder_name in folder_names:
        folders = gi.libraries.create_folder(library_id, folder_name, description=None,
                base_folder_id=folder_id)
#        if args.user:
#            for folder in folders:
#                gi.folders.set_library_permissions(folder['id'], access_ids=args.user, 
#                    manage_ids=args.user, modify_ids=args.user)
        logging.info(f'Created Folder:\n{folders}')

    # Create folders and data symlinks as needed
    collection_datasets = []
    for collection in collections:
        name = collection['name']
        if name == 'config':
            folder_path = lib_path
            link_data_only='copy_files'
        else:
            folder_path = lib_path+[name]
            link_data_only='link_to_files'
        (library_id, folder_id, folder_names) = get_folder_id(gi, folder_path)
        # Create library as needed
        if not library_id:
            library = gi.libraries.create_library(folder_path[0], description=None,
                    synopsis=None)
            library_id = library['id']
#            if args.user:
#                gi.libraries.set_library_permissions(library_id, access_ids=args.user,
#                        manage_ids=args.user, modify_ids=args.user)
            logging.info(f'Created Library:\n{library}')
        # Create folders as needed
        for folder_name in folder_names:
            folders = gi.libraries.create_folder(library_id, folder_name, description=None,
                    base_folder_id=folder_id)
#            if args.user:
#                for folder in folders:
#                    gi.folders.set_library_permissions(folder['id'], access_ids=args.user, 
#                        manage_ids=args.user, modify_ids=args.user)
            logging.info(f'Created Folder:\n{folders}')
            folder_id = folders[0]['id']
        # Create sym links
        files = collection['files']
        logging.debug(f'files:\n{files}')
        if files:
            # Test FIXME
            #files = files[0:5]
            # Bioblend has a timeout, so split in batches for large numbers of files
            gi_datasets = []
            start_index = 0
            num_files_max = 100
            if len(files) > num_files_max:
                for _ in range(int(len(files)/num_files_max)):
                    file_paths = '\n'.join(files[start_index:start_index+num_files_max])
                    gi_datasets += gi.libraries.upload_from_galaxy_filesystem(library_id,
                            file_paths, folder_id=folder_id, file_type='auto', dbkey='?',
                            link_data_only=link_data_only, roles='', preserve_dirs=False,
                            tag_using_filenames=False, tags=None)
                    start_index += num_files_max
            file_paths = '\n'.join(files[start_index:])
            gi_datasets += gi.libraries.upload_from_galaxy_filesystem(library_id, file_paths,
                    folder_id=folder_id, file_type='auto', dbkey='?',
                    link_data_only='link_to_files', roles='', preserve_dirs=False,
                    tag_using_filenames=False, tags=None)
            collection_datasets.append({'name' : name, 'folder_id' : folder_id,
                    'datasets' : gi_datasets})
            folder = gi.folders.show_folder(folder_id)
            folder_name = folder['name']
    logging.debug(f'collection_datasets:\n{collection_datasets}\n\n')

    # Make a history for the data
    history_name = f'tomo {userid} {os.path.split(work_folder)[1]}'
    history = gi.histories.create_history(name=history_name)
    logging.info(f'Created history:\n{history}')
    history_id = history['id']
    for collection_dataset in collection_datasets:
        name = collection_dataset["name"]
        if name == 'config':
            for dataset in collection_dataset['datasets']:
                gi.histories.copy_dataset(history_id, dataset['id'], source='library')
        else:
            folder = gi.folders.show_folder(collection_dataset['folder_id'])
            folder_name = folder['name']
            collection_description = {'collection_type' : 'list', 'name' : name,
                    'hide_source_items' : True}
            element_identifiers = []
            for dataset in collection_dataset['datasets']:
                # Prepend folder_name to file name to identify files in Galaxy
                name = f'{folder_name}_{dataset["name"]}'
                element = {'id' : dataset['id'], 'name' : name, 'src' : 'ldda'}
                logging.debug(f'added collection element: {element}')
                element_identifiers.append(element)
            collection_description['element_identifiers'] = element_identifiers
            logging.debug(f'collection_description:\n{collection_description}')
            gi_collection = gi.histories.create_dataset_collection(history_id,
                    collection_description)
            # Hide individual files in the history
            for element in gi_collection['elements']:
                dataset_id = element['object']['id']
                resp = gi.histories.update_dataset(history_id, dataset_id, visible=False)
            logging.debug(f'gi_collection:\n{gi_collection}')


# TODO add option to either 
#     get a URL to share the history
#     or to share with specific users
# This might require using: 
#     https://bioblend.readthedocs.io/en/latest/api_docs/galaxy/docs.html#using-bioblend-for-raw-api-calls

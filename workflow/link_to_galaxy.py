#!/usr/bin/env python3

import logging

from bioblend.galaxy import GalaxyInstance
from nexusformat.nexus import *
from os import path

from .models import TOMOWorkflow

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
                if library_path == folder_names:
                    return (library_id, fid, [])
                elif len(library_path) < len(folder_names):
                    if library_path == folder_names[:len(library_path)]:
                        nf = folder_names[len(library_path):]
                        if len(nf) < len(new_folders):
                            folder_id = fid
                            new_folders = nf
    return (library_id, folder_id, new_folders)

def link_to_galaxy(filename:str, logger=logging.getLogger(__name__), galaxy=None,
        user=None, password=None, api_key=None) -> None:

    wf = TOMOWorkflow.construct_from_nexus(filename)
    nxroot = nxload(filename, 'r')
    nxroot.close()

    # Get a Galaxy instance
    if user is not None and password is not None :
        gi = GalaxyInstance(url=galaxy, email=user, password=password)
    elif api_key is not None:
        gi = GalaxyInstance(url=galaxy, key=api_key)
    else:
        exit('Please specify either a valid Galaxy username/password or an API key.')

    for sample_map in wf.sample_maps:
        nxentry = nxroot[sample_map.title]

        cycle = nxentry.instrument.source.attrs['cycle']
        btr = nxentry.instrument.source.attrs['btr']
        sample = nxentry.sample.name

        # Create a Galaxy work library/folder
        # Combine the cycle, BTR and sample name as the base library name
        lib_path = [p.strip() for p in f'{cycle}/{btr}/{sample}'.split('/')]
        (library_id, folder_id, folder_names) = get_folder_id(gi, lib_path)
        if not library_id:
            library = gi.libraries.create_library(lib_path[0], description=None, synopsis=None)
            library_id = library['id']
#            if user:
#                gi.libraries.set_library_permissions(library_id, access_ids=user,
#                        manage_ids=user, modify_ids=user)
            logging.info(f'Created Library:\n{library}')
        if len(folder_names):
            folder = gi.libraries.create_folder(library_id, folder_names[0], description=None,
                    base_folder_id=folder_id)[0]
            folder_id = folder['id']
            logging.info(f'Created Folder:\n{folder}')
            folder_names.pop(0)
            while len(folder_names):
                folder = gi.folders.create_folder(folder['id'], folder_names[0],
                        description=None)
                folder_id = folder['id']
                logging.info(f'Created Folder:\n{folder}')
                folder_names.pop(0)

        # Create a sym link for the Nexus file
        dataset = gi.libraries.upload_from_galaxy_filesystem(library_id, path.abspath(filename),
                folder_id=folder_id, file_type='auto', dbkey='?', link_data_only='link_to_files',
                roles='', preserve_dirs=False, tag_using_filenames=False, tags=None)[0]

        # Make a history for the data
        history_name = f'tomo {btr} {sample}'
        history = gi.histories.create_history(name=history_name)
        logging.info(f'Created history:\n{history}')
        history_id = history['id']
        gi.histories.copy_dataset(history_id, dataset['id'], source='library')

# TODO add option to either 
#     get a URL to share the history
#     or to share with specific users
# This might require using: 
#     https://bioblend.readthedocs.io/en/latest/api_docs/galaxy/docs.html#using-bioblend-for-raw-api-calls


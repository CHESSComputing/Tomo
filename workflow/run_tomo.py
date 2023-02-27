#!/usr/bin/env python3

# RV TODO FIX add find center metadata to Nexus file

import logging
logger = logging.getLogger(__name__)

import numpy as np
try:
    import numexpr as ne
except:
    pass
try:
    import scipy.ndimage as spi
except:
    pass

from multiprocessing import cpu_count
from nexusformat.nexus import *
from os import mkdir, path
try:
    from skimage.transform import iradon
except:
    pass
try:
    from skimage.restoration import denoise_tv_chambolle
except:
    pass
from time import time
try:
    import tomopy
except:
    pass
from yaml import safe_load, safe_dump

from msnctools.fit import Fit
from msnctools.general import illegal_value, is_int, is_num, is_index_range, input_int, input_num, \
        input_yesno, input_menu, draw_mask_1d, select_image_bounds, select_one_image_bound, \
        clear_imshow, quick_imshow, clear_plot, quick_plot

from workflow.models import import_scanparser, FlatField, TomoField, TomoWorkflow
from workflow.__version__ import __version__

num_core_tomopy_limit = 24

def nxcopy(nxobject:NXobject, exclude_nxpaths:list[str]=[], nxpath_prefix:str='') -> NXobject:
    '''Function that returns a copy of a nexus object, optionally exluding certain child items.

    :param nxobject: the original nexus object to return a "copy" of
    :type nxobject: nexusformat.nexus.NXobject
    :param exlude_nxpaths: a list of paths to child nexus objects that
        should be exluded from the returned "copy", defaults to `[]`
    :type exclude_nxpaths: list[str], optional
    :param nxpath_prefix: For use in recursive calls from inside this
        function only!
    :type nxpath_prefix: str
    :return: a copy of `nxobject` with some children optionally exluded.
    :rtype: NXobject
    '''

    nxobject_copy = nxobject.__class__()
    if not len(nxpath_prefix):
        if 'default' in nxobject.attrs:
            nxobject_copy.attrs['default'] = nxobject.attrs['default']
    else:
        for k, v in nxobject.attrs.items():
            nxobject_copy.attrs[k] = v

    for k, v in nxobject.items():
        nxpath = path.join(nxpath_prefix, k)

        if nxpath in exclude_nxpaths:
            continue

        if isinstance(v, NXgroup):
            nxobject_copy[k] = nxcopy(v, exclude_nxpaths=exclude_nxpaths,
                    nxpath_prefix=path.join(nxpath_prefix, k))
        else:
            nxobject_copy[k] = v

    return(nxobject_copy)

class set_numexpr_threads:

    def __init__(self, num_core):
        if num_core is None or num_core < 1 or num_core > cpu_count():
            self.num_core = cpu_count()
        else:
            self.num_core = num_core

    def __enter__(self):
        self.num_core_org = ne.set_num_threads(self.num_core)

    def __exit__(self, exc_type, exc_value, traceback):
        ne.set_num_threads(self.num_core_org)

class Tomo:
    """Processing tomography data with misalignment.
    """
    def __init__(self, num_core=-1, output_folder='.', save_figs='no', test_mode=False):
        """Initialize with optional config input file or dictionary
        """
        self.num_core = num_core
        self.output_folder = path.abspath(output_folder)
        if not path.isdir(output_folder):
            mkdir(path.abspath(output_folder))
        if not isinstance(test_mode, bool):
            raise ValueError(f'Invalid parameter test_mode ({test_mode})')
        self.test_mode = test_mode
        self.test_config = {}
        if self.test_mode:
            if save_figs != 'only':
                logger.warning('Ignoring save_figs in test mode')
            save_figs = 'only'
        if save_figs == 'only':
            self.save_only = True
            self.save_figs = True
        elif save_figs == 'yes':
            self.save_only = False
            self.save_figs = True
        elif save_figs == 'no':
            self.save_only = False
            self.save_figs = False
        else:
            raise ValueError(f'Invalid parameter save_figs ({save_figs})')
        if self.save_only:
            self.block = False
        else:
            self.block = True
        if self.num_core == -1:
            self.num_core = cpu_count()
        if not is_int(self.num_core, gt=0, log=False):
            raise ValueError(f'Invalid parameter num_core ({num_core})')
        if self.num_core > cpu_count():
            logger.warning(f'num_core = {self.num_core} is larger than the number of available '
                    f'processors and reduced to {cpu_count()}')
            self.num_core= cpu_count()

    def read(self, filename):
        extension = path.splitext(filename)[1]
        if extension == '.yml' or extension == '.yaml':
            with open(filename, 'r') as f:
                config = safe_load(f)
#            if len(config) > 1:
#                raise ValueError(f'Multiple root entries in {filename} not yet implemented')
#            if len(list(config.values())[0]) > 1:
#                raise ValueError(f'Multiple sample maps in {filename} not yet implemented')
            return(config)
        elif extension == '.nxs':
            with NXFile(filename, mode='r') as nxfile:
                nxroot = nxfile.readfile()
            return(nxroot)
        else:
            raise ValueError(f'Invalid filename extension ({extension})')

    def write(self, data, filename):
        extension = path.splitext(filename)[1]
        if extension == '.yml' or extension == '.yaml':
            with open(filename, 'w') as f:
                safe_dump(data, f)
        elif extension == '.nxs':
            data.save(filename, mode='w')
        elif extension == '.nc':
            data.to_netcdf(path=filename)
        else:
            raise ValueError(f'Invalid filename extension ({extension})')

    def gen_reduced_data(self, data):
        """Generate the reduced tomography images.
        """
        logger.info('Generate the reduced tomography images')

        if isinstance(data, dict):
            # Create Nexus format object from input dictionary
            wf = TomoWorkflow(**data)
            if len(wf.sample_maps) > 1:
                raise ValueError(f'Multiple sample maps not yet implemented')
#            print(f'\nwf:\n{wf}\n')
            nxroot = NXroot()
            t0 = time()
            for sample_map in wf.sample_maps:
                logger.info(f'Start constructing the {sample_map.title} map.')
                import_scanparser(sample_map.station)
                sample_map.construct_nxentry(nxroot, include_raw_data=False)
            logger.info(f'Constructed all sample maps in {time()-t0:.2f} seconds.')
            nxentry = nxroot[nxroot.attrs['default']]
            # Get test mode configuration info
            if self.test_mode:
                self.test_config = data['sample_maps'][0]['test_mode']
        elif isinstance(data, NXroot):
            nxentry = data[data.attrs['default']]
        else:
            raise ValueError(f'Invalid parameter data ({data})')

        # Create an NXprocess to store data reduction (meta)data
        reduced_data = NXprocess()

        # Generate dark field
        if 'dark_field' in nxentry['spec_scans']:
            reduced_data = self._gen_dark(nxentry, reduced_data)

        # Generate bright field
        reduced_data = self._gen_bright(nxentry, reduced_data)

        # Set vertical detector bounds for image stack
        img_x_bounds = self._set_detector_bounds(nxentry, reduced_data)
        logger.info(f'img_x_bounds = {img_x_bounds}')
        reduced_data['img_x_bounds'] = img_x_bounds

        # Set zoom and/or theta skip to reduce memory the requirement
        zoom_perc, num_theta_skip = self._set_zoom_or_skip()
        if zoom_perc is not None:
            reduced_data.attrs['zoom_perc'] = zoom_perc
        if num_theta_skip is not None:
            reduced_data.attrs['num_theta_skip'] = num_theta_skip

        # Generate reduced tomography fields
        reduced_data = self._gen_tomo(nxentry, reduced_data)

        # Create a copy of the input Nexus object and remove raw and any existing reduced data
        if isinstance(data, NXroot):
            exclude_items = [f'{nxentry._name}/reduced_data/data',
                    f'{nxentry._name}/instrument/detector/data',
                    f'{nxentry._name}/instrument/detector/image_key',
                    f'{nxentry._name}/instrument/detector/sequence_number',
                    f'{nxentry._name}/sample/rotation_angle',
                    f'{nxentry._name}/sample/x_translation',
                    f'{nxentry._name}/sample/z_translation',
                    f'{nxentry._name}/data/data',
                    f'{nxentry._name}/data/image_key',
                    f'{nxentry._name}/data/rotation_angle',
                    f'{nxentry._name}/data/x_translation',
                    f'{nxentry._name}/data/z_translation']
            nxroot = nxcopy(data, exclude_nxpaths=exclude_items)
            nxentry = nxroot[nxroot.attrs['default']]

        # Add the reduced data NXprocess
        nxentry.reduced_data = reduced_data

        nxentry.attrs['default'] = 'data'
        nxentry.data.makelink(nxentry.reduced_data.data.tomo_fields, name='reduced_data')
        nxentry.data.makelink(nxentry.reduced_data.rotation_angle, name='rotation_angle')
        nxentry.data.attrs['signal'] = 'reduced_data'
 
        return(nxroot)

    def find_centers(self, nxroot):
        """Find the calibrated center axis info
        """
        logger.info('Find the calibrated center axis info')

        if not isinstance(nxroot, NXroot):
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')
        nxentry = nxroot[nxroot.attrs['default']]
        if not isinstance(nxentry, NXentry):
            raise ValueError(f'Invalid nxentry ({nxentry})')

        # Check if reduced data is available
        if ('reduced_data' not in nxentry or 'reduced_data' not in nxentry.data):
            raise KeyError(f'Unable to find valid reduced data in {nxentry}.')

        # Select the image stack to calibrate the center axis
        #   reduced data axes order: stack,row,theta,column
        #   Note: Nexus cannot follow a link if the data it points to is too big,
        #         so get the data from the actual place, not from nxentry.data
        num_tomo_stacks = nxentry.reduced_data.data.tomo_fields.shape[0]
        if num_tomo_stacks == 1:
            center_stack_index = 0
            center_stack = np.asarray(nxentry.reduced_data.data.tomo_fields[0])
            if not center_stack.size:
                raise KeyError('Unable to load the required reduced tomography stack')
            default = 'n'
        else:
            if self.test_mode:
                center_stack_index = self.test_config['center_stack_index']-1 # make offset 0
            else:
                center_stack_index = input_int('\nEnter tomography stack index to calibrate the '
                        'center axis', ge=0, le=num_tomo_stacks-1, default=int(num_tomo_stacks/2))
            center_stack = \
                    np.asarray(nxentry.reduced_data.data.tomo_fields[center_stack_index])
            if not center_stack.size:
                raise KeyError('Unable to load the required reduced tomography stack')
            default = 'y'

        # Get thetas (in degrees)
        thetas = np.asarray(nxentry.reduced_data.rotation_angle)

        # Get effective pixel_size
        if 'zoom_perc' in nxentry.reduced_data:
            eff_pixel_size = 100.*(nxentry.instrument.detector.x_pixel_size/
                nxentry.reduced_data.attrs['zoom_perc'])
        else:
            eff_pixel_size = nxentry.instrument.detector.x_pixel_size

        # Get cross sectional diameter
        cross_sectional_dim = center_stack.shape[2]*eff_pixel_size
        logger.debug(f'cross_sectional_dim = {cross_sectional_dim}')

        # Determine center offset at sample row boundaries
        logger.info('Determine center offset at sample row boundaries')

        # Lower row center
        # center_stack order: row,theta,column
        if self.test_mode:
            lower_row = self.test_config['lower_row']
        else:
            lower_row = select_one_image_bound(center_stack[:,0,:], 0, bound=0,
                    title=f'theta={round(thetas[0], 2)+0}',
                    bound_name='row index to find lower center', default=default)
        lower_center_offset = self._find_center_one_plane(center_stack[lower_row,:,:], lower_row,
                thetas, eff_pixel_size, cross_sectional_dim, num_core=self.num_core)
        logger.debug(f'lower_row = {lower_row:.2f}')
        logger.debug(f'lower_center_offset = {lower_center_offset:.2f}')

        # Upper row center
        if self.test_mode:
            upper_row = self.test_config['upper_row']
        else:
            upper_row = select_one_image_bound(center_stack[:,0,:], 0,
                    bound=center_stack.shape[0]-1, title=f'theta={round(thetas[0], 2)+0}',
                    bound_name='row index to find upper center', default=default)
        upper_center_offset = self._find_center_one_plane(center_stack[upper_row,:,:], upper_row,
                thetas, eff_pixel_size, cross_sectional_dim, num_core=self.num_core)
        logger.debug(f'upper_row = {upper_row:.2f}')
        logger.debug(f'upper_center_offset = {upper_center_offset:.2f}')
        del center_stack

        center_config = {'lower_row': lower_row, 'lower_center_offset': lower_center_offset,
                'upper_row': upper_row, 'upper_center_offset': upper_center_offset}
        if num_tomo_stacks > 1:
            center_config['center_stack_index'] = center_stack_index+1 # save as offset 1

        # Save test data to file
        if self.test_mode:
            with open(f'{self.output_folder}/center_config.yaml', 'w') as f:
                safe_dump(center_config, f)

        return(center_config)

    def reconstruct_data(self, nxroot, center_info):
        """Reconstruct the tomography data.
        """
        logger.info('Reconstruct the tomography data')

        if not isinstance(nxroot, NXroot):
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')
        nxentry = nxroot[nxroot.attrs['default']]
        if not isinstance(nxentry, NXentry):
            raise ValueError(f'Invalid nxentry ({nxentry})')
        if not isinstance(center_info, dict):
            raise ValueError(f'Invalid parameter center_info ({center_info})')

        # Check if reduced data is available
        if ('reduced_data' not in nxentry or 'reduced_data' not in nxentry.data):
            raise KeyError(f'Unable to find valid reduced data in {nxentry}.')

        # Create an NXprocess to store image reconstruction (meta)data
#        if 'reconstructed_data' in nxentry:
#            logger.warning(f'Existing reconstructed data in {nxentry} will be overwritten.')
#            del nxentry['reconstructed_data']
#        if 'data' in nxentry and 'reconstructed_data' in nxentry.data:
#            del nxentry.data['reconstructed_data']
        nxprocess = NXprocess()

        # Get rotation axis rows and centers
        lower_row = center_info.get('lower_row')
        lower_center_offset = center_info.get('lower_center_offset')
        upper_row = center_info.get('upper_row')
        upper_center_offset = center_info.get('upper_center_offset')
        if (lower_row is None or lower_center_offset is None or upper_row is None or
                upper_center_offset is None):
            raise KeyError(f'Unable to find valid calibrated center axis info in {center_info}.')
        center_slope = (upper_center_offset-lower_center_offset)/(upper_row-lower_row)

        # Get thetas (in degrees)
        thetas = np.asarray(nxentry.reduced_data.rotation_angle)

        # Reconstruct tomography data
        #   reduced data axes order: stack,row,theta,column
        #   reconstructed data order in each stack: row/z,x,y
        #   Note: Nexus cannot follow a link if the data it points to is too big,
        #         so get the data from the actual place, not from nxentry.data
        if 'zoom_perc' in nxentry.reduced_data:
            res_title = f'{nxentry.reduced_data.attrs["zoom_perc"]}p'
        else:
            res_title = 'fullres'
        load_error = False
        num_tomo_stacks = nxentry.reduced_data.data.tomo_fields.shape[0]
        tomo_recon_stacks = num_tomo_stacks*[np.array([])]
        for i in range(num_tomo_stacks):
            tomo_stack = np.asarray(nxentry.reduced_data.data.tomo_fields[i])
            if not tomo_stack.size:
                raise KeyError(f'Unable to load tomography stack {i} for reconstruction')
            assert(0 <= lower_row < upper_row < tomo_stack.shape[0])
            center_offsets = [lower_center_offset-lower_row*center_slope,
                    upper_center_offset+(tomo_stack.shape[0]-1-upper_row)*center_slope]
            t0 = time()
            logger.debug(f'Running _reconstruct_one_tomo_stack on {self.num_core} cores ...')
            tomo_recon_stack = self._reconstruct_one_tomo_stack(tomo_stack, thetas,
                    center_offsets=center_offsets, num_core=self.num_core, algorithm='gridrec')
            logger.debug(f'... done in {time()-t0:.2f} seconds')
            logger.info(f'Reconstruction of stack {i} took {time()-t0:.2f} seconds')

            # Combine stacks
            tomo_recon_stacks[i] = tomo_recon_stack

        # Resize the reconstructed tomography data
        #   reconstructed data order in each stack: row/z,x,y
        if self.test_mode:
            x_bounds = self.test_config.get('x_bounds')
            y_bounds = self.test_config.get('y_bounds')
            z_bounds = None
        else:
            x_bounds, y_bounds, z_bounds = self._resize_reconstructed_data(tomo_recon_stacks)
        if x_bounds is None:
            x_range = (0, tomo_recon_stacks[0].shape[1])
            x_slice = int(x_range[1]/2)
        else:
            x_range = tuple(x_bounds)
            x_slice = int((x_bounds[0]+x_bounds[1])/2)
        if y_bounds is None:
            y_range = (0, tomo_recon_stacks[0].shape[2])
            y_slice = int(y_range[1]/2)
        else:
            y_range = tuple(y_bounds)
            y_slice = int((y_bounds[0]+y_bounds[1])/2)
        if z_bounds is None:
            z_range = (0, tomo_recon_stacks[0].shape[0])
            z_slice = int(z_range[1]/2)
        else:
            z_range = tuple(z_bounds)
            z_slice = int((z_bounds[0]+z_bounds[1])/2)

        # Create a few reconstructed image slice plots
        if num_tomo_stacks == 1:
            basetitle = 'recon'
        else:
            basetitle = f'recon stack {i}'
        for i, stack in enumerate(tomo_recon_stacks):
            title = f'{basetitle} {res_title} xslice{x_slice}'
            quick_imshow(stack[z_range[0]:z_range[1],x_slice,y_range[0]:y_range[1]],
                    title=title, path=self.output_folder, save_fig=self.save_figs,
                    save_only=self.save_only, block=self.block)
            title = f'{basetitle} {res_title} yslice{y_slice}'
            quick_imshow(stack[z_range[0]:z_range[1],x_range[0]:x_range[1],y_slice],
                    title=title, path=self.output_folder, save_fig=self.save_figs,
                    save_only=self.save_only, block=self.block)
            title = f'{basetitle} {res_title} zslice{z_slice}'
            quick_imshow(stack[z_slice,x_range[0]:x_range[1],y_range[0]:y_range[1]],
                    title=title, path=self.output_folder, save_fig=self.save_figs,
                    save_only=self.save_only, block=self.block)

        # Save test data to file
        #   reconstructed data order in each stack: row/z,x,y
        if self.test_mode:
            for i, stack in enumerate(tomo_recon_stacks):
                np.savetxt(f'{self.output_folder}/recon_stack_{i+1}.txt',
                        stack[z_slice,x_range[0]:x_range[1],y_range[0]:y_range[1]], fmt='%.6e')

        # Add image reconstruction to reconstructed data NXprocess
        #   reconstructed data order in each stack: row/z,x,y
        nxprocess.data = NXdata()
        nxprocess.attrs['default'] = 'data'
        for k, v in center_info.items():
            nxprocess[k] = v
        if x_bounds is not None:
            nxprocess.x_bounds = x_bounds
        if y_bounds is not None:
            nxprocess.y_bounds = y_bounds
        if z_bounds is not None:
            nxprocess.z_bounds = z_bounds
        nxprocess.data['reconstructed_data'] = np.asarray([stack[z_range[0]:z_range[1],
                x_range[0]:x_range[1],y_range[0]:y_range[1]] for stack in tomo_recon_stacks])
        nxprocess.data.attrs['signal'] = 'reconstructed_data'

        # Create a copy of the input Nexus object and remove reduced data
        exclude_items = [f'{nxentry._name}/reduced_data/data', f'{nxentry._name}/data/reduced_data']
        nxroot_copy = nxcopy(nxroot, exclude_nxpaths=exclude_items)

        # Add the reconstructed data NXprocess to the new Nexus object
        nxentry_copy = nxroot_copy[nxroot_copy.attrs['default']]
        nxentry_copy.reconstructed_data = nxprocess
        if 'data' not in nxentry_copy:
            nxentry_copy.data = NXdata()
        nxentry_copy.attrs['default'] = 'data'
        nxentry_copy.data.makelink(nxprocess.data.reconstructed_data, name='reconstructed_data')
        nxentry_copy.data.attrs['signal'] = 'reconstructed_data'

        return(nxroot_copy)

    def combine_data(self, nxroot):
        """Combine the reconstructed tomography stacks.
        """
        logger.info('Combine the reconstructed tomography stacks')

        if not isinstance(nxroot, NXroot):
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')
        nxentry = nxroot[nxroot.attrs['default']]
        if not isinstance(nxentry, NXentry):
            raise ValueError(f'Invalid nxentry ({nxentry})')

        # Check if reconstructed image data is available
        if ('reconstructed_data' not in nxentry or 'reconstructed_data' not in nxentry.data):
            raise KeyError(f'Unable to find valid reconstructed image data in {nxentry}.')

        # Create an NXprocess to store combined image reconstruction (meta)data
#        if 'combined_data' in nxentry:
#            logger.warning(f'Existing combined data in {nxentry} will be overwritten.')
#            del nxentry['combined_data']
#        if 'data' in nxentry 'combined_data' in nxentry.data:
#            del nxentry.data['combined_data']
        nxprocess = NXprocess()

        # Get the reconstructed data
        #   reconstructed data order: stack,row(z),x,y
        #   Note: Nexus cannot follow a link if the data it points to is too big,
        #         so get the data from the actual place, not from nxentry.data
        tomo_recon_stacks = np.asarray(nxentry.reconstructed_data.data.reconstructed_data)
        num_tomo_stacks = tomo_recon_stacks.shape[0]
        if num_tomo_stacks == 1:
            return(nxroot)
        t0 = time()
        logger.debug(f'Combining the reconstructed stacks ...')
        tomo_recon_combined = tomo_recon_stacks[0,:,:,:]
        if num_tomo_stacks > 2:
            tomo_recon_combined = np.concatenate([tomo_recon_combined]+
                    [tomo_recon_stacks[i,:,:,:] for i in range(1, num_tomo_stacks-1)])
        if num_tomo_stacks > 1:
            tomo_recon_combined = np.concatenate([tomo_recon_combined]+
                    [tomo_recon_stacks[num_tomo_stacks-1,:,:,:]])
        logger.debug(f'... done in {time()-t0:.2f} seconds')
        logger.info(f'Combining the reconstructed stacks took {time()-t0:.2f} seconds')

        # Resize the combined tomography data set
        #   combined data order: row/z,x,y
        if self.test_mode:
            x_bounds = None
            y_bounds = None
            z_bounds = self.test_config.get('z_bounds')
        else:
            x_bounds, y_bounds, z_bounds = self._resize_reconstructed_data(tomo_recon_combined,
                    z_only=True)
        if x_bounds is None:
            x_range = (0, tomo_recon_combined.shape[1])
            x_slice = int(x_range[1]/2)
        else:
            x_range = x_bounds
            x_slice = int((x_bounds[0]+x_bounds[1])/2)
        if y_bounds is None:
            y_range = (0, tomo_recon_combined.shape[2])
            y_slice = int(y_range[1]/2)
        else:
            y_range = y_bounds
            y_slice = int((y_bounds[0]+y_bounds[1])/2)
        if z_bounds is None:
            z_range = (0, tomo_recon_combined.shape[0])
            z_slice = int(z_range[1]/2)
        else:
            z_range = z_bounds
            z_slice = int((z_bounds[0]+z_bounds[1])/2)

        # Create a few combined image slice plots
        quick_imshow(tomo_recon_combined[z_range[0]:z_range[1],x_slice,y_range[0]:y_range[1]],
                title=f'recon combined xslice{x_slice}', path=self.output_folder,
                save_fig=self.save_figs, save_only=self.save_only, block=self.block)
        quick_imshow(tomo_recon_combined[z_range[0]:z_range[1],x_range[0]:x_range[1],y_slice],
                title=f'recon combined yslice{y_slice}', path=self.output_folder,
                save_fig=self.save_figs, save_only=self.save_only, block=self.block)
        quick_imshow(tomo_recon_combined[z_slice,x_range[0]:x_range[1],y_range[0]:y_range[1]],
                title=f'recon combined zslice{z_slice}', path=self.output_folder,
                save_fig=self.save_figs, save_only=self.save_only, block=self.block)

        # Save test data to file
        #   combined data order: row/z,x,y
        if self.test_mode:
            np.savetxt(f'{self.output_folder}/recon_combined.txt', tomo_recon_combined[
                    z_slice,x_range[0]:x_range[1],y_range[0]:y_range[1]], fmt='%.6e')

        # Add image reconstruction to reconstructed data NXprocess
        #   combined data order: row/z,x,y
        nxprocess.data = NXdata()
        nxprocess.attrs['default'] = 'data'
        if x_bounds is not None:
            nxprocess.x_bounds = x_bounds
        if y_bounds is not None:
            nxprocess.y_bounds = y_bounds
        if z_bounds is not None:
            nxprocess.z_bounds = z_bounds
        nxprocess.data['combined_data'] = tomo_recon_combined
        nxprocess.data.attrs['signal'] = 'combined_data'

        # Create a copy of the input Nexus object and remove reconstructed data
        exclude_items = [f'{nxentry._name}/reconstructed_data/data',
                f'{nxentry._name}/data/reconstructed_data']
        nxroot_copy = nxcopy(nxroot, exclude_nxpaths=exclude_items)

        # Add the combined data NXprocess to the new Nexus object
        nxentry_copy = nxroot_copy[nxroot_copy.attrs['default']]
        nxentry_copy.combined_data = nxprocess
        if 'data' not in nxentry_copy:
            nxentry_copy.data = NXdata()
        nxentry_copy.attrs['default'] = 'data'
        nxentry_copy.data.makelink(nxprocess.data.combined_data, name='combined_data')
        nxentry_copy.data.attrs['signal'] = 'combined_data'

        return(nxroot_copy)

    def _gen_dark(self, nxentry):
        """Generate dark field.
        """
        # Get the dark field images
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key and 'data' in nxentry.instrument.detector:
            field_indices = [index for index, key in enumerate(image_key) if key == 2]
            tdf_stack = nxentry.instrument.detector.data[field_indices,:,:]
            # RV the default NXtomo form does not accomodate bright or dark field stacks
        else:
            dark_field_scans = nxentry.spec_scans.dark_field
            dark_field = FlatField.construct_from_nxcollection(dark_field_scans)
            prefix = str(nxentry.instrument.detector.local_name)
            tdf_stack = dark_field.get_detector_data(prefix)
            if isinstance(tdf_stack, list):
                exit('TODO')

        # Take median
        if tdf_stack.ndim == 2:
            tdf = tdf_stack
        elif tdf_stack.ndim == 3:
            tdf = np.median(tdf_stack, axis=0)
            del tdf_stack
        else:
           raise ValueError(f'Invalid tdf_stack shape ({tdf_stack.shape})')

        # Remove dark field intensities above the cutoff
#RV        tdf_cutoff = None
        tdf_cutoff = tdf.min()+2*(np.median(tdf)-tdf.min())
        logger.debug(f'tdf_cutoff = {tdf_cutoff}')
        if tdf_cutoff is not None:
            if not is_num(tdf_cutoff, ge=0):
                logger.warning(f'Ignoring illegal value of tdf_cutoff {tdf_cutoff}')
            else:
                tdf[tdf > tdf_cutoff] = np.nan
                logger.debug(f'tdf_cutoff = {tdf_cutoff}')

        tdf_mean = np.nanmean(tdf)
        logger.debug(f'tdf_mean = {tdf_mean}')
        np.nan_to_num(tdf, copy=False, nan=tdf_mean, posinf=tdf_mean, neginf=0.)
#        if self.galaxy_flag:
#            quick_imshow(tdf, title='dark field', path='setup_pngs',
#                    save_fig=True, save_only=True)
#        elif not self.test_mode:
#            quick_imshow(tdf, title='dark field', path=self.output_folder,
#                    save_fig=self.save_figs, save_only=self.save_only)
#        quick_imshow(tdf, title='dark field', block=True)
        if not self.test_mode:
            quick_imshow(tdf, title='dark field', path=self.output_folder, save_fig=self.save_figs,
                    save_only=self.save_only)
            clear_imshow('dark field')

        # Add dark field to reduced data NXprocess
        reduced_data.data = NXdata()
        reduced_data.data['dark_field'] = tdf

        return(reduced_data)

    def _gen_bright(self, nxentry, reduced_data):
        """Generate bright field.
        """
        # Get the bright field images
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key and 'data' in nxentry.instrument.detector:
            field_indices = [index for index, key in enumerate(image_key) if key == 1]
            tbf_stack = nxentry.instrument.detector.data[field_indices,:,:]
            # RV the default NXtomo form does not accomodate bright or dark field stacks
        else:
            bright_field_scans = nxentry.spec_scans.bright_field
            bright_field = FlatField.construct_from_nxcollection(bright_field_scans)
            prefix = str(nxentry.instrument.detector.local_name)
            tbf_stack = bright_field.get_detector_data(prefix)
            if isinstance(tbf_stack, list):
                exit('TODO')

        # Take median if more than one image
        """Median or mean: It may be best to try the median because of some image 
           artifacts that arise due to crinkles in the upstream kapton tape windows 
           causing some phase contrast images to appear on the detector.
           One thing that also may be useful in a future implementation is to do a 
           brightfield adjustment on EACH frame of the tomo based on a ROI in the 
           corner of the frame where there is no sample but there is the direct X-ray 
           beam because there is frame to frame fluctuations from the incoming beam. 
           We don’t typically account for them but potentially could.
        """
        if tbf_stack.ndim == 2:
            tbf = tbf_stack
        elif tbf_stack.ndim == 3:
            tbf = np.median(tbf_stack, axis=0)
            del tbf_stack
        else:
           raise ValueError(f'Invalid tbf_stack shape ({tbf_stacks.shape})')

        # Subtract dark field
        if 'data' in reduced_data and 'dark_field' in reduced_data.data:
            tbf -= reduced_data.data.dark_field
        else:
            logger.warning('Dark field unavailable')

        # Set any non-positive values to one
        # (avoid negative bright field values for spikes in dark field)
        tbf[tbf < 1] = 1

#        if self.galaxy_flag:
#            quick_imshow(tbf, title='bright field', path='setup_pngs',
#                    save_fig=True, save_only=True)
#        elif not self.test_mode:
#            quick_imshow(tbf, title='bright field', path=self.output_folder,
#                    save_fig=self.save_figs, save_only=self.save_only)
#        quick_imshow(tbf, title='bright field', block=True)
        if not self.test_mode:
            quick_imshow(tbf, title='bright field', path=self.output_folder,
                    save_fig=self.save_figs, save_only=self.save_only)
            clear_imshow('bright field')

        # Add bright field to reduced data NXprocess
        if 'data' not in reduced_data: 
            reduced_data.data = NXdata()
        reduced_data.data['bright_field'] = tbf

        return(reduced_data)

    def _set_detector_bounds(self, nxentry, reduced_data):
        """Set vertical detector bounds for each image stack.
        Right now the range is the same for each set in the image stack.
        """
        if self.test_mode:
            return(tuple(self.test_config['img_x_bounds']))

        # Get bright field
        tbf = np.asarray(reduced_data.data.bright_field)
        tbf_shape = tbf.shape

        # Get the first tomography image and the reference heights
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key and 'data' in nxentry.instrument.detector:
            field_indices = [index for index, key in enumerate(image_key) if key == 0]
            first_image = np.asarray(nxentry.instrument.detector.data[field_indices[0],:,:])
            theta = float(nxentry.sample.rotation_angle[field_indices[0]])
            z_translation_all = nxentry.sample.z_translation[field_indices]
            z_translation_levels = sorted(list(set(z_translation_all)))
            num_tomo_stacks = len(z_translation_levels)
        else:
            tomo_field_scans = nxentry.spec_scans.tomo_fields
            tomo_fields = TomoField.construct_from_nxcollection(tomo_field_scans)
            vertical_shifts = tomo_fields.get_vertical_shifts()
            if not isinstance(vertical_shifts, list):
               vertical_shifts = [vertical_shifts]
            prefix = str(nxentry.instrument.detector.local_name)
            t0 = time()
            first_image = tomo_fields.get_detector_data(prefix, tomo_fields.scan_numbers[0], 0)
            logger.debug(f'Getting first image took {time()-t0:.2f} seconds')
            num_tomo_stacks = len(tomo_fields.scan_numbers)
            theta = tomo_fields.theta_range['start']

        # Select image bounds
        title = f'tomography image at theta={round(theta, 2)+0}'
        if nxentry.instrument.source.attrs['station'] in ('id1a3', 'id3a'):
            pixel_size = nxentry.instrument.detector.x_pixel_size
            # Try to get a fit from the bright field
            x_sum = np.sum(tbf, 1)
            x_sum_min = x_sum.min()
            x_sum_max = x_sum.max()
            fit = Fit.fit_data(x_sum, 'rectangle', x=np.array(range(len(x_sum))), form='atan',
                    guess=True)
            parameters = fit.best_values
            x_low_fit = parameters.get('center1', None)
            x_upp_fit = parameters.get('center2', None)
            sig_low = parameters.get('sigma1', None)
            sig_upp = parameters.get('sigma2', None)
            have_fit = fit.success and x_low_fit is not None and x_upp_fit is not None and \
                    sig_low is not None and sig_upp is not None and \
                    0 <= x_low_fit < x_upp_fit <= x_sum.size and \
                    (sig_low+sig_upp)/(x_upp_fit-x_low_fit) < 0.1
            if have_fit:
                # Set a 5% margin on each side
                margin = 0.05*(x_upp_fit-x_low_fit)
                x_low_fit = max(0, x_low_fit-margin)
                x_upp_fit = min(tbf_shape[0], x_upp_fit+margin)
            if num_tomo_stacks == 1:
                if have_fit:
                    # Set the default range to enclose the full fitted window
                    x_low = int(x_low_fit)
                    x_upp = int(x_upp_fit)
                else:
                    # Center a default range of 1 mm (RV: can we get this from the slits?)
                    num_x_min = int((1.0-0.5*pixel_size)/pixel_size)
                    x_low = int(0.5*(tbf_shape[0]-num_x_min))
                    x_upp = x_low+num_x_min
            else:
                # Get the default range from the reference heights
                delta_z = vertical_shifts[1]-vertical_shifts[0]
                for i in range(2, num_tomo_stacks):
                    delta_z = min(delta_z, vertical_shifts[i]-vertical_shifts[i-1])
                logger.debug(f'delta_z = {delta_z}')
                num_x_min = int((delta_z-0.5*pixel_size)/pixel_size)
                logger.debug(f'num_x_min = {num_x_min}')
                if num_x_min > tbf_shape[0]:
                    logger.warning('Image bounds and pixel size prevent seamless stacking')
                if have_fit:
                    # Center the default range relative to the fitted window
                    x_low = int(0.5*(x_low_fit+x_upp_fit-num_x_min))
                    x_upp = x_low+num_x_min
                else:
                    # Center the default range
                    x_low = int(0.5*(tbf_shape[0]-num_x_min))
                    x_upp = x_low+num_x_min
            tmp = np.copy(tbf)
            tmp_max = tmp.max()
            tmp[x_low,:] = tmp_max
            tmp[x_upp-1,:] = tmp_max
            quick_imshow(tmp, title='bright field')
            tmp = np.copy(first_image)
            tmp_max = tmp.max()
            tmp[x_low,:] = tmp_max
            tmp[x_upp-1,:] = tmp_max
            quick_imshow(tmp, title=title)
            del tmp
            quick_plot((range(x_sum.size), x_sum),
                    ([x_low, x_low], [x_sum_min, x_sum_max], 'r-'),
                    ([x_upp, x_upp], [x_sum_min, x_sum_max], 'r-'),
                    title='sum over theta and y')
            print(f'lower bound = {x_low} (inclusive)')
            print(f'upper bound = {x_upp} (exclusive)]')
            accept =  input_yesno('Accept these bounds (y/n)?', 'y')
            clear_imshow('bright field')
            clear_imshow(title)
            clear_plot('sum over theta and y')
            if accept:
                img_x_bounds = (x_low, x_upp)
            else:
                while True:
                    mask, img_x_bounds = draw_mask_1d(x_sum, title='select x data range',
                            legend='sum over theta and y')
                    if len(img_x_bounds) == 1:
                        break
                    else:
                        print(f'Choose a single connected data range')
                img_x_bounds = tuple(img_x_bounds[0])
            if (num_tomo_stacks > 1 and img_x_bounds[1]-img_x_bounds[0]+1 < 
                    int((delta_z-0.5*pixel_size)/pixel_size)):
                logger.warning('Image bounds and pixel size prevent seamless stacking')
        else:
            if num_tomo_stacks > 1:
                raise NotImplementedError('Selecting image bounds for multiple stacks on FMB')
            # For FMB: use the first tomography image to select range
            # RV: revisit if they do tomography with multiple stacks
            quick_imshow(first_image, title=title)
            x_sum = np.sum(first_image, 1)
            x_sum_min = x_sum.min()
            x_sum_max = x_sum.max()
            print('Select vertical data reduction range from first tomography image')
            img_x_bounds = select_image_bounds(first_image, 0, title=title)
            clear_imshow(title)
            if img_x_bounds is None:
                raise ValueError('Unable to select image bounds')
        logger.debug(f'img_x_bounds: {img_x_bounds}')

        x_low = img_x_bounds[0]
        x_upp = img_x_bounds[1]
        tmp = np.copy(tbf)
        tmp_max = tmp.max()
        tmp[x_low,:] = tmp_max
        tmp[x_upp-1,:] = tmp_max
        quick_imshow(tmp, title='bright field', path=self.output_folder, save_fig=self.save_figs,
                save_only=self.save_only, block=self.block)
        tmp = np.copy(first_image)
        tmp_max = tmp.max()
        tmp[x_low,:] = tmp_max
        tmp[x_upp-1,:] = tmp_max
        quick_imshow(tmp, title=title, path=self.output_folder, save_fig=self.save_figs,
                save_only=self.save_only, block=self.block)
        del tmp
        quick_plot((range(x_sum.size), x_sum),
                ([x_low, x_low], [x_sum_min, x_sum_max], 'r-'),
                ([x_upp, x_upp], [x_sum_min, x_sum_max], 'r-'),
                title='sum over theta and y', path=self.output_folder, save_fig=self.save_figs,
                save_only=self.save_only, block=self.block)
#        if not self.block:
#            clear_imshow('bright field')
#            clear_plot(title)
#            clear_plot('sum over theta and y')

        return(img_x_bounds)

    def _set_zoom_or_skip(self):
        """Set zoom and/or theta skip to reduce memory the requirement for the analysis.
        """
#        if input_yesno('\nDo you want to zoom in to reduce memory requirement (y/n)?', 'n'):
#            zoom_perc = input_int('    Enter zoom percentage', ge=1, le=100)
#        else:
#            zoom_perc = None
        zoom_perc = None
#        if input_yesno('Do you want to skip thetas to reduce memory requirement (y/n)?', 'n'):
#            num_theta_skip = input_int('    Enter the number skip theta interval', ge=0,
#                    lt=num_theta)
#        else:
#            num_theta_skip = None
        num_theta_skip = None
        logger.debug(f'zoom_perc = {zoom_perc}')
        logger.debug(f'num_theta_skip = {num_theta_skip}')

        return(zoom_perc, num_theta_skip)

    def _gen_tomo(self, nxentry, reduced_data):
        """Generate tomography fields.
        """
        # Get full bright field
        tbf = np.asarray(reduced_data.data.bright_field)
        tbf_shape = tbf.shape

        # Get image bounds
        img_x_bounds = tuple(reduced_data.get('img_x_bounds', (0, tbf_shape[0])))
        img_y_bounds = tuple(reduced_data.get('img_y_bounds', (0, tbf_shape[1])))

        # Get resized dark field
#        if 'dark_field' in data:
#            tbf = np.asarray(reduced_data.data.dark_field[
#                    img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]])
#        else:
#            logger.warning('Dark field unavailable')
#            tdf = None
        tdf = None

        # Resize bright field
        if img_x_bounds != (0, tbf.shape[0]) or img_y_bounds != (0, tbf.shape[1]):
            tbf = tbf[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]

        # Get the tomography images
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key and 'data' in nxentry.instrument.detector:
            field_indices_all = [index for index, key in enumerate(image_key) if key == 0]
            z_translation_all = nxentry.sample.z_translation[field_indices_all]
            z_translation_levels = sorted(list(set(z_translation_all)))
            num_tomo_stacks = len(z_translation_levels)
            tomo_stacks = num_tomo_stacks*[np.array([])]
            horizontal_shifts = []
            vertical_shifts = []
            thetas = None
            tomo_stacks = []
            for i, z_translation in enumerate(z_translation_levels):
                field_indices = [field_indices_all[index]
                        for index, z in enumerate(z_translation_all) if z == z_translation]
                horizontal_shift = list(set(nxentry.sample.x_translation[field_indices]))
                assert(len(horizontal_shift) == 1)
                horizontal_shifts += horizontal_shift
                vertical_shift = list(set(nxentry.sample.z_translation[field_indices]))
                assert(len(vertical_shift) == 1)
                vertical_shifts += vertical_shift
                sequence_numbers = nxentry.instrument.detector.sequence_number[field_indices]
                if thetas is None:
                    thetas = np.asarray(nxentry.sample.rotation_angle[field_indices]) \
                             [sequence_numbers]
                else:
                    assert(all(thetas[i] == nxentry.sample.rotation_angle[field_indices[index]]
                            for i, index in enumerate(sequence_numbers)))
                assert(list(set(sequence_numbers)) == [i for i in range(len(sequence_numbers))])
                if list(sequence_numbers) == [i for i in range(len(sequence_numbers))]:
                    tomo_stack = np.asarray(nxentry.instrument.detector.data[field_indices])
                else:
                    raise ValueError('Unable to load the tomography images')
                tomo_stacks.append(tomo_stack)
        else:
            tomo_field_scans = nxentry.spec_scans.tomo_fields
            tomo_fields = TomoField.construct_from_nxcollection(tomo_field_scans)
            horizontal_shifts = tomo_fields.get_horizontal_shifts()
            vertical_shifts = tomo_fields.get_vertical_shifts()
            prefix = str(nxentry.instrument.detector.local_name)
            t0 = time()
            tomo_stacks = tomo_fields.get_detector_data(prefix)
            logger.debug(f'Getting tomography images took {time()-t0:.2f} seconds')
            logger.debug(f'Getting all images took {time()-t0:.2f} seconds')
            thetas = np.linspace(tomo_fields.theta_range['start'], tomo_fields.theta_range['end'],
                    tomo_fields.theta_range['num'])
            if not isinstance(tomo_stacks, list):
                horizontal_shifts = [horizontal_shifts]
                vertical_shifts = [vertical_shifts]
                tomo_stacks = [tomo_stacks]

        reduced_tomo_stacks = []
        for i, tomo_stack in enumerate(tomo_stacks):
            # Resize the tomography images
            # Right now the range is the same for each set in the image stack.
            if img_x_bounds != (0, tbf.shape[0]) or img_y_bounds != (0, tbf.shape[1]):
                t0 = time()
                tomo_stack = tomo_stack[:,img_x_bounds[0]:img_x_bounds[1],
                        img_y_bounds[0]:img_y_bounds[1]].astype('float64')
                logger.debug(f'Resizing tomography images took {time()-t0:.2f} seconds')

            # Subtract dark field
            if tdf is not None:
                t0 = time()
                with set_numexpr_threads(self.num_core):
                    ne.evaluate('tomo_stack-tdf', out=tomo_stack)
                logger.debug(f'Subtracting dark field took {time()-t0:.2f} seconds')

            # Normalize
            t0 = time()
            with set_numexpr_threads(self.num_core):
                ne.evaluate('tomo_stack/tbf', out=tomo_stack, truediv=True)
            logger.debug(f'Normalizing took {time()-t0:.2f} seconds')

            # Remove non-positive values and linearize data
            t0 = time()
            cutoff = 1.e-6
            with set_numexpr_threads(self.num_core):
                ne.evaluate('where(tomo_stack<cutoff, cutoff, tomo_stack)', out=tomo_stack)
            with set_numexpr_threads(self.num_core):
                ne.evaluate('-log(tomo_stack)', out=tomo_stack)
            logger.debug('Removing non-positive values and linearizing data took '+
                    f'{time()-t0:.2f} seconds')

            # Get rid of nans/infs that may be introduced by normalization
            t0 = time()
            np.where(np.isfinite(tomo_stack), tomo_stack, 0.)
            logger.debug(f'Remove nans/infs took {time()-t0:.2f} seconds')

            # Downsize tomography stack to smaller size
            # TODO use theta_skip as well
            tomo_stack = tomo_stack.astype('float32')
            if not self.test_mode:
                if len(tomo_stacks) == 1:
                    title = f'red fullres theta {round(thetas[0], 2)+0}'
                else:
                    title = f'red stack {i} fullres theta {round(thetas[0], 2)+0}'
                quick_imshow(tomo_stack[0,:,:], title=title, path=self.output_folder,
                        save_fig=self.save_figs, save_only=self.save_only, block=self.block)
#                if not self.block:
#                    clear_imshow(title)
            if False and zoom_perc != 100:
                t0 = time()
                logger.debug(f'Zooming in ...')
                tomo_zoom_list = []
                for j in range(tomo_stack.shape[0]):
                    tomo_zoom = spi.zoom(tomo_stack[j,:,:], 0.01*zoom_perc)
                    tomo_zoom_list.append(tomo_zoom)
                tomo_stack = np.stack([tomo_zoom for tomo_zoom in tomo_zoom_list])
                logger.debug(f'... done in {time()-t0:.2f} seconds')
                logger.info(f'Zooming in took {time()-t0:.2f} seconds')
                del tomo_zoom_list
                if not self.test_mode:
                    title = f'red stack {zoom_perc}p theta {round(thetas[0], 2)+0}'
                    quick_imshow(tomo_stack[0,:,:], title=title, path=self.output_folder,
                            save_fig=self.save_figs, save_only=self.save_only, block=self.block)
#                    if not self.block:
#                        clear_imshow(title)

            # Convert tomography stack from theta,row,column to row,theta,column
            t0 = time()
            tomo_stack = np.swapaxes(tomo_stack, 0, 1)
            logger.debug(f'Converting coordinate order took {time()-t0:.2f} seconds')

            # Save test data to file
            if self.test_mode:
                row_index = int(tomo_stack.shape[0]/2)
                np.savetxt(f'{self.output_folder}/red_stack_{i+1}.txt', tomo_stack[row_index,:,:],
                        fmt='%.6e')

            # Combine resized stacks
            reduced_tomo_stacks.append(tomo_stack)

        # Add tomo field info to reduced data NXprocess
        reduced_data['rotation_angle'] = thetas
        reduced_data['x_translation'] = np.asarray(horizontal_shifts)
        reduced_data['z_translation'] = np.asarray(vertical_shifts)
        reduced_data.data['tomo_fields'] = np.asarray(reduced_tomo_stacks)

        if tdf is not None:
            del tdf
        del tbf

        return(reduced_data)

    def _find_center_one_plane(self, sinogram, row, thetas, eff_pixel_size, cross_sectional_dim,
            tol=0.1, num_core=1):
        """Find center for a single tomography plane.
        """
        # Try automatic center finding routines for initial value
        # sinogram index order: theta,column
        # need column,theta for iradon, so take transpose
        sinogram_T = sinogram.T
        center = sinogram.shape[1]/2

        # Try using Nghia Vo’s method
        t0 = time()
        if num_core > num_core_tomopy_limit:
            logger.debug(f'Running find_center_vo on {num_core_tomopy_limit} cores ...')
            tomo_center = tomopy.find_center_vo(sinogram, ncore=num_core_tomopy_limit)
        else:
            logger.debug(f'Running find_center_vo on {num_core} cores ...')
            tomo_center = tomopy.find_center_vo(sinogram, ncore=num_core)
        logger.debug(f'... done in {time()-t0:.2f} seconds')
        logger.info(f'Finding the center using Nghia Vo’s method took {time()-t0:.2f} seconds')
        center_offset_vo = tomo_center-center
        logger.info(f'Center at row {row} using Nghia Vo’s method = {center_offset_vo:.2f}')
        t0 = time()
        logger.debug(f'Running _reconstruct_one_plane on {self.num_core} cores ...')
        recon_plane = self._reconstruct_one_plane(sinogram_T, tomo_center, thetas,
                eff_pixel_size, cross_sectional_dim, False, num_core)
        logger.debug(f'... done in {time()-t0:.2f} seconds')
        logger.info(f'Reconstructing row {row} took {time()-t0:.2f} seconds')

        title = f'edges row{row} center offset{center_offset_vo:.2f} Vo'
        self._plot_edges_one_plane(recon_plane, title)

        # Try using phase correlation method
#        if input_yesno('Try finding center using phase correlation (y/n)?', 'n'):
#            t0 = time()
#            logger.debug(f'Running find_center_pc ...')
#            tomo_center = tomopy.find_center_pc(sinogram, sinogram, tol=0.1, rotc_guess=tomo_center)
#            error = 1.
#            while error > tol:
#                prev = tomo_center
#                tomo_center = tomopy.find_center_pc(sinogram, sinogram, tol=tol,
#                        rotc_guess=tomo_center)
#                error = np.abs(tomo_center-prev)
#            logger.debug(f'... done in {time()-t0:.2f} seconds')
#            logger.info('Finding the center using the phase correlation method took '+
#                    f'{time()-t0:.2f} seconds')
#            center_offset = tomo_center-center
#            print(f'Center at row {row} using phase correlation = {center_offset:.2f}')
#            t0 = time()
#            logger.debug(f'Running _reconstruct_one_plane on {self.num_core} cores ...')
#            recon_plane = self._reconstruct_one_plane(sinogram_T, tomo_center, thetas,
#                    eff_pixel_size, cross_sectional_dim, False, num_core)
#            logger.debug(f'... done in {time()-t0:.2f} seconds')
#            logger.info(f'Reconstructing row {row} took {time()-t0:.2f} seconds')
#
#            title = f'edges row{row} center_offset{center_offset:.2f} PC'
#            self._plot_edges_one_plane(recon_plane, title)

        # Select center location
#        if input_yesno('Accept a center location (y) or continue search (n)?', 'y'):
        if True:
#            center_offset = input_num('    Enter chosen center offset', ge=-center, le=center,
#                    default=center_offset_vo)
            center_offset = center_offset_vo
            del sinogram_T
            del recon_plane
            return float(center_offset)

        # perform center finding search
        while True:
            center_offset_low = input_int('\nEnter lower bound for center offset', ge=-center,
                    le=center)
            center_offset_upp = input_int('Enter upper bound for center offset',
                    ge=center_offset_low, le=center)
            if center_offset_upp == center_offset_low:
                center_offset_step = 1
            else:
                center_offset_step = input_int('Enter step size for center offset search', ge=1,
                        le=center_offset_upp-center_offset_low)
            num_center_offset = 1+int((center_offset_upp-center_offset_low)/center_offset_step)
            center_offsets = np.linspace(center_offset_low, center_offset_upp, num_center_offset)
            for center_offset in center_offsets:
                if center_offset == center_offset_vo:
                    continue
                t0 = time()
                logger.debug(f'Running _reconstruct_one_plane on {num_core} cores ...')
                recon_plane = self._reconstruct_one_plane(sinogram_T, center_offset+center, thetas,
                        eff_pixel_size, cross_sectional_dim, False, num_core)
                logger.debug(f'... done in {time()-t0:.2f} seconds')
                logger.info(f'Reconstructing center_offset {center_offset} took '+
                        f'{time()-t0:.2f} seconds')
                title = f'edges row{row} center_offset{center_offset:.2f}'
                self._plot_edges_one_plane(recon_plane, title)
            if input_int('\nContinue (0) or end the search (1)', ge=0, le=1):
                break

        del sinogram_T
        del recon_plane
        center_offset = input_num('    Enter chosen center offset', ge=-center, le=center)
        return float(center_offset)

    def _reconstruct_one_plane(self, tomo_plane_T, center, thetas, eff_pixel_size,
            cross_sectional_dim, plot_sinogram=True, num_core=1):
        """Invert the sinogram for a single tomography plane.
        """
        # tomo_plane_T index order: column,theta
        assert(0 <= center < tomo_plane_T.shape[0])
        center_offset = center-tomo_plane_T.shape[0]/2
        two_offset = 2*int(np.round(center_offset))
        two_offset_abs = np.abs(two_offset)
        max_rad = int(0.55*(cross_sectional_dim/eff_pixel_size)) # 10% slack to avoid edge effects
        if max_rad > 0.5*tomo_plane_T.shape[0]:
            max_rad = 0.5*tomo_plane_T.shape[0]
        dist_from_edge = max(1, int(np.floor((tomo_plane_T.shape[0]-two_offset_abs)/2.)-max_rad))
        if two_offset >= 0:
            logger.debug(f'sinogram range = [{two_offset+dist_from_edge}, {-dist_from_edge}]')
            sinogram = tomo_plane_T[two_offset+dist_from_edge:-dist_from_edge,:]
        else:
            logger.debug(f'sinogram range = [{dist_from_edge}, {two_offset-dist_from_edge}]')
            sinogram = tomo_plane_T[dist_from_edge:two_offset-dist_from_edge,:]
        if plot_sinogram:
            quick_imshow(sinogram.T, f'sinogram center offset{center_offset:.2f}', aspect='auto',
                    path=self.output_folder, save_fig=self.save_figs, save_only=self.save_only,
                    block=self.block)

        # Inverting sinogram
        t0 = time()
        recon_sinogram = iradon(sinogram, theta=thetas, circle=True)
        logger.debug(f'Inverting sinogram took {time()-t0:.2f} seconds')
        del sinogram

        # Performing Gaussian filtering and removing ring artifacts
        recon_parameters = None#self.config.get('recon_parameters')
        if recon_parameters is None:
            sigma = 1.0
            ring_width = 15
        else:
            sigma = recon_parameters.get('gaussian_sigma', 1.0)
            if not is_num(sigma, ge=0.0):
                logger.warning(f'Invalid gaussian_sigma ({sigma}) in _reconstruct_one_plane, '+
                        'set to a default value of 1.0')
                sigma = 1.0
            ring_width = recon_parameters.get('ring_width', 15)
            if not is_int(ring_width, ge=0):
                logger.warning(f'Invalid ring_width ({ring_width}) in _reconstruct_one_plane, '+
                        'set to a default value of 15')
                ring_width = 15
        t0 = time()
        recon_sinogram = spi.gaussian_filter(recon_sinogram, sigma, mode='nearest')
        recon_clean = np.expand_dims(recon_sinogram, axis=0)
        del recon_sinogram
        recon_clean = tomopy.misc.corr.remove_ring(recon_clean, rwidth=ring_width, ncore=num_core)
        logger.debug(f'Filtering and removing ring artifacts took {time()-t0:.2f} seconds')

        return recon_clean

    def _plot_edges_one_plane(self, recon_plane, title, path=None):
        vis_parameters = None#self.config.get('vis_parameters')
        if vis_parameters is None:
            weight = 0.1
        else:
            weight = vis_parameters.get('denoise_weight', 0.1)
            if not is_num(weight, ge=0.0):
                logger.warning(f'Invalid weight ({weight}) in _plot_edges_one_plane, '+
                        'set to a default value of 0.1')
                weight = 0.1
        edges = denoise_tv_chambolle(recon_plane, weight=weight)
        vmax = np.max(edges[0,:,:])
        vmin = -vmax
        if path is None:
            path = self.output_folder
        quick_imshow(edges[0,:,:], f'{title} coolwarm', path=path, cmap='coolwarm',
                save_fig=self.save_figs, save_only=self.save_only, block=self.block)
        quick_imshow(edges[0,:,:], f'{title} gray', path=path, cmap='gray', vmin=vmin, vmax=vmax,
                save_fig=self.save_figs, save_only=self.save_only, block=self.block)
        del edges

    def _reconstruct_one_tomo_stack(self, tomo_stack, thetas, center_offsets=[], num_core=1,
            algorithm='gridrec'):
        """Reconstruct a single tomography stack.
        """
        # tomo_stack order: row,theta,column
        # input thetas must be in degrees 
        # centers_offset: tomography axis shift in pixels relative to column center
        # RV should we remove stripes?
        # https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html
        # RV should we remove rings?
        # https://tomopy.readthedocs.io/en/latest/api/tomopy.misc.corr.html
        # RV: Add an option to do (extra) secondary iterations later or to do some sort of convergence test?
        if not len(center_offsets):
            centers = np.zeros((tomo_stack.shape[0]))
        elif len(center_offsets) == 2:
            centers = np.linspace(center_offsets[0], center_offsets[1], tomo_stack.shape[0])
        else:
            if center_offsets.size != tomo_stack.shape[0]:
                raise ValueError('center_offsets dimension mismatch in reconstruct_one_tomo_stack')
            centers = center_offsets
        centers += tomo_stack.shape[2]/2

        # Get reconstruction parameters
        recon_parameters = None#self.config.get('recon_parameters')
        if recon_parameters is None:
            sigma = 2.0
            secondary_iters = 0
            ring_width = 15
        else:
            sigma = recon_parameters.get('stripe_fw_sigma', 2.0)
            if not is_num(sigma, ge=0):
                logger.warning(f'Invalid stripe_fw_sigma ({sigma}) in '+
                        '_reconstruct_one_tomo_stack, set to a default value of 2.0')
                ring_width = 15
            secondary_iters = recon_parameters.get('secondary_iters', 0)
            if not is_int(secondary_iters, ge=0):
                logger.warning(f'Invalid secondary_iters ({secondary_iters}) in '+
                        '_reconstruct_one_tomo_stack, set to a default value of 0 (skip them)')
                ring_width = 0
            ring_width = recon_parameters.get('ring_width', 15)
            if not is_int(ring_width, ge=0):
                logger.warning(f'Invalid ring_width ({ring_width}) in _reconstruct_one_plane, '+
                        'set to a default value of 15')
                ring_width = 15

        # Remove horizontal stripe
        t0 = time()
        if num_core > num_core_tomopy_limit:
            logger.debug('Running remove_stripe_fw on {num_core_tomopy_limit} cores ...')
            tomo_stack = tomopy.prep.stripe.remove_stripe_fw(tomo_stack, sigma=sigma,
                    ncore=num_core_tomopy_limit)
        else:
            logger.debug(f'Running remove_stripe_fw on {num_core} cores ...')
            tomo_stack = tomopy.prep.stripe.remove_stripe_fw(tomo_stack, sigma=sigma,
                    ncore=num_core)
        logger.debug(f'... tomopy.prep.stripe.remove_stripe_fw took {time()-t0:.2f} seconds')

        # Perform initial image reconstruction
        logger.debug('Performing initial image reconstruction')
        t0 = time()
        logger.debug(f'Running recon on {num_core} cores ...')
        tomo_recon_stack = tomopy.recon(tomo_stack, np.radians(thetas), centers,
                sinogram_order=True, algorithm=algorithm, ncore=num_core)
        logger.debug(f'... done in {time()-t0:.2f} seconds')
        logger.info(f'Performing initial image reconstruction took {time()-t0:.2f} seconds')

        # Run optional secondary iterations
        if secondary_iters > 0:
            logger.debug(f'Running {secondary_iters} secondary iterations')
            #options = {'method':'SIRT_CUDA', 'proj_type':'cuda', 'num_iter':secondary_iters}
            #RV: doesn't work for me:
            #"Error: CUDA error 803: system has unsupported display driver/cuda driver combination."
            #options = {'method':'SIRT', 'proj_type':'linear', 'MinConstraint': 0, 'num_iter':secondary_iters}
            #SIRT did not finish while running overnight
            #options = {'method':'SART', 'proj_type':'linear', 'num_iter':secondary_iters}
            options = {'method':'SART', 'proj_type':'linear', 'MinConstraint': 0,
                    'num_iter':secondary_iters}
            t0 = time()
            logger.debug(f'Running recon on {num_core} cores ...')
            tomo_recon_stack  = tomopy.recon(tomo_stack, np.radians(thetas), centers,
                    init_recon=tomo_recon_stack, options=options, sinogram_order=True,
                    algorithm=tomopy.astra, ncore=num_core)
            logger.debug(f'... done in {time()-t0:.2f} seconds')
            logger.info(f'Performing secondary iterations took {time()-t0:.2f} seconds')

        # Remove ring artifacts
        t0 = time()
        tomopy.misc.corr.remove_ring(tomo_recon_stack, rwidth=ring_width, out=tomo_recon_stack,
                ncore=num_core)
        logger.debug(f'Removing ring artifacts took {time()-t0:.2f} seconds')

        return tomo_recon_stack

    def _resize_reconstructed_data(self, data, z_only=False):
        """Resize the reconstructed tomography data.
        """
        # Data order: row(z),x,y or stack,row(z),x,y
        if isinstance(data, list):
            for stack in data:
                assert(stack.ndim == 3)
            num_tomo_stacks = len(data)
            tomo_recon_stacks = data
        else:
            assert(data.ndim == 3)
            num_tomo_stacks = 1
            tomo_recon_stacks = [data]

        if z_only:
            x_bounds = None
            y_bounds = None
        else:
            # Selecting x bounds (in yz-plane)
            tomosum = 0
            [tomosum := tomosum+np.sum(tomo_recon_stacks[i], axis=(0,2))
                    for i in range(num_tomo_stacks)]
            select_x_bounds = input_yesno('\nDo you want to change the image x-bounds (y/n)?', 'y')
            if not select_x_bounds:
                x_bounds = None
            else:
                accept = False
                index_ranges = None
                while not accept:
                    mask, x_bounds = draw_mask_1d(tomosum, current_index_ranges=index_ranges,
                            title='select x data range', legend='recon stack sum yz')
                    while len(x_bounds) != 1:
                        print('Please select exactly one continuous range')
                        mask, x_bounds = draw_mask_1d(tomosum, title='select x data range',
                                legend='recon stack sum yz')
                    x_bounds = x_bounds[0]
#                    quick_plot(tomosum, vlines=x_bounds, title='recon stack sum yz')
#                    print(f'x_bounds = {x_bounds} (lower bound inclusive, upper bound '+
#                            'exclusive)')
#                    accept = input_yesno('Accept these bounds (y/n)?', 'y')
                    accept = True
            logger.debug(f'x_bounds = {x_bounds}')

            # Selecting y bounds (in xz-plane)
            tomosum = 0
            [tomosum := tomosum+np.sum(tomo_recon_stacks[i], axis=(0,1))
                    for i in range(num_tomo_stacks)]
            select_y_bounds = input_yesno('\nDo you want to change the image y-bounds (y/n)?', 'y')
            if not select_y_bounds:
                y_bounds = None
            else:
                accept = False
                index_ranges = None
                while not accept:
                    mask, y_bounds = draw_mask_1d(tomosum, current_index_ranges=index_ranges,
                            title='select x data range', legend='recon stack sum xz')
                    while len(y_bounds) != 1:
                        print('Please select exactly one continuous range')
                        mask, y_bounds = draw_mask_1d(tomosum, title='select x data range',
                                legend='recon stack sum xz')
                    y_bounds = y_bounds[0]
#                    quick_plot(tomosum, vlines=y_bounds, title='recon stack sum xz')
#                    print(f'y_bounds = {y_bounds} (lower bound inclusive, upper bound '+
#                            'exclusive)')
#                    accept = input_yesno('Accept these bounds (y/n)?', 'y')
                    accept = True
            logger.debug(f'y_bounds = {y_bounds}')

        # Selecting z bounds (in xy-plane) (only valid for a single image set)
        if num_tomo_stacks != 1:
            z_bounds = None
        else:
            tomosum = 0
            [tomosum := tomosum+np.sum(tomo_recon_stacks[i], axis=(1,2))
                    for i in range(num_tomo_stacks)]
            select_z_bounds = input_yesno('Do you want to change the image z-bounds (y/n)?', 'n')
            if not select_z_bounds:
                z_bounds = None
            else:
                accept = False
                index_ranges = None
                while not accept:
                    mask, z_bounds = draw_mask_1d(tomosum, current_index_ranges=index_ranges,
                            title='select x data range', legend='recon stack sum xy')
                    while len(z_bounds) != 1:
                        print('Please select exactly one continuous range')
                        mask, z_bounds = draw_mask_1d(tomosum, title='select x data range',
                                legend='recon stack sum xy')
                    z_bounds = z_bounds[0]
#                    quick_plot(tomosum, vlines=z_bounds, title='recon stack sum xy')
#                    print(f'z_bounds = {z_bounds} (lower bound inclusive, upper bound '+
#                            'exclusive)')
#                    accept = input_yesno('Accept these bounds (y/n)?', 'y')
                    accept = True
            logger.debug(f'z_bounds = {z_bounds}')

        return(x_bounds, y_bounds, z_bounds)


def run_tomo(input_file:str, output_file:str, modes:list[str], center_file=None, num_core=-1,
        output_folder='.', save_figs='no', test_mode=False) -> None:

    if test_mode:
        logging_format = '%(asctime)s : %(levelname)s - %(module)s : %(funcName)s - %(message)s'
        level = logging.getLevelName('INFO')
        logging.basicConfig(filename=f'{output_folder}/tomo.log', filemode='w',
                format=logging_format, level=level, force=True)
    logger.info(f'input_file = {input_file}')
    logger.info(f'center_file = {center_file}')
    logger.info(f'output_file = {output_file}')
    logger.debug(f'modes= {modes}')
    logger.debug(f'num_core= {num_core}')
    logger.info(f'output_folder = {output_folder}')
    logger.info(f'save_figs = {save_figs}')
    logger.info(f'test_mode = {test_mode}')

    # Check for correction modes
    if modes is None:
        modes = ['all']
    logger.debug(f'modes {type(modes)} = {modes}')

    # Instantiate Tomo object
    tomo = Tomo(num_core=num_core, output_folder=output_folder, save_figs=save_figs,
            test_mode=test_mode)

    # Read input file
    data = tomo.read(input_file)

    # Generate reduced tomography images
    if 'reduce_data' in modes or 'all' in modes:
        data = tomo.gen_reduced_data(data)

    # Find rotation axis centers for the tomography stacks.
    center_data = None
    if 'find_center' in modes or 'all' in modes:
        center_data = tomo.find_centers(data)

    # Reconstruct tomography stacks
    if 'reconstruct_data' in modes or 'all' in modes:
        if center_data is None:
            # Read input file
            center_data = tomo.read(center_file)
        data = tomo.reconstruct_data(data, center_data)
        center_data = None

    # Combine reconstructed tomography stacks
    if 'combine_data' in modes or 'all' in modes:
        data = tomo.combine_data(data)

    # Write output file
    if not test_mode:
        if center_data is None:
            data = tomo.write(data, output_file)
        else:
            data = tomo.write(center_data, output_file)


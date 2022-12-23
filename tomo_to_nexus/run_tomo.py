import logging
from multiprocessing import cpu_count
import numpy as np
from nexusformat.nexus import *
from time import time
try:
    import numexpr as ne
except:
    pass
try:
    import scipy.ndimage as spi
except:
    pass
try:
    from skimage.transform import iradon
except:
    pass
try:
    from skimage.restoration import denoise_tv_chambolle
except:
    pass
try:
    import tomopy
except:
    pass

from msnctools.fit import Fit
from general import illegal_value, is_int, is_index_range, input_int, input_num, input_yesno, \
        input_menu, draw_mask_1d, selectImageBounds, selectOneImageBound, clearImshow, \
        quickImshow, clearPlot, quickPlot

from .models import TOMOWorkflow
from .__version__ import __version__

num_core_tomopy_limit = 24

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
    def __init__(self, nxentry, logger, force_overwrite=False, num_core=-1):
        """Initialize with optional config input file or dictionary
        """
        self.force_overwrite = force_overwrite
        self.logger = logger
        self.num_core = num_core
        self.nxentry = nxentry
        if not isinstance(self.nxentry, tree.NXentry):
            illegal_value(nxentry, 'nxentry', 'Tomo:__init__', raise_error=True)
        if not isinstance(self.force_overwrite, bool):
            illegal_value(force_overwrite, 'force_overwrite', 'Tomo:__init__', raise_error=True)
        if self.num_core == -1:
            self.num_core = cpu_count()
        is_int(self.num_core, gt=0, raise_error=True)
        if self.num_core > cpu_count():
            logger.warning(f'num_core = {self.num_core} is larger than the number of available '
                    f'processors and reduced to {cpu_count()}')
            self.num_core= cpu_count()

    def genReducedData(self):
        """Generate the reduced tomography images.
        """
        self.logger.debug('Create the reduced tomography images')

        # Create an NXprocess to store data reduction (meta)data
        if 'reduced_data' in self.nxentry and self.force_overwrite:
            self.logger.warning(f'Existing reduced data in {self.nxentry} will be overwritten.')
            del self.nxentry['reduced_data']
        if 'reduced_data' in self.nxentry.data and self.force_overwrite:
            del self.nxentry.data['reduced_data']
        nxprocess = NXprocess()
        self.nxentry.reduced_data = nxprocess
        nxprocess.attrs['success'] = False

        # Generate dark field
        image_key = self.nxentry.instrument.detector.image_key
        dark_field_indices = [index for index, key in enumerate(image_key) if key == 2]
        if len(dark_field_indices) > 0:
            self._genDark(dark_field_indices, nxprocess)

        # Generate bright field
        bright_field_indices = [index for index, key in enumerate(image_key) if key == 1]
        self._genBright(bright_field_indices, nxprocess)

        # Get tomo field indices for each set
        tomo_field_indices_all = [index for index, key in enumerate(image_key) if key == 0]
        z_translation_all = self.nxentry.sample.z_translation[tomo_field_indices_all]
        z_translation_levels = sorted(list(set(z_translation_all)))
        tomo_field_indices = len(z_translation_levels)*[np.array([])]
        for i, z_translation in enumerate(z_translation_levels):
            tomo_field_indices[i] = [tomo_field_indices_all[index]
                    for index, z in enumerate(z_translation_all) if z == z_translation]

        # Set vertical detector bounds for image stack
        nxprocess.img_x_bounds = self._setDetectorBounds(tomo_field_indices)

        # Set zoom and/or theta skip to reduce memory the requirement
        zoom_perc, num_theta_skip = self._setZoomOrSkip(len(tomo_field_indices[0]))
        if zoom_perc is not None:
            nxprocess.attrs['zoom_perc'] = zoom_perc
        if num_theta_skip is not None:
            nxprocess.attrs['num_theta_skip'] = num_theta_skip

        # Generate reduced tomography fields
        self._genTomo(tomo_field_indices, nxprocess)

        # Succesfull data reduction
        nxprocess.attrs['success'] = True

    def findCenters(self):
        """Find the calibrated center axis info
        """
        self.logger.debug('Find the calibrated center axis info')

        # Check if reduced data is available
        if 'reduced_data' not in self.nxentry or not self.nxentry.reduced_data.attrs['success']:
            raise(KeyError(f'Unable to find valid reduced data in {self.nxentry}.'))

        # Create an NXprocess to store calibrated center axis info (meta)data
        if 'find_center' in self.nxentry and self.force_overwrite:
            self.logger.warning(f'Existing calibrated center axis info in {self.nxentry} will be '+
                    'overwritten.')
            del self.nxentry['find_center']
        nxprocess = NXprocess()
        self.nxentry.find_center = nxprocess
        nxprocess.attrs['success'] = False

        # Select the image stack to calibrate the center axis
        #   reduced data axes order: stack,row,theta,column
        #   Note: Nexus cannot follow a link if the data it points to is too big,
        #         so get the data from the actual place, not from self.nxentry.data
        num_tomo_stacks = self.nxentry.data.reduced_data.shape[0]
        if num_tomo_stacks == 1:
            center_stack_index = 0
            center_stack = np.asarray(self.nxentry.reduce_data.data.reduced_data[0])
            if not center_stack.size:
                raise KeyError('Unable to load the required reduced tomography stack')
        else:
            center_stack_index = input_int('\nEnter tomography stack index to calibrate the '
                    'center axis', ge=0, le=num_tomo_stacks-1, default=int(num_tomo_stacks/2))
            center_stack = \
                    np.asarray(self.nxentry.reduce_data.data.reduced_data[center_stack_index])
            if not center_stack.size:
                raise KeyError('Unable to load the required reduced tomography stack')

        # Get thetas (in degrees)
        thetas = np.asarray(self.nxentry.reduced_data.rotation_angle)

        # Get effective pixel_size
        if 'zoom_perc' in self.nxentry.reduced_data:
            eff_pixel_size = 100.*(self.nxentry.instrument.detector.x_pixel_size/
                self.nxentry.reduced_data.attrs['zoom_perc'])
        else:
            eff_pixel_size = self.nxentry.instrument.detector.x_pixel_size

        # Get cross sectional diameter
        cross_sectional_dim = center_stack.shape[2]*eff_pixel_size
        self.logger.debug(f'cross_sectional_dim = {cross_sectional_dim}')

        # Determine center offset at sample row boundaries
        self.logger.info('Determine center offset at sample row boundaries')

        # Lower row center
        # center_stack order: row,theta,column
        lower_row = selectOneImageBound(center_stack[:,0,:], 0, 0, title=f'theta={thetas[0]}',
                bound_name='row index to find lower center')
        lower_center_offset = self._findCenterOnePlane(center_stack[lower_row,:,:], lower_row,
                thetas, eff_pixel_size, cross_sectional_dim, num_core=self.num_core)
        self.logger.info(f'lower_center_offset = {lower_center_offset:.2f}')

        # Upper row center
        upper_row = selectOneImageBound(center_stack[:,0,:], 0, center_stack.shape[0]-1,
                title=f'theta={thetas[0]}', bound_name='row index to find upper center')
        upper_center_offset = self._findCenterOnePlane(center_stack[upper_row,:,:], upper_row,
                thetas, eff_pixel_size, cross_sectional_dim, num_core=self.num_core)
        self.logger.info(f'upper_center_offset = {upper_center_offset:.2f}')
        del center_stack

        if num_tomo_stacks > 1:
            nxprocess.center_stack_index = center_stack_index
        nxprocess.lower_row = lower_row
        nxprocess.lower_center_offset = lower_center_offset
        nxprocess.upper_row = upper_row
        nxprocess.upper_center_offset = upper_center_offset
        nxprocess.attrs['success'] = True

    def reconstructTomoStacks(self):
        """Reconstruct the tomography stacks.
        """
        logging.debug('Reconstruct the tomography stacks')

        # Check if reduced data is available
        if 'reduced_data' not in self.nxentry or not self.nxentry.reduced_data.attrs['success']:
            raise(KeyError(f'Unable to find valid reduced data in {self.nxentry}.'))

        # Check if calibrated center axis info is available
        if 'find_center' not in self.nxentry or not self.nxentry.find_center.attrs['success']:
            raise(KeyError(f'Unable to find valid calibrated center axis info in {self.nxentry}.'))

        # Create an NXprocess to store image reconstruction (meta)data
        if 'reconstructed_image_data' in self.nxentry and self.force_overwrite:
            self.logger.warning(f'Existing reconstructed image data in {self.nxentry} will be '+
                    'overwritten.')
            del self.nxentry['reconstructed_image_data']
        if 'reconstructed_images' in self.nxentry.data and self.force_overwrite:
            del self.nxentry.data['reconstructed_images']
        nxprocess = NXprocess()
        self.nxentry.reconstructed_image_data = nxprocess
        nxprocess.attrs['success'] = False

        # Get rotation axis rows and centers
        lower_row = self.nxentry.find_center.lower_row
        lower_center_offset = self.nxentry.find_center.lower_center_offset
        upper_row = self.nxentry.find_center.upper_row
        upper_center_offset = self.nxentry.find_center.upper_center_offset
        center_slope = (upper_center_offset-lower_center_offset)/(upper_row-lower_row)

        # Get thetas (in degrees)
        thetas = np.asarray(self.nxentry.reduced_data.rotation_angle)

        # Reconstruct tomo stacks
        #   reduced data axes order: stack,row,theta,column
        #   reconstructed image data order in each stack: row/z,x,y
        #   Note: Nexus cannot follow a link if the data it points to is too big,
        #         so get the data from the actual place, not from self.nxentry.data
        if 'zoom_perc' in self.nxentry.reduced_data:
            basetitle = f'recon stack {self.nxentry.reduced_data.attrs["zoom_perc"]}p'
        else:
            basetitle = 'recon stack fullres'
        load_error = False
        num_tomo_stacks = self.nxentry.data.reduced_data.shape[0]
        tomo_recon_stacks = num_tomo_stacks*[np.array([])]
        for i in range(num_tomo_stacks):
            tomo_stack = np.asarray(self.nxentry.reduced_data.data.reduced_data[i])
            if not tomo_stack.size:
                raise(KeyError(f'Unable to load tomography stack {i} for reconstruction'))
            assert(0 <= lower_row < upper_row < tomo_stack.shape[0])
            center_offsets = [lower_center_offset-lower_row*center_slope,
                    upper_center_offset+(tomo_stack.shape[0]-1-upper_row)*center_slope]
            t0 = time()
            logging.debug(f'running _reconstructOneTomoStack on {self.num_core} cores ...')
            tomo_recon_stack = self._reconstructOneTomoStack(tomo_stack, thetas,
                    center_offsets=center_offsets, num_core=self.num_core, algorithm='gridrec')
            logging.debug(f'... _reconstructOneTomoStack took {time()-t0:.2f} seconds!')
            logging.info(f'Reconstruction of stack {i} took {time()-t0:.2f} seconds!')
            x_slice = int(tomo_recon_stack.shape[0]/2)
            title = f'{basetitle} {i} xslice{x_slice}'
            quickImshow(tomo_recon_stack[x_slice,:,:], title=title)#, path=self.output_folder,
#                    save_fig=self.save_plots, save_only=self.save_plots_only)
            y_slice = int(tomo_recon_stack.shape[1]/2)
            title = f'{basetitle} {i} yslice{y_slice}'
            quickImshow(tomo_recon_stack[:,y_slice,:], title=title)#, path=self.output_folder,
#                    save_fig=self.save_plots, save_only=self.save_plots_only)
            z_slice = int(tomo_recon_stack.shape[2]/2)
            title = f'{basetitle} {i} zslice{z_slice}'
            quickImshow(tomo_recon_stack[:,:,z_slice], title=title, block=True)#, path=self.output_folder,
#                    save_fig=self.save_plots, save_only=self.save_plots_only)

            # Combine stacks
            tomo_recon_stacks[i] = tomo_recon_stack

        # Add image reconstruction to reconstructed data NXprocess
        nxdata = NXdata()
        nxprocess.data = nxdata
        nxdata['reconstructed_images'] = tomo_recon_stacks
        nxdata.attrs['signal'] = 'reconstructed_images'
        nxprocess.attrs['default'] = 'data'
        self.nxentry.data.makelink(nxprocess.data.reconstructed_images, name='reconstructed_images')
        self.nxentry.data.attrs['signal'] = 'reconstructed_images'

        # Succesfull image reconstruction
        nxprocess.attrs['success'] = True

    def combineTomoStacks(self, galaxy_param=None):
        """Combine the reconstructed tomography stacks.
        """
        # Check if reconstructed images data is available
        if ('reconstructed_image_data' not in self.nxentry or
                not self.nxentry.reconstructed_image_data.attrs['success']):
            raise(KeyError(f'Unable to find valid reconstructed image data in {self.nxentry}.'))

        # Create an NXprocess to store combined image reconstruction (meta)data
        if 'combined_image_data' in self.nxentry and self.force_overwrite:
            self.logger.warning(f'Existing combined reconstructed image data in {self.nxentry} '+
                    'will be overwritten.')
            del self.nxentry['combined_image_data']
        if 'combined_image' in self.nxentry.data and self.force_overwrite:
            del self.nxentry.data['combined_image']
        nxprocess = NXprocess()
        self.nxentry.combined_image_data = nxprocess
        nxprocess.attrs['success'] = False

        # Get the reconstructed image data stack
        #   reconstructed image stack order: stack,row(z),x,y
        #   Note: Nexus cannot follow a link if the data it points to is too big,
        #         so get the data from the actual place, not from self.nxentry.data
        tomo_recon_stacks = \
                np.asarray(self.nxentry.reconstructed_image_data.data.reconstructed_images)
        num_tomo_stacks = tomo_recon_stacks.shape[0]

        # Selecting x bounds (in yz-plane)
        tomosum = 0
        [tomosum := tomosum+np.sum(tomo_recon_stacks[i], axis=(0,2))
                for i in range(num_tomo_stacks)]
        if not input_yesno('\nDo you want to change the image x-bounds (y/n)?', 'y'):
            x_bounds = [0, tomosum.size]
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
                x_bounds = list(x_bounds[0])
                quickPlot(tomosum, vlines=x_bounds, title='recon stack sum yz')
                print(f'x_bounds = {x_bounds} (lower bound inclusive, upper bound '+
                        'exclusive)')
                accept = input_yesno('Accept these bounds (y/n)?', 'y')
        logging.info(f'x_bounds = {x_bounds}')

        # Selecting y bounds (in xz-plane)
        tomosum = 0
        [tomosum := tomosum+np.sum(tomo_recon_stacks[i], axis=(0,1))
                for i in range(num_tomo_stacks)]
        if not input_yesno('\nDo you want to change the image y-bounds (y/n)?', 'y'):
            y_bounds = [0, tomosum.size]
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
                y_bounds = list(y_bounds[0])
                quickPlot(tomosum, vlines=y_bounds, title='recon stack sum xz')
                print(f'y_bounds = {y_bounds} (lower bound inclusive, upper bound '+
                        'exclusive)')
                accept = input_yesno('Accept these bounds (y/n)?', 'y')
        logging.info(f'y_bounds = {y_bounds}')

        # Combine reconstructed tomography stacks
        logging.info(f'Combining reconstructed stacks ...')
        t0 = time()
        tomo_recon_combined = tomo_recon_stacks[0][:,x_bounds[0]:x_bounds[1],
                y_bounds[0]:y_bounds[1]]
        if num_tomo_stacks > 2:
            tomo_recon_combined = np.concatenate([tomo_recon_combined]+
                    [tomo_recon_stacks[i][:,x_bounds[0]:x_bounds[1],y_bounds[0]:y_bounds[1]]
                    for i in range(1, num_tomo_stacks-1)])
        if num_tomo_stacks > 1:
            tomo_recon_combined = np.concatenate([tomo_recon_combined]+
                    [tomo_recon_stacks[num_tomo_stacks-1][:,x_bounds[0]:x_bounds[1],
                    y_bounds[0]:y_bounds[1]]])
        logging.info(f'... done in {time()-t0:.2f} seconds!')

        # Selecting z bounds (in xy-plane)
        tomosum = np.sum(tomo_recon_combined, axis=(1,2))
        if not input_yesno('\nDo you want to change the image z-bounds (y/n)?', 'n'):
            z_bounds = [0, tomosum.size]
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
                z_bounds = list(z_bounds[0])
                quickPlot(tomosum, vlines=z_bounds, title='recon stack sum xy')
                print(f'z_bounds = {z_bounds} (lower bound inclusive, upper bound '+
                        'exclusive)')
                accept = input_yesno('Accept these bounds (y/n)?', 'y')
        logging.info(f'z_bounds = {z_bounds}')

        path = '.'#self.output_folder
        save_fig = False#self.save_plots
        save_only = False#self.save_plots_only
        quickImshow(tomo_recon_combined[int(tomo_recon_combined.shape[0]/2),:,:],
                title=f'recon combined xslice{int(tomo_recon_combined.shape[0]/2)}',
                path=path, save_fig=save_fig, save_only=save_only)
        quickImshow(tomo_recon_combined[:,int(tomo_recon_combined.shape[1]/2),:],
                title=f'recon combined yslice{int(tomo_recon_combined.shape[1]/2)}',
                path=path, save_fig=save_fig, save_only=save_only)
        quickImshow(tomo_recon_combined[:,:,int(tomo_recon_combined.shape[2]/2)],
                title=f'recon combined zslice{int(tomo_recon_combined.shape[2]/2)}',
                path=path, save_fig=save_fig, save_only=save_only, block=True)

        # Add combined image reconstruction to combined image data NXprocess
        nxdata = NXdata()
        nxprocess.data = nxdata
        nxdata['combined_image'] = tomo_recon_combined
        nxdata.attrs['signal'] = 'combined_image'
        nxprocess.attrs['default'] = 'data'
        self.nxentry.data.makelink(nxprocess.data.combined_image, name='combined_image')
        self.nxentry.data.attrs['signal'] = 'combined_image'

        # Succesfull image reconstruction
        nxprocess.attrs['success'] = True


    def _genDark(self, dark_field_indices, nxprocess):
        """Generate dark field.
        """
        # Get the bright field images
        tdf_stack = self.nxentry.instrument.detector.data[dark_field_indices,:,:]

        # Take median
        tdf = np.median(tdf_stack, axis=0)
        del tdf_stack

        # Remove dark field intensities above the cutoff
        tdf_cutoff = None
        if tdf_cutoff is not None:
            if not is_num(tdf_cutoff, ge=0):
                self.logger.warning(f'Ignoring illegal value of tdf_cutoff {tdf_cutoff}')
            else:
                tdf[tdf > tdf_cutoff] = np.nan
                self.logger.debug(f'tdf_cutoff = {tdf_cutoff}')

        tdf_mean = np.nanmean(tdf)
        self.logger.debug(f'tdf_mean = {tdf_mean}')
        np.nan_to_num(tdf, copy=False, nan=tdf_mean, posinf=tdf_mean, neginf=0.)
#        if self.galaxy_flag:
#            quickImshow(tdf, title='dark field', path='setup_pngs',
#                    save_fig=True, save_only=True)
#        elif not self.test_mode:
#            quickImshow(tdf, title='dark field', path=self.output_folder,
#                    save_fig=self.save_plots, save_only=self.save_plots_only)
#        quickImshow(tdf, title='dark field', block=True)

        # Add dark field to reduced data NXprocess
        nxdata = NXdata()
        nxprocess.data = nxdata
        nxdata['dark_field'] = tdf

    def _genBright(self, bright_field_indices, nxprocess):
        """Generate bright field.
        """
        # Get the bright field images
        tbf_stack = self.nxentry.instrument.detector.data[bright_field_indices,:,:]

        # Take median
        """Median or mean: It may be best to try the median because of some image 
           artifacts that arise due to crinkles in the upstream kapton tape windows 
           causing some phase contrast images to appear on the detector.
           One thing that also may be useful in a future implementation is to do a 
           brightfield adjustment on EACH frame of the tomo based on a ROI in the 
           corner of the frame where there is no sample but there is the direct X-ray 
           beam because there is frame to frame fluctuations from the incoming beam. 
           We don’t typically account for them but potentially could.
        """
        tbf = np.median(tbf_stack, axis=0)
        del tbf_stack

        # Subtract dark field
        if 'data' in nxprocess:
            tbf -= nxprocess.data.dark_field
        else:
            self.logger.warning('Dark field unavailable')
#        if self.galaxy_flag:
#            quickImshow(tbf, title='bright field', path='setup_pngs',
#                    save_fig=True, save_only=True)
#        elif not self.test_mode:
#            quickImshow(tbf, title='bright field', path=self.output_folder,
#                    save_fig=self.save_plots, save_only=self.save_plots_only)
#        quickImshow(tbf, title='bright field', block=True)

        # Add bright field to reduced data NXprocess
        if 'data' not in nxprocess:
            nxprocess.data = NXdata()
        nxdata = nxprocess.data
        nxdata['bright_field'] = tbf

    def _setDetectorBounds(self, tomo_field_indices):
        """Set vertical detector bounds for each image stack.
        """
        # Get bright field
        tbf = np.asarray(self.nxentry.reduced_data.data.bright_field)

        # Check reference heights
        pixel_size = self.nxentry.instrument.detector.x_pixel_size
        num_x_min = None
        num_tomo_stacks = len(tomo_field_indices)
        if num_tomo_stacks > 1:
            z_translation = self.nxentry.sample.z_translation
            delta_z = z_translation[tomo_field_indices[1][0]]- \
                    z_translation[tomo_field_indices[0][0]]
            for i in range(2, num_tomo_stacks):
                delta_z = min(delta_z, z_translation[tomo_field_indices[i][0]]- \
                    z_translation[tomo_field_indices[i-1][0]])
            self.logger.debug(f'delta_z = {delta_z}')
            num_x_min = int((delta_z-0.5*pixel_size)/pixel_size)
            self.logger.debug(f'num_x_min = {num_x_min}')
            if num_x_min > tbf.shape[0]:
                self.logger.warning('Image bounds and pixel size prevent seamless stacking')
                num_x_min = None

        # Select image bounds
        title = None
        quickImshow(tbf, title='bright field')
        if num_tomo_stacks == 1:
            # For one tomography stack only: load the first image
            image = np.asarray(self.nxentry.instrument.detector.data[tomo_field_indices[0][0]])
            theta = float(self.nxentry.sample.rotation_angle[tomo_field_indices[0][0]])
            title = f'tomography image at theta={theta}'
            quickImshow(image, title=title)
            tomo_or_bright = input_menu(['bright field', 'first tomography image'],
                    header='\nSelect image bounds from')
        else:
            tomo_or_bright = 0
        if tomo_or_bright:
            x_sum = np.sum(image, 1)
            title = f'tomography image at theta={theta}'
            img_x_bounds = selectImageBounds(image, 0, num_min=num_x_min, title=title,
                    raise_error=True)
            if num_x_min is not None and img_x_bounds[1]-img_x_bounds[0]+1 < num_x_min:
                self.logger.warning('Image bounds and pixel size prevent seamless stacking')
            quickImshow(image, title=title)#, path=self.output_folder,
#                    save_fig=self.save_plots, save_only=True)
            quickPlot(range(img_x_bounds[0], img_x_bounds[1]),
                    x_sum[img_x_bounds[0]:img_x_bounds[1]],
                    title='sum over theta and y')#, path=self.output_folder,
#                    save_fig=self.save_plots, save_only=True)
        else:
            x_sum = np.sum(tbf, 1)
            x_sum_min = x_sum.min()
            x_sum_max = x_sum.max()
            use_fit = False
            fit = Fit.fit_data(x_sum, 'rectangle', x=np.array(range(len(x_sum))), form='atan',
                    guess=True)
            parameters = fit.best_values
            x_low = parameters.get('center1', None)
            x_upp = parameters.get('center2', None)
            sig_low = parameters.get('sigma1', None)
            sig_upp = parameters.get('sigma2', None)
            if (x_low is not None and x_upp is not None and sig_low is not None and
                    sig_upp is not None and 0 <= x_low < x_upp <= x_sum.size and
                    (sig_low+sig_upp)/(x_upp-x_low) < 0.1):
                if num_tomo_stacks == 1 or num_x_min is None:
                    x_low = int(x_low-(x_upp-x_low)/10)
                    x_upp = int(x_upp+(x_upp-x_low)/10)
                else:
                    x_low = int((x_low+x_upp)/2-num_x_min/2)
                    x_upp = x_low+num_x_min
                if x_low < 0:
                    x_low = 0
                if x_upp > x_sum.size:
                    x_upp = x_sum.size
                tmp = np.copy(tbf)
                tmp_max = tmp.max()
                tmp[x_low,:] = tmp_max
                tmp[x_upp-1,:] = tmp_max
                title = 'bright field'
                quickImshow(tmp, title=title)
                del tmp
                quickPlot((range(x_sum.size), x_sum),
                        ([x_low, x_low], [x_sum_min, x_sum_max], 'r-'),
                        ([x_upp, x_upp], [x_sum_min, x_sum_max], 'r-'),
                        title='sum over theta and y')
                print(f'lower bound = {x_low} (inclusive)')
                print(f'upper bound = {x_upp} (exclusive)]')
                use_fit =  input_yesno('Accept these bounds (y/n)?', 'y')
            if use_fit:
                img_x_bounds = [x_low, x_upp]
            else:
                accept = False
                while not accept:
                    mask, img_x_bounds = draw_mask_1d(x_sum, title='select x data range',
                            legend='sum over theta and y')
                    img_x_bounds = list(img_x_bounds[0])
                    quickPlot(x_sum, vlines=img_x_bounds, title='sum over theta and y')
                    print(f'img_x_bounds = {img_x_bounds} (lower bound inclusive, upper bound '+
                            'exclusive)')
                    accept = input_yesno('Accept these bounds (y/n)?', 'y')
            if num_x_min is not None and img_x_bounds[1]-img_x_bounds[0]+1 < num_x_min:
                self.logger.warning('Image bounds and pixel size prevent seamless stacking')
            quickPlot((range(x_sum.size), x_sum),
                    ([img_x_bounds[0], img_x_bounds[0]], [x_sum_min, x_sum_max], 'r-'),
                    ([img_x_bounds[1], img_x_bounds[1]], [x_sum_min, x_sum_max], 'r-'),
                    title='sum over theta and y')#, path=self.output_folder,
#                    save_fig=self.save_plots, save_only=True)
            del x_sum
        self.logger.debug(f'img_x_bounds: {img_x_bounds}')

#        if self.save_plots_only:
#            clearImshow('bright field')
#            clearPlot('sum over theta and y')
#            if title:
#                clearPlot(title)
        clearImshow('bright field')
        clearPlot('sum over theta and y')
        if title:
            clearPlot(title)

        return(img_x_bounds)

    def _setZoomOrSkip(self, num_theta):
        """Set zoom and/or theta skip to reduce memory the requirement for the analysis.
        """
        if input_yesno('\nDo you want to zoom in to reduce memory requirement (y/n)?', 'n'):
            zoom_perc = input_int('    Enter zoom percentage', ge=1, le=100)
        else:
            zoom_perc = None
        if input_yesno('Do you want to skip thetas to reduce memory requirement (y/n)?', 'n'):
            num_theta_skip = input_int('    Enter the number skip theta interval', ge=0,
                    lt=num_theta)
        else:
            num_theta_skip = None
        self.logger.debug(f'zoom_perc = {zoom_perc}')
        self.logger.debug(f'num_theta_skip = {num_theta_skip}')
        return(zoom_perc, num_theta_skip)

    def _genTomo(self, tomo_field_indices, nxprocess):
        """Generate tomography fields.
        """
        tbf_shape = nxprocess.data.bright_field.shape
        img_x_bounds = tuple(nxprocess.get('img_x_bounds', (0, tbf_shape[0])))
        img_y_bounds = tuple(nxprocess.get('img_y_bounds', (0, tbf_shape[1])))
        zoom_perc = nxprocess.attrs.get('zoom_perc', 100)
        #num_theta_skip = nxprocess.attrs.get('num_theta_skip', 0)
        if nxprocess.attrs.get('num_theta_skip', None) is not None:
            raise(ValueError('num_theta_skip is not yet implemented'))

        # Get dark field
        if 'dark_field' in nxprocess.data:
            tdf = nxprocess.data.dark_field[
                    img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]
        else:
            self.logger.warning('Dark field unavailable')
            tdf = None

        # Get bright field
        tbf = nxprocess.data.bright_field[
                img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]

        num_tomo_stacks = len(tomo_field_indices)
        tomo_stacks = num_tomo_stacks*[np.array([])]
        z_translations = []
        thetas = None
        for i, field_indices in enumerate(tomo_field_indices):
            # Check and set the relavant stack info
            z_translation = list(set(self.nxentry.sample.z_translation[field_indices]))
            assert(len(z_translation) == 1)
            z_translations += z_translation
            sequence_numbers = self.nxentry.instrument.detector.sequence_number[field_indices]
            assert(list(set(sequence_numbers)) == [i for i in range(len(sequence_numbers))])
            if thetas is None:
                thetas = np.asarray(self.nxentry.sample.rotation_angle[field_indices]) \
                         [sequence_numbers]
            else:
                assert(all(thetas[i] == self.nxentry.sample.rotation_angle[field_indices[index]]
                        for i, index in enumerate(sequence_numbers)))

            # Get the tomography images
            t0 = time()
            if list(sequence_numbers) == [i for i in range(len(sequence_numbers))]:
                tomo_stack = np.asarray(self.nxentry.instrument.detector.data[field_indices,
                        img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]])
            else:
                raise(ValueError)
            tomo_stack = tomo_stack.astype('float64')
            self.logger.debug(f'getting tomography images took {time()-t0:.2f} seconds!')

            # Subtract dark field
            if tdf is not None:
                t0 = time()
                with set_numexpr_threads(self.num_core):
                    ne.evaluate('tomo_stack-tdf', out=tomo_stack)
                self.logger.debug(f'subtracting dark field took {time()-t0:.2f} seconds!')

            # Normalize
            t0 = time()
            with set_numexpr_threads(self.num_core):
                ne.evaluate('tomo_stack/tbf', out=tomo_stack, truediv=True)
            self.logger.debug(f'normalizing took {time()-t0:.2f} seconds!')

            # Remove non-positive values and linearize data
            t0 = time()
            cutoff = 1.e-6
            with set_numexpr_threads(self.num_core):
                ne.evaluate('where(tomo_stack<cutoff, cutoff, tomo_stack)', out=tomo_stack)
            with set_numexpr_threads(self.num_core):
                ne.evaluate('-log(tomo_stack)', out=tomo_stack)
            self.logger.debug('removing non-positive values and linearizing data took '+
                    f'{time()-t0:.2f} seconds!')

            # Get rid of nans/infs that may be introduced by normalization
            t0 = time()
            np.where(np.isfinite(tomo_stack), tomo_stack, 0.)
            self.logger.debug(f'remove nans/infs took {time()-t0:.2f} seconds!')

            # Downsize tomography stack to smaller size
            # TODO use theta_skip as well
            tomo_stack = tomo_stack.astype('float32')
            #title = f'red stack fullres {index}'
            #quickImshow(tomo_stack[0,:,:], title=title)#, path=self.output_folder,
            #        save_fig=self.save_plots, save_only=self.save_plots_only)
            if zoom_perc != 100:
                t0 = time()
                self.logger.info(f'Zooming in ...')
                tomo_zoom_list = []
                for j in range(tomo_stack.shape[0]):
                    tomo_zoom = spi.zoom(tomo_stack[j,:,:], 0.01*zoom_perc)
                    tomo_zoom_list.append(tomo_zoom)
                tomo_stack = np.stack([tomo_zoom for tomo_zoom in tomo_zoom_list])
                self.logger.info(f'... done in {time()-t0:.2f} seconds!')
                del tomo_zoom_list
                #title = f'red stack {zoom_perc}p {index}'
                #quickImshow(tomo_stack[0,:,:], title=title, path=self.output_folder,
                #        save_fig=self.save_plots, save_only=self.save_plots_only)

            # Convert tomography stack from theta,row,column to row,theta,column
            t0 = time()
            tomo_stack = np.swapaxes(tomo_stack, 0, 1)
            self.logger.debug(f'converting coordinate order took {time()-t0:.2f} seconds!')

            # Combine stacks
            tomo_stacks[i] = tomo_stack

        if tdf is not None:
            del tdf
        del tbf

        # Add tomo fields to reduced data NXprocess
        nxdata = nxprocess.data
        nxdata['tomo_fields'] = tomo_stacks
        nxdata.attrs['signal'] = 'tomo_fields'
        nxprocess.attrs['default'] = 'data'
        nxprocess.z_translation = z_translations
        nxprocess.z_translation.units = self.nxentry.sample.z_translation.units
        nxprocess.rotation_angle = thetas
        nxprocess.rotation_angle.units = self.nxentry.sample.rotation_angle.units
        self.nxentry.data.makelink(nxprocess.data.tomo_fields, name='reduced_data')
        self.nxentry.data.attrs['signal'] = 'reduced_data'

    def _findCenterOnePlane(self, sinogram, row, thetas, eff_pixel_size, cross_sectional_dim,
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
            self.logger.debug(f'running find_center_vo on {num_core_tomopy_limit} cores ...')
            tomo_center = tomopy.find_center_vo(sinogram, ncore=num_core_tomopy_limit)
        else:
            self.logger.debug(f'running find_center_vo on {num_core} cores ...')
            tomo_center = tomopy.find_center_vo(sinogram, ncore=num_core)
        self.logger.debug(f'... find_center_vo took {time()-t0:.2f} seconds!')
        center_offset_vo = tomo_center-center
        print(f'Center at row {row} using Nghia Vo’s method = {center_offset_vo:.2f}')
        recon_plane = self._reconstructOnePlane(sinogram_T, tomo_center, thetas,
                eff_pixel_size, cross_sectional_dim, False, num_core)
        title = f'edges row{row} center offset{center_offset_vo:.2f} Vo'
        self._plotEdgesOnePlane(recon_plane, title)

        # Try using phase correlation method
        if input_yesno('Try finding center using phase correlation (y/n)?', 'n'):
            t0 = time()
            tomo_center = tomopy.find_center_pc(sinogram, sinogram, tol=0.1, rotc_guess=tomo_center)
            error = 1.
            while error > tol:
                prev = tomo_center
                tomo_center = tomopy.find_center_pc(sinogram, sinogram, tol=tol,
                        rotc_guess=tomo_center)
                error = np.abs(tomo_center-prev)
            self.logger.debug(f'... find_center_pc took {time()-t0:.2f} seconds!')
            center_offset = tomo_center-center
            print(f'Center at row {row} using phase correlation = {center_offset:.2f}')
            recon_plane = self._reconstructOnePlane(sinogram_T, tomo_center, thetas,
                    eff_pixel_size, cross_sectional_dim, False, num_core)
            title = f'edges row{row} center_offset{center_offset:.2f} PC'
            self._plotEdgesOnePlane(recon_plane, title)

        # Select center location
        if input_yesno('Accept a center location (y) or continue search (n)?', 'y'):
            center_offset = input_num('    Enter chosen center offset', ge=-center, le=center,
                    default=center_offset_vo)
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
                self.logger.debug(f'running _reconstructOnePlane on {num_core} cores ...')
                recon_plane = self._reconstructOnePlane(sinogram_T, center_offset+center, thetas,
                        eff_pixel_size, cross_sectional_dim, False, num_core)
                self.logger.debug(f'... _reconstructOnePlane took {time()-t0:.2f} seconds!')
                title = f'edges row{row} center_offset{center_offset:.2f}'
                self._plotEdgesOnePlane(recon_plane, title)
            if input_int('\nContinue (0) or end the search (1)', ge=0, le=1):
                break

        del sinogram_T
        del recon_plane
        center_offset = input_num('    Enter chosen center offset', ge=-center, le=center)
        return float(center_offset)

    def _reconstructOnePlane(self, tomo_plane_T, center, thetas, eff_pixel_size,
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
            self.logger.debug(f'sinogram range = [{two_offset+dist_from_edge}, {-dist_from_edge}]')
            sinogram = tomo_plane_T[two_offset+dist_from_edge:-dist_from_edge,:]
        else:
            self.logger.debug(f'sinogram range = [{dist_from_edge}, {two_offset-dist_from_edge}]')
            sinogram = tomo_plane_T[dist_from_edge:two_offset-dist_from_edge,:]
        if plot_sinogram:
            quickImshow(sinogram.T, f'sinogram center offset{center_offset:.2f}', aspect='auto')#,
#                    path=self.output_folder, save_fig=self.save_plots,
#                    save_only=self.save_plots_only)

        # Inverting sinogram
        t0 = time()
        recon_sinogram = iradon(sinogram, theta=thetas, circle=True)
        self.logger.debug(f'inverting sinogram took {time()-t0:.2f} seconds!')
        del sinogram

        # Performing Gaussian filtering and removing ring artifacts
        recon_parameters = None#self.config.get('recon_parameters')
        if recon_parameters is None:
            sigma = 1.0
            ring_width = 15
        else:
            sigma = recon_parameters.get('gaussian_sigma', 1.0)
            if not is_num(sigma, ge=0.0):
                self.logger.warning(f'Illegal gaussian_sigma ({sigma}) in _reconstructOnePlane, '+
                        'set to a default value of 1.0')
                sigma = 1.0
            ring_width = recon_parameters.get('ring_width', 15)
            if not is_int(ring_width, ge=0):
                self.logger.warning(f'Illegal ring_width ({ring_width}) in _reconstructOnePlane, '+
                        'set to a default value of 15')
                ring_width = 15
        t0 = time()
        recon_sinogram = spi.gaussian_filter(recon_sinogram, sigma, mode='nearest')
        recon_clean = np.expand_dims(recon_sinogram, axis=0)
        del recon_sinogram
        t1 = time()
        self.logger.debug(f'running remove_ring on {num_core} cores ...')
        recon_clean = tomopy.misc.corr.remove_ring(recon_clean, rwidth=ring_width, ncore=num_core)
        self.logger.debug(f'... remove_ring took {time()-t1:.2f} seconds!')
        self.logger.debug(f'filtering and removing ring artifact took {time()-t0:.2f} seconds!')
        return recon_clean

    def _plotEdgesOnePlane(self, recon_plane, title, path=None):
        vis_parameters = None#self.config.get('vis_parameters')
        if vis_parameters is None:
            weight = 0.1
        else:
            weight = vis_parameters.get('denoise_weight', 0.1)
            if not is_num(weight, ge=0.0):
                self.logger.warning(f'Illegal weight ({weight}) in _plotEdgesOnePlane, '+
                        'set to a default value of 0.1')
                weight = 0.1
        edges = denoise_tv_chambolle(recon_plane, weight=weight)
        vmax = np.max(edges[0,:,:])
        vmin = -vmax
        if path is None:
            path='.'#self.output_folder
        quickImshow(edges[0,:,:], f'{title} coolwarm', path=path, cmap='coolwarm')#,
#                save_fig=self.save_plots, save_only=self.save_plots_only)
        quickImshow(edges[0,:,:], f'{title} gray', path=path, cmap='gray', vmin=vmin, vmax=vmax)#,
#                save_fig=self.save_plots, save_only=self.save_plots_only)
        del edges

    def _reconstructOneTomoStack(self, tomo_stack, thetas, center_offsets=[], num_core=1,
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
                raise ValueError('center_offsets dimension mismatch in reconstructOneTomoStack')
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
                logging.warning(f'Illegal stripe_fw_sigma ({sigma}) in '+
                        '_reconstructOneTomoStack, set to a default value of 2.0')
                ring_width = 15
            secondary_iters = recon_parameters.get('secondary_iters', 0)
            if not is_int(secondary_iters, ge=0):
                logging.warning(f'Illegal secondary_iters ({secondary_iters}) in '+
                        '_reconstructOneTomoStack, set to a default value of 0 (skip them)')
                ring_width = 0
            ring_width = recon_parameters.get('ring_width', 15)
            if not is_int(ring_width, ge=0):
                logging.warning(f'Illegal ring_width ({ring_width}) in _reconstructOnePlane, '+
                        'set to a default value of 15')
                ring_width = 15

        # Remove horizontal stripe
        t0 = time()
        if num_core > num_core_tomopy_limit:
            logging.debug('running remove_stripe_fw on {num_core_tomopy_limit} cores ...')
            tomo_stack = tomopy.prep.stripe.remove_stripe_fw(tomo_stack, sigma=sigma,
                    ncore=num_core_tomopy_limit)
        else:
            logging.debug(f'running remove_stripe_fw on {num_core} cores ...')
            tomo_stack = tomopy.prep.stripe.remove_stripe_fw(tomo_stack, sigma=sigma,
                    ncore=num_core)
        logging.debug(f'... tomopy.prep.stripe.remove_stripe_fw took {time()-t0:.2f} seconds!')

        # Perform initial image reconstruction
        logging.debug('performing initial image reconstruction')
        t0 = time()
        logging.debug(f'running recon on {num_core} cores ...')
        tomo_recon_stack = tomopy.recon(tomo_stack, np.radians(thetas), centers,
                sinogram_order=True, algorithm=algorithm, ncore=num_core)
        logging.debug(f'... recon took {time()-t0:.2f} seconds!')

        # Run optional secondary iterations
        if secondary_iters > 0:
            logging.debug(f'running {secondary_iters} secondary iterations')
            #options = {'method':'SIRT_CUDA', 'proj_type':'cuda', 'num_iter':secondary_iters}
            #RV: doesn't work for me:
            #"Error: CUDA error 803: system has unsupported display driver/cuda driver combination."
            #options = {'method':'SIRT', 'proj_type':'linear', 'MinConstraint': 0, 'num_iter':secondary_iters}
            #SIRT did not finish while running overnight
            #options = {'method':'SART', 'proj_type':'linear', 'num_iter':secondary_iters}
            options = {'method':'SART', 'proj_type':'linear', 'MinConstraint': 0,
                    'num_iter':secondary_iters}
            t0 = time()
            logging.debug(f'running recon on {num_core} cores ...')
            tomo_recon_stack  = tomopy.recon(tomo_stack, np.radians(thetas), centers,
                    init_recon=tomo_recon_stack, options=options, sinogram_order=True,
                    algorithm=tomopy.astra, ncore=num_core)
            logging.debug(f'... recon took {time()-t0:.2f} seconds!')

        # Remove ring artifacts
        t0 = time()
        logging.debug(f'running remove_ring on {num_core} cores ...')
        tomopy.misc.corr.remove_ring(tomo_recon_stack, rwidth=ring_width, out=tomo_recon_stack,
                ncore=num_core)
        logging.debug(f'... remove_ring took {time()-t0:.2f} seconds!')

        return tomo_recon_stack


def run_tomo(filename:str, correction_modes:list[str], force_overwrite=False,
        logger=logging.getLogger(__name__), num_core=-1) -> None:
    # Check for correction modes
    if correction_modes is None:
        correction_modes = ['all']
    logger.debug(f'correction_modes {type(correction_modes)} = {correction_modes}')

    wf = TOMOWorkflow.construct_from_nexus(filename)
    nxroot = nxload(filename, 'rw')
    nxroot.close()

    for sample_map in wf.sample_maps:
        nxentry = nxroot[sample_map.title]

        # Instantiate Tomo object
        tomo = Tomo(nxentry, logger, force_overwrite=force_overwrite, num_core=num_core)

        # Generate reduced tomography images
        if 'reduce_data' in correction_modes or 'all' in correction_modes:
            tomo.genReducedData()

        # Find rotation axis centers for the tomography stacks.
        if 'find_center' in correction_modes or 'all' in correction_modes:
            tomo.findCenters()

        # Reconstruct tomography stacks
        if 'reconstruct_image' in correction_modes or 'all' in correction_modes:
            tomo.reconstructTomoStacks()

        # Combine reconstructed tomography stacks
        if 'combine_images' in correction_modes or 'all' in correction_modes:
            tomo.combineTomoStacks()

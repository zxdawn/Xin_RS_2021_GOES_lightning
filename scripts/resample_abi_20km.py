'''
FUNCTION:
   Resample GOES-16 L1b C13 or L2 CTH CONUS data to 20 km
       and combine them into daily data.

INPUTS:
    ABI L1b C13 data
        OR_ABI-L1b-RadC-M6C13_G16_s{start_time:%Y%j%H%M%S%f}
            _e{end_time:%Y%j%H%M%S%f}_c{creation_time:%Y%j%H%M%S%f}.nc
    or ABI L2 CTH data
        OR_ABI-L2-ACHAC-M6_G16_s{start_time:%Y%j%H%M%S%f}
            _e{end_time:%Y%j%H%M%S%f}_c{creation_time:%Y%j%H%M%S%f}.nc']

OUTPUTS:
    L1 subset data
        G16_abi_C13_<%Y%m%d>.nc
    or L2 subset data
        G16_abi_HT_<%Y%m%d>.nc

UPDATES:
    Xin Zhang:
       05/23/2020: Basic
'''

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
from glob import glob
from satpy import MultiScene
from multiprocessing import Pool
from satpy.multiscene import timeseries
from skimage.measure import block_reduce

warnings.filterwarnings('ignore', category=UserWarning, append=True)
logging.getLogger('pyresample').setLevel(logging.CRITICAL)

# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

# set basic parameters
sd = '20190601'
ed = '20190930'
abi_type = 'L1'
num_pool = 3

# set data dir and saved variables
if abi_type == 'L1':
    abi_dir = '../data/GOES-16/ABI_L1'
    abi_name = 'OR_ABI-L1b-RadC-M6C13_G16_s'
    channel = 'C13'
    reader = 'abi_l1b'
    resampler = 'bucket_min'
    coarse_slice = 10  # resampled from 2 km to 20 km
elif abi_type == 'L2':
    abi_dir = '../data/GOES-16/ABI_L2'
    abi_name = 'OR_ABI-L2-ACHAC-M6_G16_s'
    channel = 'HT'
    reader = 'abi_l2_nc'
    resampler = 'bucket_max'
    coarse_slice = 2  # resampled from 10 km to 20 km

# Set up directory to save output and plots:
savedir = f'{abi_dir}_Grid/20km/'
if not os.path.exists(savedir):
    os.makedirs(savedir)


def load_abi(filenames):
    '''Load ABI data using Satpy'''
    logging.info('    Reading all data ...')
    # load MultiScene and channel data
    mscn = MultiScene.from_files(filenames, reader=reader)
    mscn.load([channel])
    blended_scene = mscn.blend(blend_function=timeseries)

    return blended_scene


def skimage_resample(da, func):
    '''Down-resample data using skimage
    https://stackoverflow.com/a/61412534/7347925
    '''
    res = block_reduce(da, block_size=(coarse_slice, coarse_slice), func=func)
    da[::coarse_slice, ::coarse_slice] = res
    da.attrs['area'] = da.area[::coarse_slice, ::coarse_slice]
    da.attrs['area'].description = '20km at nadir'

    return da[::coarse_slice, ::coarse_slice]


def resample_abi(scn):
    '''Resample scene to target area'''
    logging.info('    Resampling to target area ...')
    # get 20 km AreaDefinition
    # resample data to 20 km
    # version_1: satpy resample. there's bug ... I'll fix it when I have time
    # coarse_area = scn[channel].area[::coarse_slice, ::coarse_slice]
    # new_scn = scn.resample(coarse_area, resampler=resampler)
    # version_2:
    new_scn = scn.copy()
    if resampler == 'bucket_min':
        func = np.min
    elif resampler == 'bucket_max':
        func = np.max

    new_scn[channel] = new_scn[channel].groupby('time').apply(skimage_resample, args=(func,))

    # update the "resolution" attribute
    new_scn[channel].attrs['resolution'] = 'y: 0.000560 rad x: 0.000560 rad'

    return new_scn


def process_abi(file_pattern):
    '''Read and resample ABI data'''
    logging.info(f'Processing {file_pattern} ...')
    # get the files on the day
    filenames = glob(file_pattern)
    # get the date
    date = os.path.basename(filenames[0]).split('_')[4][1:8]

    if glob(savedir+f'*{date}*'):
        logging.info(f'    Skip {file_pattern} ...')
    else:
        scn = load_abi(filenames)
        resample_scn = resample_abi(scn)
        logging.info('    Saving ...')
        resample_scn.save_datasets(base_dir=savedir,
                                   datasets=[channel],
                                   encoding=enc,
                                   filename='{platform_shortname}_{sensor}_{name}_'+f'{date}.nc',
                                   include_lonlats=False
                                   )


# set parameters for saving netcdf
if abi_type == 'L1':
    enc = {channel: {'dtype': 'int16',
                     'add_offset': 273.15,
                     'scale_factor': 0.01,
                     '_FillValue': -999,
                     'zlib': True,
                     'complevel': 7}}
elif abi_type == 'L2':
    comp = dict(zlib=True, complevel=7)
    enc = {channel: comp}

# generate file patterns
if abi_type == 'L1':
    file_patterns = pd.date_range(sd, ed, freq='d').strftime(f'{abi_dir}/{abi_name}%Y%j*.nc')
elif abi_type == 'L2':
    file_patterns = pd.date_range(sd, ed, freq='d').strftime(f'{abi_dir}/{abi_name}%Y%j*.nc')

start_time = time.time()
# multiprocessing
pool = Pool(num_pool)
pool.map(process_abi, file_patterns)

print("--- %s seconds ---" % (time.time() - start_time))
pool.close()

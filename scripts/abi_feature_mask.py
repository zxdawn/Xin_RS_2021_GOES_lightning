'''
FUNCTION:
    Track convections using GOES-16 L1b C13 data.

INPUTS:
    - ABI L1b C13 data
        OR_ABI-L1b-RadC-M6C13_G16_s{start_time:%Y%j%H%M%S%f}
            _e{end_time:%Y%j%H%M%S%f}_c{creation_time:%Y%j%H%M%S%f}.nc

OUTPUTS:
    - features_<%Y%m%d>.nc
        data of detected features

    - masks_<%Y%m%d>.nc
        data of masks for features

UPDATES:
    Xin Zhang:
       06/01/2021: Basic
'''

import os
import time
import tobac
import logging
import numpy as np
import pandas as pd
from glob import glob
from satpy import MultiScene
from multiprocessing import Pool
from satpy.multiscene import timeseries

# Disable a few warnings:
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
warnings.filterwarnings('ignore', category=FutureWarning, append=True)
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

logging.getLogger('satpy').setLevel(logging.ERROR)
logging.getLogger('trackpy').setLevel(logging.ERROR)
logging.getLogger('tobac').setLevel(logging.ERROR)
logging.getLogger('pyproj').setLevel(logging.ERROR)

# set basic parameters
sd = '20200601'
ed = '20200930'
num_pool = 3

abi_dir = '../data/GOES-16/ABI_L1/'  # directory of ABI L1 data
abi_name = 'OR_ABI-L1b-RadC-M6C13_G16_s'  # prefix of ABI L1 data
dxy, dt = 2000, 300  # data resolution; Unit: m and s
threshold = np.arange(280, 190, -5)  # multi-thresholds for tracking
comp = dict(zlib=True, complevel=7)  # compression for saving netcdf output files

# Set up directory to save output and plots:
savedir = '../data/track_feature_mask/'
if not os.path.exists(savedir):
    os.makedirs(savedir)


def load_tbb(filenames):
    '''Read ABI C13 data using Satpy'''
    logging.debug('Read daily ABI into one Scene ...')
    mscn = MultiScene.from_files(filenames, reader='abi_l1b')
    mscn.load(['C13'])
    tbb = mscn.blend(blend_function=timeseries)['C13']
    tbb = clean_tbb(tbb).load()

    return tbb


def clean_tbb(tbb):
    '''Remove useless coords and all attrs'''
    # delete coords useless for tracking
    keep_coords = ['x', 'y', 'time']
    drop_coords = list(tbb.coords)

    for keep_coord in keep_coords:
        drop_coords.remove(keep_coord)
    tbb = tbb.drop_vars(drop_coords)
    tbb.attrs = ''

    return tbb


def feature(threshold, target='minimum',
            position_threshold='weighted_diff',
            coord_interp_kind='nearest',
            sigma_threshold=0.5,
            min_distance=0,
            n_erosion_threshold=0,
            n_min_threshold=30):
    '''Set keyword arguments for the feature detection step'''
    parameters_features = {}
    parameters_features['target'] = target

    # diff between specific value and threshold for weighting when finding the center location (instead of just mean lon/lat)
    parameters_features['position_threshold'] = position_threshold

    # we want to keep the original x/y instead of interpolated x/y
    # https://github.com/climate-processes/tobac/pull/51
    parameters_features['coord_interp_kind'] = coord_interp_kind

    # for slightly smoothing (gaussian filter)
    parameters_features['sigma_threshold'] = sigma_threshold

    # Minumum number of cells above threshold in the feature to be tracked
    # parameters_features['min_num'] = 4

    # K, step-wise threshold for feature detection
    parameters_features['threshold'] = threshold

    # minimum distance between features
    parameters_features['min_distance'] = min_distance

    # pixel erosion (for more robust results)
    parameters_features['n_erosion_threshold'] = n_erosion_threshold

    # minimum number of contiguous pixels for thresholds
    parameters_features['n_min_threshold'] = n_min_threshold

    return parameters_features


def segmentation(threshold, target='minimum', method='watershed'):
    '''Set keyword arguments for the segmentation step'''
    parameters_segmentation = {}
    parameters_segmentation['target'] = target
    parameters_segmentation['method'] = method
    # until which threshold the area is taken into account
    parameters_segmentation['threshold'] = threshold

    return parameters_segmentation


def save_data(features_tbb, masks_tbb, date):
    '''Save the results'''
    # set encoding
    encoding = {var: comp for var in features_tbb.data_vars}

    # save features
    features_name = f'features_{date}.nc'
    logging.info(f'Saving {features_name} ...')
    features_tbb.to_netcdf(path=os.path.join(savedir, features_name),
                           encoding=encoding,
                           engine='netcdf4')

    # save masks
    masks_name = f'masks_{date}.nc'
    logging.info(f'Saving {masks_name} ...')
    masks_tbb.to_netcdf(path=os.path.join(savedir, masks_name),
                        engine='netcdf4')


def track_tbb(pattern):
    '''Loop each day's data and generate feature/mask by tobac'''
    logging.info(f'Processing {pattern} ...')
    # get the files on the day
    filenames = glob(pattern)
    # get the date
    date = os.path.basename(filenames[0]).split('_')[4][1:8]

    feature_name = f'features_{date}.nc'
    if os.path.isfile(savedir+feature_name):
        logging.info(f'{feature_name} exists, Skip ...')
    else:
        # read data
        tbb = load_tbb(filenames)
        logging.debug(f'tbb data: {tbb}')

        # generate features
        logging.info(f'Perform detect features {date} ...')
        features = tobac.themes.tobac_v1.feature_detection_multithreshold(tbb, dxy,
                                                                          **parameters_features)
        logging.debug(f'features: {features}')

        # perform segmentation
        logging.info(f'Perform segmentation {date} ...')
        masks_tbb, features_tbb = tobac.themes.tobac_v1.segmentation(features, tbb, dxy,
                                                                     **parameters_segmentation)
        logging.debug(f'features_tbb: {features_tbb}')
        logging.debug(f'masks: {masks_tbb}')

        # save data
        save_data(features_tbb, masks_tbb, date)


# generate file patterns
file_patterns = pd.date_range(sd, ed, freq='d').strftime(f'{abi_dir}/{abi_name}%Y%j*.nc')

# set parameters_features
parameters_features = feature(threshold)

# set parameters_segmentation
parameters_segmentation = segmentation(np.max(threshold))

# multiprocessing
pool = Pool(num_pool)

start_time = time.time()

# download data
pool.map(track_tbb, file_patterns)
print("--- %s seconds ---" % (time.time() - start_time))
pool.close()
pool.join()

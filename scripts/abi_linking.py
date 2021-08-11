'''
FUNCTION:
    Link tracked convections.

INPUTS:
    - feature data generated from `h8_feature_mask.py`
        features_<%Y%j>.nc

    - ABI L1b C13 data
        OR_ABI-L1b-RadC-M6C13_G16_s{start_time:%Y%j%H%M%S%f}
            _e{end_time:%Y%j%H%M%S%f}_c{creation_time:%Y%j%H%M%S%f}.nc

OUTPUTS:
    data of tracks for selected months in one year
        - tracks_<%Y>.nc

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
import xarray as xr
from glob import glob
from satpy import MultiScene
from satpy.multiscene import timeseries

logging.getLogger('trackpy').setLevel(logging.ERROR)
logging.getLogger('tobac').setLevel(logging.ERROR)
# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

# set threads to max
# os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())

# set parameters
sd = '20200601'
ed = '20200630'

abi_dir = '../data/GOES-16/ABI_L1/'  # directory of ABI L1 data
feature_mask_dir = '../data/track_feature_mask/'  # directory of feature and mask data
dxy, dt = 2000, 300  # data resolution; Unit: m and s
comp = dict(zlib=True, complevel=7)  # compression for saving netcdf output files

# Set up directory to save output:
savedir = '../data/track_link/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

# keyword arguments for linking step
parameters_linking = {}

# search_range=int(dt*v_max/dxy); dt(s), v(m/s), dxy(m)
parameters_linking['v_max'] = 50
# keeps only trajectories that last for a given number of frames.
parameters_linking['stubs'] = 3
parameters_linking['order'] = 1
parameters_linking['extrapolate'] = 0
parameters_linking['memory'] = 0
parameters_linking['adaptive_stop'] = 0.2
parameters_linking['adaptive_step'] = 0.95
parameters_linking['subnetwork_size'] = 100
parameters_linking['d_min'] = 2*dxy  # twice times the grid spacing
parameters_linking['method_linking'] = 'predict'


def preproc_feature(ds):
    '''Add the time dimension/coordinate'''
    ds = ds.set_coords('time').drop_vars('index').swap_dims({'index': 'time'})
    return ds


def merge_features(feature_patterns):
    '''Read features into one Dataset'''
    logging.info('reading features ...')
    features = xr.open_mfdataset(feature_patterns,
                                 preprocess=preproc_feature,
                                 parallel=True
                                 )
    logging.debug(f'features: {features}')

    # reassign frame values for linking
    logging.info('processing features ...')
    features['frame'] = xr.DataArray(np.unique(features['frame'].time,
                                               return_inverse=True)[1],
                                     dims=['time'],
                                     coords={'time': features.time})
    # add time variable for linking
    features = features.rename_dims({'time': 'index'}).reset_coords('time')
    features['idx'] = features['frame']

    return features


def load_tbb(filenames):
    '''Read ABI C13 data using Satpy'''
    logging.debug('Read daily ABI into one Scene ...')
    mscn = MultiScene.from_files(filenames, reader='abi_l1b')
    mscn.load(['C13'])
    tbb = mscn.blend(blend_function=timeseries)['C13']
    tbb = clean_tbb(tbb).persist()

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


def filter_vars(track):
    '''
    Delte useless variables and convert into int or float32

    Example of track Dataset:

    Data variables: (12/14)
        frame            (index) float64 0.0 0.0 0.0 0.0 0.0 ... 11.0 11.0 11.0 11.0
        idx              (index) float64 0.0 0.0 0.0 0.0 0.0 ... 11.0 11.0 11.0 11.0
        hdim_1           (index) float64 2.077 11.63 20.02 ... 1.473e+03 1.365e+03
        hdim_2           (index) float64 1.338e+03 1.547e+03 ... 877.7 1.02e+03
        num              (index) float64 74.0 477.0 85.0 47.0 ... 51.0 40.0 103.0
        threshold_value  (index) float64 280.0 280.0 280.0 ... 205.0 205.0 195.0
        feature          ...
        time             ...
        timestr          (index) object '2020-06-01 00:01:13.100000' ... '2020-06...
        y                (index) float64 4.584e+06 4.565e+06 ... 1.637e+06 1.852e+06
        x                (index) float64 -9.444e+05 -5.262e+05 ... -1.583e+06
        ncells           (index) float64 2.509e+03 2.509e+03 ... 5.052e+03 5.052e+03
        cell             (index) float64 1.0 3.0 4.0 ... 5.196e+03 2.124e+03
        time_cell        (index) timedelta64[ns] 00:00:00 00:00:00 ... 00:55:00
    '''

    all_vars = track.keys()
    keep_vars = ['x', 'y', 'time', 'time_cell', 'cell', 'ncells', 'feature']
    drop_vars = list(set(all_vars) - set(keep_vars))

    track = track.drop_vars(drop_vars)
    track['cell'] = track['cell'].astype('int')
    track['feature'] = track['feature'].astype('int')

#     for key in track.keys():
#         if track[key].dtype == 'float64':
#             track[key] = track[key].astype('float32')

    return track


start_time = time.time()

# get feature filelist
feature_patterns = pd.date_range(sd, ed, freq='d').strftime(f'{feature_mask_dir}features_%Y%j.nc')
# merge features
features = merge_features(feature_patterns)

# get L1 tbb filelist
l1_patterns = pd.date_range(sd, ed, freq='d').strftime(f'{abi_dir}OR_ABI-L1b-RadC-M6C13_G16_s%Y%j*.nc')
l1_files = []
for f in l1_patterns:
    l1_files.extend(glob(f))

# get L1 data
tbb = load_tbb(l1_files)

# link convections
logging.info('linking features ...')
track = tobac.themes.tobac_v1.linking_trackpy(features, tbb, dt=dt, dxy=dxy, **parameters_linking)
track = track.where(track['cell'] >= 0, drop=True)

# filter variables
track = filter_vars(track)
savename = os.path.join(savedir, f'tracks_{sd}-{ed[4:]}.nc')

logging.info(f'saving linking results to {savename} ...')
track.to_netcdf(savename)

print("--- %s seconds ---" % (time.time() - start_time))

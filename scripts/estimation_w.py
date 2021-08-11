'''
FUNCTION:
    Estimation of vertical velocity (w) using tracked time-series TBB data.

INPUTS:
    - Link file generated by `h8_linking.py`
        tracks_<%Y%m%d>.nc

    - ABI L1b C13 data
        OR_ABI-L1b-RadC-M6C13_G16_s{start_time:%Y%j%H%M%S%f}
            _e{end_time:%Y%j%H%M%S%f}_c{creation_time:%Y%j%H%M%S%f}.nc
    - ABI L2 CTH data
        OR_ABI-L2-ACHAC-M6_G16_s{start_time:%Y%j%H%M%S%f}
            _e{end_time:%Y%j%H%M%S%f}_c{creation_time:%Y%j%H%M%S%f}.nc']

    - ERA5 pressure level data processed by `era5_lapse.py`
        era5_<%Y>.nc

OUTPUTS:
    data of tracks and w
        - tracks_<%Y%m%d>_w.nc

UPDATES:
    Xin Zhang:
       06/02/2021: Basic

NOTE:
    You need to make sure ABI L1/L2 data have the data during days as same as track data,
        otherwise the `nearest()` method will lead to wrong results.
    Briefly, `sd` and `ed` should be same as these in `abi_linking.py`.
'''

import os
import time
import logging
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import proplot as plot
from glob import glob
from satpy import MultiScene
from satpy.multiscene import timeseries

# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

# set parameters
sd = '20200601'
ed = '20200930'

track_dir = '../data/track_link/'
abi_l1_dir = '../data/GOES-16/ABI_L1/'
abi_l2_dir = '../data/GOES-16/ABI_L2/'
era5_dir = '../data/ERA5/'

def interp1d_np(data, x, xi):
    return np.interp(xi, x, data)

def load_tbb(l1_patterns):
    '''Read ABI L1 C13 data into one Scene'''
    # get filename list
    l1_files = []
    for f in l1_patterns:
        l1_files.extend(glob(f))

    mscn = MultiScene.from_files(l1_files, reader='abi_l1b')
    mscn.load(['C13'])
    tbb = mscn.blend(blend_function=timeseries)['C13']
    abi_area = tbb.area
    tbb = clean_tbb(tbb)

    return tbb, abi_area

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

def load_cth(l2_patterns):
    '''Read ABI L2 HT data into one Scene'''
    # get filename list
    l2_files = []
    for f in l2_patterns:
        l2_files.extend(glob(f))

    mscn = MultiScene.from_files(l2_files, reader='abi_l2_nc')
    mscn.load(['HT'])
    blended_scene = mscn.blend(blend_function=timeseries)
    cth = mscn.blend(blend_function=timeseries)['HT'].load()
    logging.debug(f'cth: {cth}')

    return cth

def pair_tbb(l1_patterns):
    '''pair TBB with tracks'''
    logging.info('pair TBB with tracks ...')
    # load L1 data
    tbb, abi_area = load_tbb(l1_patterns)

    # read in tracks
    tracks = xr.open_mfdataset(track_dir + f'tracks_{sd}-{ed[4:]}.nc')
    tracks = tracks.dropna('index').sortby(['cell', 'time'])

    # select the neaest tbb value
    logging.info('pair tbb to tracks ...')
    tbb_track = tbb.sel(x=tracks['x'],
                        y=tracks['y'],
                        time=tracks['time']
                       )

    # add tbb to Dataset
    tracks['tbb_track'] = tbb_track.drop_vars(['x', 'y', 'time']).load()
    tracks = tracks.reset_coords()
    logging.debug(f'tracks: {tracks}')

    return tracks.to_dataframe(), abi_area

def pair_cth(l2_patterns, df):
    '''pair CTH data with deep convections'''
    logging.info('pair CTH with deep convections ...')

    # merge L2 data into one Dataset
    cth = load_cth(l2_patterns)

    x = xr.DataArray.from_series(df['x'])
    y = xr.DataArray.from_series(df['y'])
    t = xr.DataArray.from_series(df['time'])

    cth_track = cth.sel(x=x, y=y, time=t, method='nearest')
    logging.debug(f'cth_track: {cth_track}')

    return cth_track

def pair_era5(df, cth_track, abi_area):
    '''pair ERA5 data with tracked convections '''
    logging.info('pair ERA5 with tracks')

    # get ERA5 data
    ds_era5 = xr.open_mfdataset(era5_dir+f'era5_{sd[:4]}.nc',
                                parallel=True)

    lons, lats = abi_area.get_lonlat_from_projection_coordinates(df['x'], df['y'])
    t = xr.DataArray.from_series(df['time'])
    era5_deep = ds_era5.sel(longitude=xr.DataArray(lons, dims=['index'], coords=[df['time'].index]),
                            latitude=xr.DataArray(lats, dims=['index'], coords=[df['time'].index]),
                            time=t,
                            method='nearest')

    logging.info('calculating lapse rate and w')

    # interpolate lapse rate to CTH
    lapse_rate = xr.apply_ufunc(interp1d_np,
                                era5_deep['lapse_rate'],
                                era5_deep['h'],
                                cth_track.load(),
                                input_core_dims=[["level"], ["level"], []],
                                exclude_dims=set(("level",)),
                                vectorize=True,
                                dask="parallelized",
                                output_dtypes=[cth_track.dtype],
                             )

    return lapse_rate

def clean_era5(data):
    '''Remove useless coords and all attrs'''
    # delete coords useless for tracking
    keep_coords = ['index']
    drop_coords = list(data.coords)

    for keep_coord in keep_coords:
        drop_coords.remove(keep_coord)
    data = data.drop_vars(drop_coords)
    data.attrs = ''

    return data

def subset_deep(df):
    '''subset data to deep convections'''
    logging.info('subset data to deep convections ...')

#     # select convections longer than 3 time steps
#     df = df.groupby('cell').filter(lambda x: len(x) > 3)

    # set conditions
    begin = df.groupby('cell').head(1)['tbb_track'] > 265
    mature = df.groupby('cell').min()['tbb_track'] < 240
    complete_convection = begin.values & mature.values

    # select the cells meeting the condition
    complete_cell = df.loc[begin[complete_convection].index.values]['cell'].values
    deep_convections = df.loc[df['cell'].isin(complete_cell)]

    logging.info(f'deep_convections: {deep_convections}')
#     logging.debug(f'deep_convections: {deep_convections}')

    return deep_convections

def calc_deltatbb(df):
    '''calculate tbb decreasing rate (K/s)'''
    logging.info('calculating tbb decreasing rates ...')
    df['delta_tbb'] = df[['cell', 'tbb_track']].groupby('cell').diff()
    df['delta_seconds'] = pd.to_timedelta(df[['cell', 'time_cell']].groupby('cell').diff()['time_cell']).dt.total_seconds()

    # estimate the delta_tbb rate
    # although the order of "delta_tbb" and "delta_seconds" is reverse, the division isn't affected
    df['delta_tbb'] /= df['delta_seconds']  # K/s

    return df


start_time = time.time()

# get L1 tbb filelist
l1_patterns = pd.date_range(sd, ed, freq='d').strftime(f'{abi_l1_dir}OR_ABI-L1b-RadC-M6C13_G16_s%Y%j*.nc')

# pair tbb with tracks
deep_convections, abi_area = pair_tbb(l1_patterns)

# get deep convections
# we don't need this any more, as we have the GLM flash data for deep convection
# deep_convections = subset_deep(df)

# calculate tbb decreasing rates
deep_convections = calc_deltatbb(deep_convections)

# get L2 cloud top height filelist
l2_patterns = pd.date_range(sd, ed, freq='d').strftime(f'{abi_l2_dir}OR_ABI-L2-ACHAC-M6_G16_s%Y%j*.nc')

# pair CTH with convections
cth_track = pair_cth(l2_patterns, deep_convections)
cth_track = clean_era5(cth_track)

# calculate lapse rate using ERA5 data
lapse_rate = pair_era5(deep_convections, cth_track, abi_area)
lapse_rate = clean_era5(lapse_rate)

# save to one Dataset
deep_convections = deep_convections.to_xarray()
deep_convections['lapse_rate'] = lapse_rate
deep_convections['cth_track'] = cth_track#.drop_vars(['longitude', 'latitude', 'time'])

# calculate vertical celocity
deep_convections['w'] = deep_convections['delta_tbb']/deep_convections['lapse_rate']

print("--- %s seconds ---" % (time.time() - start_time))

# save data
comp = dict(zlib=True, complevel=7)
enc = {var: comp for var in deep_convections.data_vars}

# deep_convections.load().to_netcdf(track_dir+f'tracks_{sd[:4]}_w.nc',
deep_convections.load().to_netcdf(track_dir+f'tracks_{sd}-{ed[4:]}_w.nc',
                                  engine='netcdf4',
                                  encoding=enc
                                 )

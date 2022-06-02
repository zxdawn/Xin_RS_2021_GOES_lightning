'''
FUNCTION:
   Merge resampled daily GOES-16 L1b, and L2 data into one csv file.
   The simple convection filter is also applied.

INPUTS:
    1) ABI L1 subset data processed by `resample_abi_20km.py`
        G16_abi_C13_<%Y%m%d>.nc
    2) ABI L2 subset data processed by `resample_abi_20km.py`
        G16_abi_HT_<%Y%m%d>.nc
    3) GLM L2 data processed by `make_GLM_grids.sh`
        OR_GLM-L2-GLMC-M3_G16_s{start_time:%Y%j%H%M%S%f}
            _e{end_time:%Y%j%H%M%S%f}_c{creation_time:%Y%j%H%M%S%f}.nc'
    4) track data processed by `abi_linking.py` and `estimation_w.py`
        tracks_<%Y%m%d>-<%m%d>_w.nc

OUTPUTS:
    merged data
        merged_<%Y%m%d>.csv

UPDATES:
    Xin Zhang:
       07/08/2020: Basic
'''

import os
import logging
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
import proplot as plot
import shapely.vectorized
from multiprocessing import Pool
from shapely.prepared import prep
from satpy import Scene, MultiScene
from shapely.ops import unary_union
from shapely.geometry import Polygon
from sklearn.neighbors import BallTree
from satpy.multiscene import timeseries
import cartopy.io.shapereader as shpreader
from scipy.stats import binned_statistic_2d

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

# set basic parameters
sd = '20200901'
ed = '20200930'
parallax_corr = False  # whether correct the ABI/GLM to ground

abi_l1_dir = '../data/GOES-16/ABI_L1_Grid/20km/'
abi_l1_raw_dir = '../data/GOES-16/ABI_L1/'
abi_l2_dir = '../data/GOES-16/ABI_L2_Grid/20km/'
glm_l2_dir = '../data/GOES-16/GLM_L2_Grid/20km/'
track_link_dir = '../data/track_link/'
abi_l1_name = 'G16_abi_C13_'
abi_l2_name = 'G16_abi_HT_'
glm_l2_name = 'OR_GLM-L2-GLMC-M3_G16_s'#'G16_glm_'


#Set up directory to save output and plots:
savedir = '../data/GOES-16/merged/'
# if parallax_corr:
#     savedir += '_parallax_corr'
# if not os.path.exists(savedir):
#     os.makedirs(savedir)

def preprocess(ds):
    '''drop useless time_bnd variable'''
    return ds.drop_vars(['time_bnds', 'GOES-East'])

def get_area_2km():
    f = glob(abi_l1_raw_dir+'OR_ABI-L1b-RadC-M6C13_G16_s*')[0]
    scn = Scene([f], reader='abi_l1b')
    scn.load(['C13'])
    return scn['C13'].area

def flag_convection(ds_cth, ds_tbb, ds_glm):
    '''filter data for convection
    
    Since we use the tobac to track the life of deep convection, we can skip this process.
    Back up here ...

    Kevin C. Thiel et al. (2020):

    all data points with cloud-top heights below 1 km are considered clear air and removed from the data set.
    In response, a second filter was applied to the data set that removes pixels that meet all three of these conditions:
    1) cloud-top temperatures warmer than 270 K
    2) cloud-top heights less than 4 km
    3) FED values exceeding 10 flashes per 5 min
    '''
    logging.info(f'    Filtering ...')

    cond_1 = ds_cth['HT'] >= 1
    cond_2 = ~np.logical_and.reduce((ds_tbb['C13'] > 270,
                                     ds_cth['HT'] < 4,
                                     ds_glm['flash_extent_density'] > 10))
    cond = np.logical_and(cond_1, cond_2)

    return cond.rename('conv_flag')

def nearest(group, df_merge):
    '''because of the different resolutionm, we need to get the nearest point data

    https://stackoverflow.com/questions/58893719/
        find-nearest-point-in-other-dataframe-with-a-lot-of-data

    https://towardsdatascience.com/
    using-scikit-learns-binary-trees-to-efficiently-find-latitude-and-longitude-neighbors-909979bd929b

    '''
    radius_max = 10e3  # meters
    earth_radius = 6371000  # meters

    # pick time according to `group`
    subset_merge = df_merge.xs(group.time[0].values, level='time', drop_level=False)\
                           .reset_index(['time', 'x', 'y'])

    # create BallTree for the query
    tree = BallTree(np.deg2rad(subset_merge[['lat', 'lon']]),
                    metric='haversine',
                    #leaf_size=2,
                   )

    # get the nearest index and sort by distance (ascending)
    ind_radius, dist = tree.query_radius(np.deg2rad(np.vstack([group['lat'], group['lon']])).T, 
                                   r=radius_max/earth_radius, 
                                   return_distance=True,
                                   sort_results=True)

    # pick the sortest distance (a.k.a nearest)
    nearest_idx = [ind[0] for ind in ind_radius if ind.size > 0]

    # create the mask for subset_track with lightning
    track_mask = xr.DataArray(np.array([True if ind.size > 0 else False for ind in ind_radius]),
                              dims='time',
                              coords={'time': group.time})
    # create the index array for paired subset_merge
    track_pixel = np.array([ind[0] for ind in ind_radius if ind.size > 0])

    # set variables copied from subset_track to subset_merge
    copy_vars = ['w', 'tbb_track', 'cth_track', 'cell', 'time_cell', 'lapse_rate']
    # add columns with nan value
    subset_merge[copy_vars] = np.nan

    # get the paired track data
    pair_track = group.where(track_mask, drop=True)[copy_vars].to_dataframe()

    # get the paired merge data
    pair_merge = subset_merge.iloc[track_pixel]
    subset_merge.loc[track_pixel,copy_vars] = pair_track.values
    subset_merge = subset_merge.dropna(subset=['w']).set_index('time')

    return subset_merge.to_xarray()

def combine_data(ds_cth, ds_tbb, ds_glm, ds_track, area_20km):
    '''combine all data into one DataFrame'''
    # convert into DataFrame as the `stack()` method of xarray is too slow
    df_merge = xr.merge([ds_cth, ds_tbb, ds_glm])\
                 .to_dataframe()

    # delete where no lightning happened
    df_merge.dropna(subset=['flash_extent_density'], inplace=True)

    # calcuate the lon/lat
    lons, lats = area_20km.get_lonlat_from_projection_coordinates(df_merge.index.get_level_values('x'),
                                                                  df_merge.index.get_level_values('y'))

    # assign the lon/lat
    df_merge.insert(0, 'lon', lons)
    df_merge.insert(1, 'lat', lats)

    # delete time not in the ABI L2 data
    valid_time = xr.DataArray(np.in1d(ds_track['time'],
                                      df_merge.index.get_level_values('time').unique()),
                              dims=['index'],
                              coords={'index':ds_track.index})

    # pair tracks with ABI/GLM 20 km data
    logging.info('    pair tracks with ABI/GLM 20 km data ...')
    track_load = ds_track.where(valid_time, drop=True)\
                         .swap_dims({'index': 'time'}).drop_vars(['index']).load()

    lons_2km, lats_2km = area_2km.get_lonlat_from_projection_coordinates(track_load['x'], track_load['y'])
    track_load['lon'] = xr.DataArray(lons_2km, dims=['time'], coords={'time': track_load.time})
    track_load['lat'] = xr.DataArray(lats_2km, dims=['time'], coords={'time': track_load.time})

    # get the paired nearest data and merge them together
    df_merge = track_load.groupby('time').map(nearest, args=(df_merge,)).to_dataframe()

    return df_merge

def get_geom(name, category='physical', resolution='50m'):
    # https://stackoverflow.com/questions/47894513/
    #    checking-if-a-geocoordinate-point-is-land-or-ocean-with-cartopy
    shp_fname = shpreader.natural_earth(name=name,
                                        resolution=resolution,
                                        category=category
                                       )
    if name == 'coastline':
        # https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
        geom = unary_union(list(shpreader.Reader(shp_fname).geometries())).buffer(0.5)
    else:
        geom = unary_union(list(shpreader.Reader(shp_fname).geometries()))

    # selected four regions
    US_nw = Polygon(np.array([[-125, 38], [-98, 38], [-98, 50], [-125, 50]]))
    US_ne = Polygon(np.array([[-98, 38], [-65, 38], [-65, 50], [-98, 50]]))
    US_sw = Polygon(np.array([[-125, 25], [-98, 25], [-98, 38], [-125, 38]]))
    US_se = Polygon(np.array([[-98, 25], [-65, 25], [-65, 38], [-98, 38]]))

    return prep(geom), {'US_nw':US_nw, 'US_ne':US_ne, 'US_sw':US_sw, 'US_se':US_se}

def get_mapkind(df):
    '''get the area and region of point
    
    area: 0: coast, 1: land; 2: ocean
    region: 0: nw, 1: ne; 2: sw; 3: se
    '''
    cond_coast = shapely.vectorized.contains(coast_geom, df['lon'], df['lat'])
    cond_land = shapely.vectorized.contains(land_geom, df['lon'], df['lat'])
    cond_ocean = shapely.vectorized.contains(ocean_geom, df['lon'], df['lat'])

    cond_nw = shapely.vectorized.contains(us_region['US_nw'], df['lon'], df['lat'])
    cond_ne = shapely.vectorized.contains(us_region['US_ne'], df['lon'], df['lat'])
    cond_sw = shapely.vectorized.contains(us_region['US_sw'], df['lon'], df['lat'])
    cond_se = shapely.vectorized.contains(us_region['US_se'], df['lon'], df['lat'])
    cond_outside = np.full(cond_se.shape, 1)

    area = np.argmax(np.vstack((cond_coast, cond_land, cond_ocean))==1, axis=0)
    region = np.argmax(np.vstack((cond_nw, cond_ne, cond_sw, cond_se, cond_outside))==1, axis=0)

    df['area'] = area
    df['region'] = region

    # replace the number with string
    df.replace({'area': {0: 'coast', 1: 'land', 2: 'ocean'},
                'region': {0: 'NW', 1: 'NE', 2: 'SW', 3: 'SE', 4: 'Other'}},
               inplace=True)

    return df
    
def read_data(fname_abi_l1, fname_abi_l2, fname_glm_l2, fname_track):
    '''read all GOES data and pair them together'''
    # read data
    logging.info('    Reading data ...')
    ds_track = xr.open_mfdataset(fname_track)
    ds_cth = xr.open_mfdataset(fname_abi_l2, preprocess=preprocess)
    ds_tbb = xr.open_mfdataset(fname_abi_l1, preprocess=preprocess)

    ds_glm = MultiScene.from_files(fname_glm_l2, reader='glm_l2')
    ds_glm.load(['flash_extent_density', 'minimum_flash_area', 'average_flash_area', 'total_energy'])
    ds_glm = ds_glm.blend(blend_function=timeseries).to_xarray_dataset()
    area_20km = ds_glm.area

    # pair data
    ds_tbb = ds_tbb.sel(time=ds_cth.time, method='nearest').assign_coords({'time': ds_cth.time,
                                                                        'y': ds_cth.y,
                                                                        'x': ds_cth.x})
    ds_glm = ds_glm.sel(time=ds_cth.time, method='nearest').assign_coords({'time': ds_cth.time,
                                                                        'y': ds_cth.y,
                                                                        'x': ds_cth.x})

    # convert CTH (m) to km
    ds_cth['HT'] /= 1e3
    ds_track['cth_track'] /= 1e3
    # convert total_energy (nJ) to fJ
    ds_glm['total_energy'] *= 1e6
    # convert time_cell type from milliseconds to minutes
    ds_track['time_cell'] = ds_track['time_cell'].astype('timedelta64[ns]').dt.seconds/60

    # merge data
    df_merge = combine_data(ds_cth, ds_tbb, ds_glm, ds_track, area_20km)

    # get the kind of location
    df_merge = get_mapkind(df_merge)

    return df_merge

def save_data(df, date):
    '''save data to .csv files'''
    savename = savedir+f'merged_{date}.csv'
    logging.info(f'    Saved to {savename} ...')
    f = open(savename, 'a')
    import re
    f.write(re.sub(' +', ' ',
                   '''# Units,
                   # time: %Y-%m-%dT%H:%M:%S; x: meters in GOES-16 CONUS proj; y: meters in GOES-16 CONUS proj; lon: east degree; lat: north degree;
                   # HT: km; C13: K; average_flash_area: km2; flash_extent_density: #/5min; total_energy: femtojoules; minimum_flash_area: km2;
                   # w: m/s; tbb_track: K; cth_track: km; cell: No.; time_cell: minutes; laspse_rate: K/m\n'''))

    # convert some variables to int type
    df['cell'] = df['cell'].astype('int32')
    df['time_cell'] = df['time_cell'].astype('int32')
    df.to_csv(f, header=df.columns, float_format='%.6f', date_format='%Y-%m-%dT%H:%M:%S')
    f.close()

def merge_data(file_pattern):
    '''Merge data'''
    # get the date (%Y%j)
    date = file_pattern.split('/')[-1][-10:-3]
    # get the save pattern
    save_pattern = savedir+f'merged_{date}.csv'  # f'*{sd}-{ed[4:]}*'

    if glob(save_pattern):
        logging.info(f'    {save_pattern} existed. Skip ...')
    else:
        logging.info(f'Processing for {save_pattern} ...')
        # get the files on the day
        fname_abi_l1 = glob(file_pattern)[0]
        fname_abi_l2 = glob(f'{abi_l2_dir}/{abi_l2_name}{date}.nc')[0]
        fname_glm_l2 = sorted(glob(f'{glm_l2_dir}/{glm_l2_name}{date}*.nc'))
        fname_track = glob(f'{track_link_dir}/tracks_{sd}-{ed[4:]}_w.nc')

        # get the merged data
        df_merge = read_data(fname_abi_l1, fname_abi_l2, fname_glm_l2, fname_track)
        
        # clean the data
        df_merge = df_merge[['x', 'y', 'lon', 'lat',
                             'HT', 'C13',
                             'average_flash_area', 'flash_extent_density', 'total_energy', 'minimum_flash_area',
                             'w', 'tbb_track', 'cth_track', 'cell', 'time_cell', 'lapse_rate', 'area', 'region']]

        # save data to .csv file
        save_data(df_merge, date)

file_patterns = pd.date_range(sd, ed, freq='d').strftime(f'{abi_l1_dir}/{abi_l1_name}%Y%j.nc')

area_2km = get_area_2km()
land_geom, _ = get_geom('land')
coast_geom, _ = get_geom('coastline')
ocean_geom, us_region = get_geom('ocean')

import time
start_time = time.time()

# multiprocessing
# pool = Pool(10)
# pool.map(merge_data, file_patterns)
for file_pattern in file_patterns:
    merge_data(file_pattern)
print("--- %s seconds ---" % (time.time() - start_time))

# pool.close()
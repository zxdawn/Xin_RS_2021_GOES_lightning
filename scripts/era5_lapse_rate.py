'''
FUNCTION:
    Calculate lapse rate using ERA5 data

INPUTS:
    ERA5 pressure level data

OUTPUTS:
    Merged data named era5_<%Y>.nc including interesting variables:
        - ciwc
        - h
        - lapse_rate

UPDATES:
    Xin Zhang:
       06/02/2021: Basic
'''

import time
import xarray as xr
import metpy.calc

era5_dir = '../data/ERA5/'
year = '2020'

# open data
ds = xr.open_mfdataset(era5_dir+f'era5_{year}*.nc', parallel=True)

start_time = time.time()

# calculate height
print('calculate height')
ds['h'] = metpy.calc.geopotential_to_height(ds['z']).metpy.dequantify()

# calculate lapse rate
print('calculate lapse rate')
diff_level = ds.diff('level')
ds['lapse_rate'] = (diff_level['t'] / diff_level['h'])

# remove unnecessary variables
all_vars = list(ds.keys())
kept_vars = ['ciwc', 'h', 'lapse_rate']
drop_vars = set(all_vars) - set(kept_vars)
ds = ds.drop_vars(drop_vars)

print("--- %s seconds ---" % (time.time() - start_time))

# set encoding
comp = dict(zlib=True, complevel=7)
encoding = {var: comp for var in ds.data_vars}

# save data
print(ds)
output_file = era5_dir + f'era5_{year}.nc'
ds.to_netcdf(path=output_file,
             engine='netcdf4',
             encoding=encoding)

print("--- %s seconds ---" % (time.time() - start_time))

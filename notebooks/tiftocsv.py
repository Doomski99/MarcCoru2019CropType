import subprocess

import xarray as xr


# convert input tifs to netcdf
subprocess.check_call(['gdal_translate', '-of', 'NetCDF', 'data/test/aadloun_crop.tif', 'data/test/aadloun_crop.nc'])
#subprocess.check_call(['gdal_translate', '-of', 'NetCDF', 'extent.tif', 'extent.nc'])

# rename bands to match data
#subprocess.check_call(['ncrename', '-v', 'Band1,biomass', 'data/test/biomass.nc'])
#subprocess.check_call(['ncrename', '-v', 'Band1,extent', 'extent.nc'])

# run a netcdf append command to add biomass data to the extent netcdf
#subprocess.check_call(['ncks', '-A', 'biomass.nc', 'extent.nc'])

# open extent and then write it directly to CSV
# unfortunately there doesn't seem to be a fast CLI tool for this
ds = xr.open_dataset('data/test/aadloun_crop.nc')
df = ds.to_dataframe().reset_index()
print(df.head())

df.to_csv('data/test/aadloun_crop.csv', index=False)
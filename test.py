# download test data to use with test2.py
import xarray as xr
ds_era5t=xr.open_zarr("gs://gcp-public-data-arco-era5/co/model-level-moisture.zarr-v2",consolidated=True,)
da = ds_era5t['q'][-1,...]
da.to_netcdf('test.nc')

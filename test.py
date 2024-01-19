# speed test
from netCDF4 import Dataset
from redtoreg import redtoreg
import numpy as np
import time

try:
    nc = Dataset('test.nc') # data already cached locally
except: # grab data from google cloud and cache locally
    # requires xarray with zarr, fsspec, gcsfs
    import xarray as xr
    ds_era5t=xr.open_zarr("gs://gcp-public-data-arco-era5/co/model-level-moisture.zarr-v2",consolidated=True,)
    da = ds_era5t['q'][-1,...]
    da.to_netcdf('test.nc')

q = nc['q'][:]
print(q.shape, q.min(), q.max())

lats = nc['latitude']
lats_distinct = np.unique(lats)
nlats = len(lats_distinct)
lonsperlat = np.empty(nlats, int)
for j in range(nlats):
    lonsperlat[j] = np.isclose(lats, lats_distinct[j]).sum()
print(lonsperlat)

nlons = lonsperlat.max()
nlevs = q.shape[0]
qq = np.empty((nlevs,nlats,nlons), q.dtype)

start = time.time()
for k in range(nlevs):
    qq[k] = redtoreg(q[k],lonsperlat)
end = time.time()
elapsed = end-start

print('time in redtoreg =',elapsed)
print(qq.shape, qq.min(), qq.max())

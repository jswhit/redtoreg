import xarray as xr
from redtoreg import redtoreg
import numpy as np

# tracer variables on gaussian grid.
ds_era5t=xr.open_zarr("gs://gcp-public-data-arco-era5/co/model-level-moisture.zarr-v2",consolidated=True,)
q = ds_era5t['q'][-1,0,:].values
lats = ds_era5t['latitude'][:].values
lats_distinct = np.unique(lats)
nlats = len(lats_distinct)
lonsperlat = np.empty(nlats, int)
for j in range(nlats):
    lonsperlat[j] = np.isclose(lats, lats_distinct[j]).sum()
print(lonsperlat)
qq = redtoreg(q,lonsperlat,missval=9999)
print(q.shape, q.dtype, q.min(), q.max())
print(qq.shape, qq.dtype, qq.min(), qq.max())

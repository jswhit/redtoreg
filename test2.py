# speed test
from netCDF4 import Dataset
from redtoreg import redtoreg
import numpy as np
import time

nc = Dataset('test.nc')
q = nc['q'][:]
print(q.shape)

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

print('time in redtoreg2 =',elapsed)
print(qq.shape, qq.min(), qq.max())

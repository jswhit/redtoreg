from netCDF4 import Dataset
from redtoreg import redtoreg
import numpy as np
import time

nc = Dataset('test.nc')
q = nc['q'][:].astype(np.float64)
print(q.shape)
lats = nc['latitude']
lats_distinct = np.unique(lats)
nlats = len(lats_distinct)
lonsperlat = np.empty(nlats, int)
for j in range(nlats):
    lonsperlat[j] = np.isclose(lats, lats_distinct[j]).sum()
print(lonsperlat)
for nlev in range(q.shape[0]):
    start = time.time()
    qq = redtoreg(q[nlev],lonsperlat)
    end = time.time()
    elapsed = end-start
    print('time in redtoreg for level %s = %s' % (nlev,elapsed))

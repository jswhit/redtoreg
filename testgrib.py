# validate with missing values, compare with pygrib implementation
# (requires pygrib)
import pygrib
import numpy as np
from redtoreg import redtoreg
grbs = pygrib.open('reduced_gg.grib')
grb = grbs.message(1)
print(grb)
grb.expand_grid(False)
data_reduced_gg = grb.values
grb.expand_grid(True)
data_full_gg_1 = grb.values
print(data_full_gg_1.min(), data_full_gg_1.max(), data_full_gg_1.shape)
data_full_gg_2 = redtoreg(data_reduced_gg,grb.pl,missval=grb.missingValue)
print(data_full_gg_2.min(), data_full_gg_2.max(), data_full_gg_2.shape)
assert np.allclose(data_full_gg_1, data_full_gg_2)

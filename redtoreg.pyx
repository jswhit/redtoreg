cimport numpy as npc
from numpy import ma
import numpy as np
npc.import_array()
cimport cython

ctypedef fused my_type:
    float
    double

def _redtoreg(cython.Py_ssize_t nlons, my_type[:] redgrid_data, long[:] lonsperlat, my_type missval):
    cdef cython.Py_ssize_t nlats = lonsperlat.shape[0]
    cdef cython.Py_ssize_t i,j,n,indx,ilons,im,ip
    cdef my_type zxi, zdx, flons
    if my_type is float:
        dtype = np.float32
    elif my_type is double:
        dtype = np.double
    reggrid_data = np.empty((nlats, nlons), dtype)
    cdef my_type[:, ::1] reggrid_data_view = reggrid_data
    indx = 0
    for j in range(nlats):
        ilons = lonsperlat[j]
        flons = ilons
        for i in range(nlons):
            # zxi is the grid index (relative to the reduced grid)
            # of the i'th point on the full grid.
            zxi = i * flons / nlons # goes from 0 to ilons
            im = <cython.Py_ssize_t>zxi; zdx = zxi - im
            im = (im + ilons)%ilons
            ip = (im + 1 + ilons)%ilons
            # if one of the nearest values is missing, use nearest
            # neighbor interpolation.
            if redgrid_data[indx+im] == missval or\
               redgrid_data[indx+ip] == missval: 
                if zdx < 0.5:
                    reggrid_data_view[j,i] = redgrid_data[indx+im]
                else:
                    reggrid_data_view[j,i] = redgrid_data[indx+ip]
            else: # linear interpolation.
                reggrid_data_view[j,i] = redgrid_data[indx+im]*(1.-zdx) +\
                                         redgrid_data[indx+ip]*zdx
        indx = indx + ilons
    return reggrid_data

def redtoreg(redgrid_data, lonsperlat, missval=None):
    """
    redtoreg(redgrid_data, lonsperlat, missval=None)

    Takes 1-d array on ECMWF reduced gaussian grid (``redgrid_data``), interpolates to corresponding
    regular gaussian grid (given by ``lonsperlat`` array, with max(lonsperlat) longitudes).
    Include handling of missing values (by using nearest neighbor interpolation)."""

    if missval is None:
        missval = np.nan
    datarr = _redtoreg(lonsperlat.max(),redgrid_data,lonsperlat,missval)
    return datarr

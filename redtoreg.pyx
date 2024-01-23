cimport numpy as npc
import numpy as np
npc.import_array()
cimport cython

ctypedef fused float_type:
    float
    double

def redtoreg(float_type[:] redgrid_data, long[:] lonsperlat, missval=None):
    """
    redtoreg(redgrid_data, lonsperlat, missval=None)

    Takes 1-d array on ECMWF reduced gaussian grid (redgrid_data), linearly interpolates
    to corresponding regular gaussian grid.

    Reduced gaussian grid defined by lonsperlat array, regular gaussian
    grid has the same number of latitudes and max(lonsperlat) longitudes.

    Includes handling of missing values using nearest neighbor interpolation.
    """

    cdef cython.Py_ssize_t nlats = lonsperlat.shape[0]
    cdef cython.Py_ssize_t nlons = np.max(lonsperlat)
    cdef cython.Py_ssize_t i,j,indx,ilons,im,ip,nlona
    cdef float_type zxi, zdx, flons, missvalc
    if float_type is float:
        float_dtype = np.float32
    elif float_type is double:
        float_dtype = np.double
    reggrid_data = np.empty((nlats, nlons), float_dtype)
    cdef float_type[:, ::1] reggrid_data_view = reggrid_data
    if missval is None:
        missvalc = np.nan
    else:
        missvalc = missval
    indx = 0
    for j in range(nlats):
        ilons = lonsperlat[j]; flons = ilons
        for i in range(nlons):
            # zxi is the grid index (relative to the reduced grid)
            # of the i'th point on the full grid.
            zxi = i * flons / nlons # goes from 0 to ilons
            im = <cython.Py_ssize_t>zxi; zdx = zxi - im
            im = (im + ilons)%ilons
            ip = (im + 1 + ilons)%ilons
            # if one of the nearest values is missing, use nearest
            # neighbor interpolation.
            if redgrid_data[indx+im] == missvalc or\
               redgrid_data[indx+ip] == missvalc: 
                if zdx < 0.5:
                    reggrid_data_view[j,i] = redgrid_data[indx+im]
                else:
                    reggrid_data_view[j,i] = redgrid_data[indx+ip]
            else: # linear interpolation.
                reggrid_data_view[j,i] = redgrid_data[indx+im]*(1.-zdx) +\
                                         redgrid_data[indx+ip]*zdx
        indx = indx + ilons
    return reggrid_data

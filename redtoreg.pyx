cimport numpy as npc
from numpy import ma
import numpy as np
npc.import_array()
cimport cython

cdef extern from "numpy/arrayobject.h":
    ctypedef int npy_intp
    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef npy_intp *dimensions
        cdef npy_intp *strides
        cdef object base
        cdef int flags
    npy_intp PyArray_ISCONTIGUOUS(ndarray arr)
    npy_intp PyArray_ISALIGNED(ndarray arr)

ctypedef fused my_type:
    float
    double

# new version
@cython.boundscheck(False)
@cython.wraparound(False)
def _redtoreg2(long nlons, my_type[:] redgrid_data, long[:] lonsperlat, double missval):
    cdef long npts = redgrid_data.shape[0]
    cdef long nlats = lonsperlat.shape[0]
    cdef long i,j,n,indx,ilons,im,ip
    cdef double zxi, zdx, flons, missvl
    if my_type is float:
        dtype = np.float32
    elif my_type is double:
        dtype = np.double
    reggrid_data = np.empty((nlats, nlons), dtype)
    indx = 0
    for j in range(nlats):
        ilons = lonsperlat[j]
        flons = <double>ilons
        for i in range(nlons):
            # zxi is the grid index (relative to the reduced grid)
            # of the i'th point on the full grid.
            zxi = i * flons / nlons # goes from 0 to ilons
            im = <long>zxi
            zdx = zxi - <double>im
            im = (im + ilons)%ilons
            ip = (im + 1 + ilons)%ilons
            # if one of the nearest values is missing, use nearest
            # neighbor interpolation.
            if redgrid_data[indx+im] == missval or\
               redgrid_data[indx+ip] == missval: 
                if zdx < 0.5:
                    reggrid_data[j,i] = redgrid_data[indx+im]
                else:
                    reggrid_data[j,i] = redgrid_data[indx+ip]
            else: # linear interpolation.
                reggrid_data[j,i] = redgrid_data[indx+im]*(1.-zdx) +\
                                    redgrid_data[indx+ip]*zdx
        indx = indx + ilons
    return reggrid_data

# for 3d arrays
@cython.boundscheck(False)
@cython.wraparound(False)
def _redtoreg3(long nlons, my_type[:,::1] redgrid_data, long[:] lonsperlat):
    cdef long nlevs = redgrid_data.shape[0]
    cdef long npts = redgrid_data.shape[1]
    cdef long nlats = lonsperlat.shape[0]
    cdef long i,j,n,indx,ilons,im,ip
    cdef double zxi, zdx, flons, missvl
    if my_type is float:
        dtype = np.float32
    elif my_type is double:
        dtype = np.double
    reggrid_data = np.empty((nlevs, nlats, nlons), dtype)
    for k in range(nlevs):
        indx = 0
        for j in range(nlats):
            ilons = lonsperlat[j]
            flons = <double>ilons
            for i in range(nlons):
                # zxi is the grid index (relative to the reduced grid)
                # of the i'th point on the full grid.
                zxi = i * flons / nlons # goes from 0 to ilons
                im = <long>zxi
                zdx = zxi - <double>im
                im = (im + ilons)%ilons
                ip = (im + 1 + ilons)%ilons
                reggrid_data[k,j,i] = redgrid_data[k,indx+im]*(1.-zdx) +\
                                      redgrid_data[k,indx+ip]*zdx
            indx = indx + ilons
    return reggrid_data

def redtoreg(redgrid_data, lonsperlat, missval=None):
    """
    redtoreg(redgrid_data, lonsperlat, missval=None)

    Takes 1-d array on ECMWF reduced gaussian grid (``redgrid_data``), linearly interpolates to corresponding
    regular gaussian grid (given by ``lonsperlat`` array, with max(lonsperlat) longitudes).
    If any values equal to specified missing value (``missval``, default NaN), a masked array is returned."""
    if missval is None:
        missval = np.nan
    datarr = _redtoreg2(lonsperlat.max(),redgrid_data,lonsperlat,missval)
    if np.count_nonzero(datarr==missval):
        datarr = ma.masked_values(datarr, missval)
    return datarr

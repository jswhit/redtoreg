cimport numpy as npc
from numpy import ma
import numpy as np
npc.import_array()

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

def _redtoreg(object nlonsin, npc.ndarray lonsperlat, npc.ndarray redgrid, \
              object missval):
    """
    convert data on global reduced gaussian to global
    full gaussian grid using linear interpolation.
    """
    cdef long i, j, n, im, ip, indx, ilons, nlats, npts
    cdef double zxi, zdx, flons, missvl
    cdef npc.ndarray reggrid
    cdef double *redgrdptr
    cdef double *reggrdptr
    cdef long *lonsptr
    nlons = nlonsin
    nlats = len(lonsperlat)
    npts = len(redgrid)
    if lonsperlat.sum() != npts:
        msg='size of reduced grid does not match number of data values'
        raise ValueError(msg)
    reggrid = missval*np.ones((nlats,nlons),np.double)
    # get data buffers and cast to desired type.
    lonsptr = <long *>lonsperlat.data
    redgrdptr = <double *>redgrid.data
    reggrdptr = <double *>reggrid.data
    missvl = <double>missval
    # iterate over full grid, do linear interpolation.
    n = 0
    indx = 0
    for j from 0 <= j < nlats:
        ilons = lonsptr[j]
        flons = <double>ilons
        for i from 0 <= i < nlons:
            # zxi is the grid index (relative to the reduced grid)
            # of the i'th point on the full grid.
            zxi = i * flons / nlons # goes from 0 to ilons
            im = <long>zxi
            zdx = zxi - <double>im
            if ilons != 0:
                im = (im + ilons)%ilons
                ip = (im + 1 + ilons)%ilons
                # if one of the nearest values is missing, use nearest
                # neighbor interpolation.
                if redgrdptr[indx+im] == missvl or\
                   redgrdptr[indx+ip] == missvl: 
                    if zdx < 0.5:
                        reggrdptr[n] = redgrdptr[indx+im]
                    else:
                        reggrdptr[n] = redgrdptr[indx+ip]
                else: # linear interpolation.
                    reggrdptr[n] = redgrdptr[indx+im]*(1.-zdx) +\
                                   redgrdptr[indx+ip]*zdx
            n = n + 1
        indx = indx + ilons
    return reggrid

def redtoreg(redgrid_data, lonsperlat, missval=None):
    """
    redtoreg(redgrid_data, lonsperlat, missval=None)

    Takes 1-d array on ECMWF reduced gaussian grid (``redgrid_data``), linearly interpolates to corresponding
    regular gaussian grid (given by ``lonsperlat`` array, with max(lonsperlat) longitudes).
    If any values equal to specified missing value (``missval``, default NaN), a masked array is returned."""
    if missval is None:
        missval = np.nan
    datarr = _redtoreg(lonsperlat.max(),lonsperlat,redgrid_data.astype(np.float64),missval)
    if np.count_nonzero(datarr==missval):
        datarr = ma.masked_values(datarr, missval)
    return datarr

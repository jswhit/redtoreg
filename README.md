Fast cython function to interpolate ECMWF reduced gaussian gridded data to full gaussian grid.

To test, run `python setup.py build_ext --inplace`, then `python test.py`
(first time will take a while, then cache will be cached data locally in `test.nc`).

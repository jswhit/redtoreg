Fast cython function to interpolate ECMWF reduced gaussian gridded data to full gaussian grid.

To test, run `python setup.py build_ext --inplace`, then `python test.py`
(first time will take a while to read data from google cloud, 
but after that data will be cached data locally in `test.nc`).

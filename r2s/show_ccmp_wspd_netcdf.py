# Data Set Title: RSS CCMP V2.0 ocean surface vector winds (Level 3.0) 
# Institution: Remote Sensing Systems
# Creator_Url: http://www.remss.com/measurements/ccmp
# Creator_Email: support@remss.com
# Summary = "CCMP V2.0 has been created using the same VAM as CCMP V1.1 only it is now running at Remote Sensing Systems. 
#   Input data have changed and now include all V7 radiometer data from RSS, V8.1 GMI data from RSS, V4 QuikSCAT and V1.2 ASCAT
#   data from RSS, quality checked moored buoy data from NDBC, PMEL, and ISDM, and ERA-Interim data from ECMWF. 
#   The entire CCMP data set has been reprocessed to create a consistent product suitable for long-term research.";

import numpy as np
from readNetCDF_CCMP import ReadNetcdf
# the readnetcdf module requires:
# numpy (http://www.scipy.org/Download)
# netcdf4-python (http://code.google.com/p/netcdf4-python/)
# matplotlib (http://matplotlib.org/)

file = 'CCMP_Wind_Analysis_19961006_V02.0_L3.0_RSS.nc'

# Read the data. Use optional 'summary' keyword argument if you want a full summary of the dataset.
#data = ReadNetcdf(file)
data = ReadNetcdf(file, summary=True) 


# Plot the data. hour can be 0=00z, 1=06z, 2=12z, 3=18z. 
# Year, month, and day should not need to be changed unless file naming structure changes
# Use optional 'minval' and 'maxval' keyword arguments to override 'valid_min' and 'valid_max' variables.
data.plot_map('uwnd', year=file[19:23], month=file[23:25], day=file[25:27], hour=1, minval=-15,maxval=15)

# Use optional 'colormap' keyword argument if you prefer a different colormap.
# Use optional 'filename' keyword argument if you want to save the image.
# data.plot_map('vwnd', year=file[19:23], month=file[23:25], day=file[25:27], hour=3, colormap='RdBu_r', filename='imageoutput.png', minval=-15, maxval=15)
data.plot_map('vwnd', year=file[19:23], month=file[23:25], day=file[25:27], hour=3, colormap='RdBu_r', minval=-15, maxval=15)

# Get the data.
one_map, map_date = data.get_map('uwnd', year=file[19:23], month=file[23:25], day=file[25:27], hour=3)
print(map_date)

# Write data to a text file.
# np.savetxt('textoutput.txt',one_map)

# Read it in Fortran using:
# real(4) one_map(1440,628)
# open(unit=2,file='textoutput.txt',status='old')
# read(2,*) one_map
# close(2)

# Print short variable list.
print('Variable list:')
for var in data.dataset.variables: print(var)

# Print first and last latitude and longitude from their lists.
print('Latitude goes from ', data.dataset.variables['latitude'][0], 'to',
      data.dataset.variables['latitude'][-1])
print('Longitude goes from ', data.dataset.variables['longitude'][0], 'to', data.dataset.variables['longitude'][-1])

# Print information about ReadNetcdf.
# help(ReadNetcdf)

# close the dataset
data.close()

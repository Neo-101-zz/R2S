# Set the URL string to point to a specific data URL. Some generic examples are:
#   https://servername/data/path/file
#   https://servername/opendap/path/file[.format[?subset]]
#   https://servername/daac-bin/OTF/HTTP_services.cgi?KEYWORD=value[&KEYWORD=value]
URL = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXFLX.5.12.4%2F2017%2F09%2FMERRA2_400.tavg1_2d_flx_Nx.20170925.nc4&FORMAT=bmM0Lw&BBOX=27.4%2C-77.1%2C35.4%2C-69.1&TIME=2017-09-25T16%3A30%3A00%2F2017-09-25T17%3A30%3A59&LABEL=MERRA2_400.tavg1_2d_flx_Nx.20170925.SUB.nc&SHORTNAME=M2T1NXFLX&SERVICE=SUBSET_MERRA2&VERSION=1.02&DATASET_VERSION=5.12.4&VARIABLES=SPEED%2CSPEEDMAX%2CULML%2CVLML'

# Set the FILENAME string to the data file name, the LABEL keyword value, or any customized name. 
FILENAME = 'merra2_test.nc'

import requests
result = requests.get(URL)
try:
    result.raise_for_status()
    f = open(FILENAME,'wb')
    f.write(result.content)
    f.close()
    print('contents of URL written to '+FILENAME)
except:
    print('requests.get() returned an error code '+str(result.status_code))

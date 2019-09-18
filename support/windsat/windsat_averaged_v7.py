#	This routine reads version-7 RSS WindSat time-averaged files 
#	The time-averaged files include:
#   3-day	(average of 3 days ending on file date),  satname_yyyymmddv7_d3d.gz
#   weekly	(average of 7 days ending on Saturday of file date),  satname_yyyymmddv7.gz
#   monthly	(average of all days in month),  satname_yyyymmv7.gz
#	 	where satname  = name of satellite (wsat)
#	       yyyy		= year
#		   mm		= month
#		   dd		= day of month
#
#   missing = fill value used for missing data;
#                  if None, then fill with byte codes (251-255)

#	the output values correspond to:
#	sst			sea surface temperature in deg Celsius
#	windLF		10m surface wind, low frequency, in meters/second
#	windMF		10m surface wind, medium frequency, in meters/second
#	vapor		columnar or integrated water vapor in millimeters
#	cloud		cloud liquid water in millimeters
#	rain		rain rate in millimeters/hour
#   longitude	Grid Cell Center Longitude', LON = 0.25*x_grid_location - 0.125 degrees east
#   latitude	Grid Cell Center Latitude',  LAT = 0.25*y_grid_location - 90.125
#   land		Is this land?
#   ice			Is this ice?
#   nodata		Is there no data

from bytemaps import sys
from bytemaps import Dataset
from bytemaps import Verify


class WindSatAveraged(Dataset):
    """ Read averaged WindSat bytemaps. """
    """
    Public data:
        filename = name of data file
        missing = fill value used for missing data;
                  if None, then fill with byte codes (251-255)
        dimensions = dictionary of dimensions for each coordinate
        variables = dictionary of data for each variable
    """

    def __init__(self, filename, missing=-999.):
        """
        Required arguments:
            filename = name of data file to be read (string)
                
        Optional arguments:
            missing = fill value for missing data,
                      default is the value used in verify file
        """       
        self.filename = filename
        self.missing = missing
        Dataset.__init__(self)

    # Dataset:

    def _attributes(self):
        return ['coordinates','long_name','units','valid_min','valid_max']

    def _coordinates(self):
        return ('variable','latitude','longitude')

    def _shape(self):
        return (8,720,1440)

    def _variables(self):
        return ['sst','w-lf','w-mf','vapor','cloud','rain',
                'w-aw','wdir','longitude','latitude','land','ice','nodata']                

    # _default_get():
    
    def _get_index(self,var):
        return {'sst' : 0,
                'w-lf' : 1,
                'w-mf' : 2,
                'vapor' : 3,
                'cloud' : 4,
                'rain' : 5,
                'w-aw' : 6,
                'wdir' : 7,
                }[var]

    def _get_scale(self,var):
        return {'sst' : 0.15,
                'w-lf' : 0.2,
                'w-mf' : 0.2,
                'vapor' : 0.3,
                'cloud' : 0.01,
                'rain' : 0.1,
                'w-aw' : 0.2,
                'wdir' : 1.5,
                }[var]

    def _get_offset(self,var):
        return {'sst' : -3.0,
                'cloud' : -0.05,
                }[var]

    # _get_ attributes:
    
    def _get_long_name(self,var):
        return {'sst' : 'Sea Surface Temperature',
                'w-lf' : '10-m Surface Wind Speed (low frequency)',
                'w-mf' : '10-m Surface Wind Speed (medium frequency)',
                'vapor' : 'Columnar Water Vapor',
                'cloud' : 'Cloud Liquid Water',
                'rain' : 'Surface Rain Rate',
                'w-aw' : 'All-Weather 10-m Surface Wind Speed',
                'wdir' : 'Surface Wind Direction',
                'longitude' : 'Grid Cell Center Longitude',
                'latitude' : 'Grid Cell Center Latitude',
                'land' : 'Is this land?',
                'ice' : 'Is this ice?',
                'nodata' : 'Is there no data?',
                }[var]

    def _get_units(self,var):
        return {'sst' : 'deg Celsius',
                'w-lf' : 'm/s',
                'w-mf' : 'm/s',
                'vapor' : 'mm',
                'cloud' : 'mm',
                'rain' : 'mm/hr',
                'w-aw' : 'm/s',
                'wdir' : 'deg oceanographic',
                'longitude' : 'degrees east',
                'latitude' : 'degrees north',
                'land' : 'True or False',
                'ice' : 'True or False',
                'nodata' : 'True or False',
                }[var]

    def _get_valid_min(self,var):
        return {'sst' : -3.0,
                'w-lf' : 0.0,
                'w-mf' : 0.0,
                'vapor' : 0.0,
                'cloud' : -0.05,
                'rain' : 0.0,
                'w-aw' : 0.2,
                'wdir' : 0.0,
                'longitude' : 0.0,
                'latitude' : -90.0,
                'land' : False,
                'ice' : False,
                'nodata' : False,
                }[var]

    def _get_valid_max(self,var):
        return {'sst' : 34.5,
                'w-lf' : 50.0,
                'w-mf' : 50.0,
                'vapor' : 75.0,
                'cloud' : 2.45,
                'rain' : 25.0,
                'w-aw' : 50.0,
                'wdir' : 360.0,
                'longitude' : 360.0,
                'latitude' : 90.0,
                'land' : True,
                'ice' : True,
                'nodata' : True,
                }[var]


class ThreedayVerify(Verify):
    """ Contains info for verification. """
    
    def __init__(self,dataset):
        self.filename = 'windsat_v7.0.1_verify.txt'
        self.ilon1 = 170
        self.ilon2 = 175
        self.ilat1 = 274
        self.ilat2 = 278
        self.variables = ['sst','w-lf','w-mf','vapor',
                          'cloud','rain','w-aw','wdir']        
        self.startline = 133
        self.columns = {'sst' : 3,
                        'w-lf' : 4,
                        'w-mf' : 5,
                        'vapor' : 6,
                        'cloud' : 7,
                        'rain' : 8,
                        'w-aw' : 9,
                        'wdir' : 10 }
        Verify.__init__(self,dataset)


class WeeklyVerify(Verify):
    """ Contains info for verification. """
    
    def __init__(self,dataset):
        self.filename = 'windsat_v7.0.1_verify.txt'
        self.ilon1 = 170
        self.ilon2 = 175
        self.ilat1 = 274
        self.ilat2 = 278        
        self.variables = ['sst','w-lf','w-mf','vapor',
                          'cloud','rain','w-aw','wdir']        
        self.startline = 168
        self.columns = {'sst' : 3,
                        'w-lf' : 4,
                        'w-mf' : 5,
                        'vapor' : 6,
                        'cloud' : 7,
                        'rain' : 8,
                        'w-aw' : 9,
                        'wdir' : 10 }
        Verify.__init__(self,dataset)


class MonthlyVerify(Verify):
    """ Contains info for verification. """
    
    def __init__(self,dataset):
        self.filename = 'windsat_v7.0.1_verify.txt'
        self.ilon1 = 170
        self.ilon2 = 175
        self.ilat1 = 274
        self.ilat2 = 278        
        self.variables = ['sst','w-lf','w-mf','vapor',
                          'cloud','rain','w-aw','wdir']        
        self.startline = 204
        self.columns = {'sst' : 3,
                        'w-lf' : 4,
                        'w-mf' : 5,
                        'vapor' : 6,
                        'cloud' : 7,
                        'rain' : 8,
                        'w-aw' : 9,
                        'wdir' : 10 }
        Verify.__init__(self,dataset)


if __name__ == '__main__':
    """ Automated testing. """    

    # read 3-day averaged:
    wsat = WindSatAveraged('wsat_20060409v7.0.1_d3d.gz')
    if not wsat.variables: sys.exit('file not found')

    # verify 3-day:
    verify = ThreedayVerify(wsat)
    if verify.success: print('successful verification for 3-day')
    else: sys.exit('verification failed for 3-day')
    print('')

    # read weekly averaged:
    wsat = WindSatAveraged('wsat_20060415v7.0.1.gz')
    if not wsat.variables: sys.exit('file not found')

    # verify weekly:
    verify = WeeklyVerify(wsat)
    if verify.success: print('successful verification for weekly')
    else: sys.exit('verification failed for weekly')     
    print('')
    
    # read monthly averaged:
    wsat = WindSatAveraged('wsat_200604v7.0.1.gz')
    if not wsat.variables: sys.exit('file not found')
    
    # verify:
    verify = MonthlyVerify(wsat)
    if verify.success: print('successful verification for monthly')
    else: sys.exit('verification failed for monthly')      
    print('')
    
    print('all tests completed successfully')
    print('')

from bytemaps import sys
from bytemaps import Dataset
from bytemaps import Verify

from bytemaps import get_data
from bytemaps import ibits
from bytemaps import is_bad
from bytemaps import where


class QuikScatAveraged(Dataset):
    """ Read averaged QSCAT bytemaps. """
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
        return (3,720,1440)

    def _variables(self):
        return ['windspd','winddir','scatflag','radrain',
                'longitude','latitude','land','ice','nodata']                

    # _default_get():

    def _get_index(self,var):
        return {'windspd' : 0,
                'winddir' : 1,
                'rain' : 2 }[var]

    def _get_scale(self,var):
        return {'windspd' : 0.2,
                'winddir' : 1.5 }[var]

    # _get_ attributes:
    
    def _get_long_name(self,var):
        return {'windspd' : '10-m Surface Wind Speed',
                'winddir' : '10-m Surface Wind Direction',
                'scatflag' : 'Scatterometer Rain Flag',
                'radrain' : 'Radiometer Rain Flag',
                'longitude' : 'Grid Cell Center Longitude',
                'latitude' : 'Grid Cell Center Latitude',
                'land' : 'Is this land?',
                'ice' : 'Is this ice?',
                'nodata' : 'Is there no data?',
                }[var]

    def _get_units(self,var):
        return {'windspd' : 'm/s',
                'winddir' : 'deg oceanographic',
                'scatflag' : '0=no-rain, 1=rain',
                'radrain' : '0=no-rain, -1=adjacent rain, >0=rain(km*mm/hr)',
                'longitude' : 'degrees east',
                'latitude' : 'degrees north',
                'land' : 'True or False',
                'ice' : 'True or False',
                'nodata' : 'True or False',
                }[var]

    def _get_valid_min(self,var):
        return {'windspd' : 0.0,
                'winddir' : 0.0,
                'scatflag' : 0,
                'radrain' : -1,
                'longitude' : 0.0,
                'latitude' : -90.0,
                'land' : False,
                'ice' : False,
                'nodata' : False,
                }[var]

    def _get_valid_max(self,var):
        return {'windspd' : 50.0,
                'winddir' : 360.0,
                'scatflag' : 1,
                'radrain' : 31,
                'longitude' : 360.0,
                'latitude' : 90.0,
                'land' : True,
                'ice' : True,
                'nodata' : True,
                }[var]

    # _get_ variables:
    
    def _get_scatflag(self,var,bmap):
        indx = self._get_index('rain')
        scatflag = get_data(ibits(bmap,ipos=0,ilen=1),indx=indx)
        bad = is_bad(get_data(bmap,indx=0))
        scatflag[bad] = self.missing
        return scatflag

    def _get_radrain(self,var,bmap):
        indx = self._get_index('rain')
        radrain = get_data(ibits(bmap,ipos=1,ilen=1),indx=indx)
        good = (radrain == 1)        
        radrain[~good] = self.missing
        intrain = get_data(ibits(bmap,ipos=2,ilen=6),indx=indx)
        nonrain = where(intrain == 0)
        adjrain = where(intrain == 1)
        hasrain = where(intrain > 1)
        intrain[nonrain] = 0.0
        intrain[adjrain] = -1.0
        intrain[hasrain] = 0.5*(intrain[hasrain]-1)
        radrain[good] = intrain[good]
        bad = is_bad(get_data(bmap,indx=0))
        radrain[bad] = self.missing
        return radrain


class ThreedayVerify(Verify):
    """ Contains info for verification. """
    
    def __init__(self,dataset):
        self.filename = 'qscat_v4_averaged_verify.txt'
        self.ilon1 = 170
        self.ilon2 = 175
        self.ilat1 = 274
        self.ilat2 = 278        
        self.variables = ['windspd','winddir','scatflag','radrain']
        self.startline = 16
        self.columns = {'windspd' : 3,
                        'winddir' : 4,
                        'scatflag' : 5,
                        'radrain' : 6 }        
        Verify.__init__(self,dataset)       


class WeeklyVerify(Verify):
    """ Contains info for verification. """
    
    def __init__(self,dataset):
        self.filename = 'qscat_v4_averaged_verify.txt'
        self.ilon1 = 170
        self.ilon2 = 175
        self.ilat1 = 274
        self.ilat2 = 278        
        self.variables = ['windspd','winddir','scatflag','radrain']
        self.startline = 51
        self.columns = {'windspd' : 3,
                        'winddir' : 4,
                        'scatflag' : 5,
                        'radrain' : 6 }        
        Verify.__init__(self,dataset)        


class MonthlyVerify(Verify):
    """ Contains info for verification. """
    
    def __init__(self,dataset):
        self.filename = 'qscat_v4_averaged_verify.txt'
        self.ilon1 = 170
        self.ilon2 = 175
        self.ilat1 = 274
        self.ilat2 = 278        
        self.variables = ['windspd','winddir','scatflag','radrain']
        self.startline = 85
        self.columns = {'windspd' : 3,
                        'winddir' : 4,
                        'scatflag' : 5,
                        'radrain' : 6 }        
        Verify.__init__(self,dataset)


if __name__ == '__main__':
    """ Automated testing. """    

    # read 3-day averaged:
    qscat = QuikScatAveraged('qscat_20000111v4_3day.gz')
    if not qscat.variables: sys.exit('file not found')

    # verify 3-day:
    verify = ThreedayVerify(qscat)
    if verify.success: print('successful verification for 3-day')
    else: sys.exit('verification failed for 3-day')
    print('')

    # read weekly averaged:
    qscat = QuikScatAveraged('qscat_20000115v4.gz')
    if not qscat.variables: sys.exit('file not found')

    # verify weekly:
    verify = WeeklyVerify(qscat)
    if verify.success: print('successful verification for weekly')
    else: sys.exit('verification failed for weekly')     
    print('')
    
    # read monthly averaged:
    qscat = QuikScatAveraged('qscat_200001v4.gz')
    if not qscat.variables: sys.exit('file not found')
    
    # verify:
    verify = MonthlyVerify(qscat)
    if verify.success: print('successful verification for monthly')
    else: sys.exit('verification failed for monthly')      
    print('')
    
    print('all tests completed successfully')
    print ('')

import re
from datetime import date
from datetime import timedelta
from calendar import monthrange
from calendar import month_name
from mpl_toolkits.basemap import Basemap

import numpy as np
# requires numpy (http://www.scipy.org/Download)

from netCDF4 import Dataset
# requires netcdf4-python (http://code.google.com/p/netcdf4-python/)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as color
# plot_map requires matplotlib (http://matplotlib.org/)

class ReadNetcdf:
    """ReadNetcdf provides methods to get and plot a map from the CCMP data set in netCDF."""
    
    def __init__(self, *args, **kwargs):
        """Read the dataset."""

        """
            Arguments:
            filename (required)

            Keyword arguments (optional):
            summary = True, if you want a full report of the file contents

            Returns:
            a read netcdf dataset instance
        """
        
        self.filename = args[0]
        self.dataset = Dataset(self.filename)
        print 'Dataset read successfully'
        print 'Dataset format: ',self.dataset.file_format
        print
        
        self._print_dataset_info(**kwargs)
        self._create_date_objects()


    def close(self):
        """ Close the dataset. """

        self.dataset.close()


    def get_map(self, var, **kwargs):
        """Get a map for a particular variable and time."""

        """
            Arguments:
            var = variable name (string)
           
            Keyword arguments:
            month = month (string)            
            year = year (string)
            day = day (string)
            hour= hour(integer)            

            Returns:
            map of variable (a numpy array)
            date of map (a datetime.date object)

            Behavior:
            If no valid data are available, returns None.
        """

        year = kwargs.get('year')
        month = kwargs.get('month')
        day = kwargs.get('day')
        hour = kwargs.get('hour')
        hours=['00Z', '06Z', '12Z', '18Z']
        
        print 'Get map for year = ' + year + ', month = ' + month + ', day = ' + day + 'hour = ' + hours[hour]

        self.long_name  = getattr(self.dataset.variables[var], 'long_name')
        print 'Get map of ' + self.long_name
            
        self._extract_one_map(var, year=year, month=month, day=day, hour=hour)        
    
        print
        
        return self.one_map, self.map_date

    def plot_map(self, var, **kwargs):
        """Plot a map for a particular variable and time."""

        """
            Arguments:
            var = variable name (string)
           
            Keyword arguments:
            month = month (string)            
            year = year (string)
            day = day (string)
            hour= hour(integer)
            Notes:
            defaults for optional keyword arguments are set in _set_plot()

            Optional keyword arguments:
            baddatacolor = color used for no/bad data (default = 'black')
            colorbar_orientation = colorbar orientation (default = 'horizontal')
            colormap = matplotlib colormap name ('coolwarm' unless otherwise specified in call to plot_map)
            facecolor = background color for plot (default = 'white')                        
            lattickspacing = latitude tick-spacing (default = 30 deg)
            lonshift = shift map by this amount in longitude in degrees (default = 30 deg)                        
            lontickspacing = longitude tick-spacing (default = 30 deg)           
            maxval = maximum data value for colorbar (default = valid_max)            
            minval = minimum data value for colorbar (default = valid_min)
            titlefontsize = font size for plot title (default = medium)
            yeartickspacing = year tick-spacing (default = 5 years)            
            
            Returns:
            an image to the screen
        """

        one_map, map_date = self.get_map(var, **kwargs)
        if one_map == None: return
        
        self._set_plot(var, **kwargs)
        fig = plt.figure(facecolor=self.facecolor)
        
        hours=['00Z', '06Z', '12Z', '18Z']
        
        year = kwargs.get('year')
        month = kwargs.get('month')
        day = kwargs.get('day')
        hour =kwargs.get('hour')
        title = self.long_name + ' for ' + month_name[int(month)] + ' ' + day + ', ' + year + ' at ' + hours[hour]
        plt.title(title, fontsize=self.titlefontsize)

        palette = cm.get_cmap(self.cmap)

        palette.set_bad(self.baddatacolor)        
        no_data = np.where(one_map == self.fillvalue)
        one_map[no_data] = np.nan
        
        coords = getattr(self.dataset.variables[var],'coordinates')
        if 'longitude' in coords:
            one_map = np.roll(one_map, -1*self.lonshift*self.lonperdeg, axis=1)
        else:
            one_map = np.transpose(one_map)
        
        m = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90, llcrnrlon=0,urcrnrlon=360)
        
        #one_map = np.flipud(one_map)            
        aximg = m.imshow(one_map, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        
        m.drawcoastlines()
        
        cbar = fig.colorbar(aximg, orientation=self.cbar_orientation)           
        cbar.set_label(self.units)
        
        plt.yticks(np.arange(-90, 91, 30))
        plt.xticks(np.arange(0, 361, 30))                          
        
        filename = kwargs.get('filename')
        if filename: fig.savefig(filename)

        plt.show()
        plt.close()

    
    # The functions that follow support the get_map() and plot_map() methods.

    # Functions for setting the plot appearance:

    def _set_plot(self, var, **kwargs):
        """Set the look of the plot."""

        self.lonshift = kwargs.get('lonshift')
        if not self.lonshift: self.lonshift = 0

        self.lattickspacing = kwargs.get('lattickspacing')
        if not self.lattickspacing: self.lattickspacing = 30

        self.lontickspacing = kwargs.get('lontickspacing')
        if not self.lontickspacing: self.lontickspacing = 30

        self.yeartickspacing = kwargs.get('yeartickspacing')
        if not self.yeartickspacing: self.yeartickspacing = 5

        self.dlat = float(getattr(self.dataset,'geospatial_lat_resolution').split()[0])           
        self.dlon = float(getattr(self.dataset,'geospatial_lon_resolution').split()[0])
        self.latperdeg = int(1.0/self.dlat)            
        self.lonperdeg = int(1.0/self.dlon)

        self.fillvalue = getattr(self.dataset.variables[var],'_Fillvalue')
        self.units = getattr(self.dataset.variables[var],'units')

        self.vmin = kwargs.get('minval')
        if not self.vmin: self.vmin = getattr(self.dataset.variables[var], 'valid_min')

        self.vmax = kwargs.get('maxval')
        if not self.vmax: self.vmax = getattr(self.dataset.variables[var], 'valid_max')
      
        self.cmap = kwargs.get('colormap')
        if not self.cmap:
            self.cmap = 'coolwarm'

        self.cbar_orientation = kwargs.get('colorbar_orientation')
        if not self.cbar_orientation: self.cbar_orientation = 'horizontal'

        self.facecolor = kwargs.get('facecolor')
        if not self.facecolor: self.facecolor = 'white'       

        self.baddatacolor = kwargs.get('baddatacolor')
        if not self.baddatacolor: self.baddatacolor = 'black'

        self.titlefontsize = kwargs.get('titlefontsize')
        if not self.titlefontsize: self.titlefontsize = 'medium'

    def _extract_one_map(self, var, **kwargs):
        """Extract one map for a particular variable and time."""

        year = kwargs.get('year')        
        month = kwargs.get('month')
        day = kwargs.get('day')
        hour = kwargs.get('hour')
        
        self.one_map = None
        self.map_date = None

        if np.ndim(self.dataset.variables[var]) == 2:
           self.one_map = np.array(self.dataset.variables[var][:,:])
           return
        
        self.timevar = getattr(self.dataset.variables[var],'coordinates').split()[0]
        if 'time' not in self.timevar: print 'ReadNetcdf could not find time coordinate'

        yearin = year
        if not year: year = self.basedate[self.timevar].year        
        
        hours=['00Z', '06Z', '12Z', '18Z']
        
        self.one_map = np.array(self.dataset.variables[var][hour,:,:])
        one_map_plus = np.zeros((92, 1440))
        self.one_map = np.append(self.one_map, one_map_plus, axis=0)
        self.one_map = np.roll(self.one_map, 46, axis=0)
        
        self.map_date = self.dates[self.timevar][hour]
        self.map_date = str(self.map_date)
        self.map_date += ' ' + hours[hour]
        if not yearin: self.map_date = None

    # Functions for date handling:

    def _create_date_objects(self, *args, **kwargs):
        """Create datetime date objects."""
        
        self.dates = {}

        try: varlist = args[0]
        except IndexError: varlist = ['time']

        for var in varlist:
            self._get_base_date(var, **kwargs)
            self._convert_time_to_dates(var)            

    def _get_base_date(self, var, **kwargs):
        """Get the base date (the date in 'hours since ...')."""

        self.basedate = {}
        
        datestring = self.dataset.variables[var].units
        yy, mm, dd = self._parse_date(datestring, **kwargs)        
        
        self.basedate[var] = date(yy,mm,dd)

    def _parse_date(self, datestring, **kwargs):
        """Parse a date string."""

        prefix = kwargs.get('prefix')
        if prefix == None: prefix = 'hours since '

        m = re.search(prefix+'(\d+)-(\d+)-(\d+).*', datestring)
        if m == None: print 'ReadNetcdf expects time units of the form: hours since y-m-d ...'

        return int(m.group(1)),int(m.group(2)),int(m.group(3))
        
    def _convert_time_to_dates(self, var):
        """Convert days since base date to datetime.date."""

        self.dates[var] = []
        
        for hours in self.dataset.variables[var]:
            self.dates[var].append(self.basedate[var] + timedelta(hours=float(hours)))
    # Function for printing information:

    def _print_dataset_info(self, **kwargs):
        """ Print a list of important dataset and provenance information."""
        
        if kwargs.get('summary'):
            print 'Dataset summary:'
            print self.dataset
            print 'Variables in dataset:'
            for var in self.dataset.variables:
                print var, self.dataset.variables[var]

        attrib_list = ['title', 'institution', 'project', 'creator_url', 'creator_email', 'summary']
        for attrib in attrib_list:
            if hasattr(self.dataset,attrib):
                print attrib.title() + ': ' + getattr(self.dataset,attrib)
            else:
                print 'Cannot find: ' + attrib.title()
        print

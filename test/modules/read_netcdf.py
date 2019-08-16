import re
from datetime import date
from datetime import timedelta
from calendar import monthrange
from calendar import month_name
from mpl_toolkits.basemap import Basemap

import numpy as np
import math

from netCDF4 import Dataset

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as color

class ReadNetcdf:

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
        print ('Dataset read successfully.')
        print ('Dataset format: ', self.dataset.file_format)
        print ()

        self.type = kwargs.get('type')
        self.time = kwargs.get('time')

        if re.search('icoads', self.type):
            self.missing = 32766
        self._print_dataset_info(**kwargs)
        # self._create_date_objects()
        # self._create_product_objects(**kwargs)

    def close(self):
        """ Close the dataset. """

        self.dataset.close()

    # def get_map(self, var, **kwargs):
    #     """Get a map for a particular variable and time."""
    #
    #     """
    #         Arguments:
    #         var = variable name (string)
    #
    #         Keyword arguments:
    #         month = month (string)
    #         year = year (string)
    #         day = day (string)
    #         hour= hour(integer)
    #
    #         Returns:
    #         map of variable (a numpy array)
    #         date of map (a datetime.date object)
    #
    #         Behavior:
    #         If no valid data are available, returns None.
    #     """
    #
    #     year = kwargs.get('year')
    #     month = kwargs.get('month')
    #     day = kwargs.get('day')
    #     hour = kwargs.get('hour')
    #     hours = ['00Z', '06Z', '12Z', '18Z']
    #
    #     print 'Get map for year = ' + year + ', month = ' + month + ', day = ' + day + ', hour = ' + hours[hour]
    #
    #     self.long_name = getattr(self.dataset.variables[var], 'long_name')
    #     print 'Get map of ' + self.long_name
    #
    #     self._extract_one_map(var, year=year, month=month, day=day, hour=hour)
    #
    #     print
    #
    #     return self.one_map, self.map_date
    #
    # def plot_map(self, var, **kwargs):
    #     """Plot a map for a particular variable and time."""
    #
    #     """
    #         Arguments:
    #         var = variable name (string)
    #
    #         Keyword arguments:
    #         month = month (string)
    #         year = year (string)
    #         day = day (string)
    #         hour= hour(integer)
    #         Notes:
    #         defaults for optional keyword arguments are set in _set_plot()
    #
    #         Optional keyword arguments:
    #         baddatacolor = color used for no/bad data (default = 'black')
    #         colorbar_orientation = colorbar orientation (default = 'horizontal')
    #         colormap = matplotlib colormap name ('coolwarm' unless otherwise specified in call to plot_map)
    #         facecolor = background color for plot (default = 'white')
    #         lattickspacing = latitude tick-spacing (default = 30 deg)
    #         lonshift = shift map by this amount in longitude in degrees (default = 30 deg)
    #         lontickspacing = longitude tick-spacing (default = 30 deg)
    #         maxval = maximum data value for colorbar (default = valid_max)
    #         minval = minimum data value for colorbar (default = valid_min)
    #         titlefontsize = font size for plot title (default = medium)
    #         yeartickspacing = year tick-spacing (default = 5 years)
    #
    #         Returns:
    #         an image to the screen
    #     """
    #
    #     one_map, map_date = self.get_map(var, **kwargs)
    #     if one_map.all() == None: return
    #
    #     self._set_plot(var, **kwargs)
    #     fig = plt.figure(facecolor=self.facecolor)
    #     hours = ['00Z', '06Z', '12Z', '18Z']
    #
    #     year = kwargs.get('year')
    #     month = kwargs.get('month')
    #     day = kwargs.get('day')
    #     hour = kwargs.get('hour')
    #     title = self.long_name + ' for ' + month_name[int(month)] + ' ' + day + ', ' + year + ' at ' + hours[hour]
    #     plt.title(title, fontsize=self.titlefontsize)
    #
    #     palette = cm.get_cmap(self.cmap)
    #
    #     palette.set_bad(self.baddatacolor)
    #     no_data = np.where(one_map == self.fillvalue)
    #     one_map[no_data] = np.nan
    #
    #     coords = getattr(self.dataset.variables[var], 'coordinates')
    #     if 'longitude' in coords:
    #         one_map = np.roll(one_map, -1 * self.lonshift * self.lonperdeg, axis=1)
    #     else:
    #         one_map = np.transpose(one_map)
    #
    #     m = Basemap(projection='cyl', llcrnrlat=10, urcrnrlat=45, llcrnrlon=100, urcrnrlon=131)
    #
    #     # one_map = np.flipud(one_map)
    #     aximg = m.imshow(one_map, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
    #
    #     m.drawcoastlines()
    #     m.fillcontinents('black')
    #     cbar = fig.colorbar(aximg, orientation=self.cbar_orientation)
    #     cbar.set_label(self.units)
    #
    #     # plt.yticks(np.arange(-90, 91, 30))
    #     # plt.xticks(np.arange(0, 361, 30))
    #
    #     plt.xlim(self.lons)
    #     plt.ylim(self.lats)
    #
    #     filename = kwargs.get('filename')
    #     if filename: fig.savefig(filename)
    #
    #     plt.show()
    #     plt.close()
    #
    # # The functions that follow support the get_map() and plot_map() methods.
    #
    # # Functions for setting the plot appearance:
    #
    # def _set_plot(self, var, **kwargs):
    #     """Set the look of the plot."""
    #
    #     self.lonshift = kwargs.get('lonshift')
    #     if not self.lonshift: self.lonshift = 0
    #
    #     self.lattickspacing = kwargs.get('lattickspacing')
    #     if not self.lattickspacing: self.lattickspacing = 30
    #
    #     self.lontickspacing = kwargs.get('lontickspacing')
    #     if not self.lontickspacing: self.lontickspacing = 30
    #
    #     self.yeartickspacing = kwargs.get('yeartickspacing')
    #     if not self.yeartickspacing: self.yeartickspacing = 5
    #
    #     self.dlat = float(getattr(self.dataset, 'geospatial_lat_resolution').split()[0])
    #     self.dlon = float(getattr(self.dataset, 'geospatial_lon_resolution').split()[0])
    #     self.latperdeg = int(1.0 / self.dlat)
    #     self.lonperdeg = int(1.0 / self.dlon)
    #
    #     self.fillvalue = getattr(self.dataset.variables[var], '_Fillvalue')
    #     self.units = getattr(self.dataset.variables[var], 'units')
    #
    #     self.vmin = kwargs.get('minval')
    #     if not self.vmin: self.vmin = getattr(self.dataset.variables[var], 'valid_min')
    #
    #     self.vmax = kwargs.get('maxval')
    #     if not self.vmax: self.vmax = getattr(self.dataset.variables[var], 'valid_max')
    #
    #     self.cmap = kwargs.get('colormap')
    #     if not self.cmap:
    #         # self.cmap = 'coolwarm'
    #         self.cmap = plt.cm.jet
    #
    #     self.cbar_orientation = kwargs.get('colorbar_orientation')
    #     if not self.cbar_orientation: self.cbar_orientation = 'horizontal'
    #
    #     self.facecolor = kwargs.get('facecolor')
    #     if not self.facecolor: self.facecolor = 'white'
    #
    #     self.baddatacolor = kwargs.get('baddatacolor')
    #     if not self.baddatacolor: self.baddatacolor = 'black'
    #
    #     self.titlefontsize = kwargs.get('titlefontsize')
    #     if not self.titlefontsize: self.titlefontsize = 'medium'
    #     ##############################################################
    #     self.lons = kwargs.get('lons')
    #     if not self.lons: self.lons = (0, 360)
    #
    #     self.lats = kwargs.get('lats')
    #     if not self.lats: self.lats(-90, 90)
    #
    # ##############################################################
    #
    # def _extract_one_map(self, var, **kwargs):
    #     """Extract one map for a particular variable and time."""
    #
    #     year = kwargs.get('year')
    #     month = kwargs.get('month')
    #     day = kwargs.get('day')
    #     hour = kwargs.get('hour')
    #
    #     self.one_map = None
    #     self.map_date = None
    #     # print
    #     # print 'Variables:'
    #     # for var in self.dataset.variables:
    #     #     aline = ' '.join([' ' * 3, var, ':', self.dataset.variables[var].long_name])
    #     #     print aline
    #     #
    #     # print
    #     # print 'Dimensions:'
    #     # for dim in self.dataset.dimensions:
    #     #     aline = ' '.join([' ' * 3, dim, ':', str(self.dataset.dimensions[dim])])
    #     #     print aline
    #     #
    #     # print np.array(self.dataset.variables[var][0,0,1])
    #
    #     if np.ndim(self.dataset.variables[var]) == 2:
    #         self.one_map = np.array(self.dataset.variables[var][:, :])
    #         return
    #
    #     self.timevar = getattr(self.dataset.variables[var], 'coordinates').split()[0]
    #     if 'time' not in self.timevar: print ('ReadNetcdf could not find time coordinate')
    #
    #     year = self.dates[self.timevar][0].year
    #     month = self.dates[self.timevar][0].month
    #     day = self.dates[self.timevar][0].day
    #     yearin = year
    #     if not yearin: year = self.basedate[self.timevar].year
    #
    #     hours = ['00Z', '06Z', '12Z', '18Z']
    #
    #     self.one_map = np.array(np.squeeze(self.dataset.variables[var], axis=0))
    #     one_map_plus = np.zeros((92, 1440))
    #     self.one_map = np.append(self.one_map, one_map_plus, axis=0)
    #     self.one_map = np.roll(self.one_map, 46, axis=0)
    #
    #     self.map_date = self.dates[self.timevar][0]
    #     self.map_date = str(self.map_date)
    #     # self.map_date += ' ' + hours[hour]
    #     if not yearin: self.map_date = None

    # Functions for date handling:

    def _create_date_objects(self, *args, **kwargs):
        """Create datetime date objects."""

        self.dates = {}

        try:
            varlist = args[0]
        except IndexError:
            varlist = ['time']

        for var in varlist:
            self._get_base_date(var, **kwargs)
            self._convert_time_to_dates(var)

    def _get_base_date(self, var, **kwargs):
        """Get the base date (the date in 'hours since ...')."""

        self.basedate = {}

        datestring = self.dataset.variables[var].units
        yy, mm, dd = self._parse_date(datestring, **kwargs)

        self.basedate[var] = date(yy, mm, dd)

    def _parse_date(self, datestring, **kwargs):
        """Parse a date string."""

        prefix = kwargs.get('prefix')
        if prefix == None: prefix = 'hours since '

        m = re.search(prefix + '(\d+)-(\d+)-(\d+).*', datestring)
        if m == None: print ('ReadNetcdf expects time units of the form: hours since y-m-d ...')

        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    def _convert_time_to_dates(self, var):
        """Convert days since base date to datetime.date."""

        self.dates[var] = []

        for hours in self.dataset.variables[var]:
            self.dates[var].append(self.basedate[var] + timedelta(hours=float(hours)))

    # Function for printing information:

    def _print_dataset_info(self, **kwargs):
        """ Print a list of important dataset and provenance information."""

        if kwargs.get('summary'):
            print ('Dataset summary:')
            print (self.dataset)
            print ('Variables in dataset:')
            for var in self.dataset.variables:
                print (var, self.dataset.variables[var])

        attrib_list = ['title', 'institution', 'project', 'creator_url', 'creator_email', 'summary']
        for attrib in attrib_list:
            if hasattr(self.dataset, attrib):
                print (attrib.title() + ': ' + getattr(self.dataset, attrib))
            else:
                print ('Cannot find: ' + attrib.title())
        print()

    def cut(self, lats, lons):
        """Cut out a sub-dataset according to lats and lons."""

        self.cut_latitude = []
        self.cut_longitude = []
        lat_indices = []
        lon_indices = []

        if self.type == 'ccmp':
            index = 0
            for i in self.dataset.variables['latitude']:
                if i >=lats[0] and i <= lats[1]:
                    self.cut_latitude.append(i)
                    lat_indices.append(index)
                index = index + 1
            index = 0
            for i in self.dataset.variables['longitude']:
                if i >=lons[0] and i <= lons[1]:
                    self.cut_longitude.append(i)
                    lon_indices.append(index)
                index = index + 1

            self.cut_uwnd = np.zeros((len(lat_indices), len(lon_indices)))
            self.cut_vwnd = np.zeros((len(lat_indices), len(lon_indices)))
            self.cut_wspd = np.zeros((len(lat_indices), len(lon_indices)))
            self.cut_wdir = np.zeros((len(lat_indices), len(lon_indices)))

            temp_uwnd = np.array(np.squeeze(self.dataset.variables['uwnd'], axis=0))
            temp_vwnd = np.array(np.squeeze(self.dataset.variables['vwnd'], axis=0))

            lat_index = 0
            for i in lat_indices:
                lon_index = 0
                for j in lon_indices:
                    self.cut_uwnd[lat_index][lon_index] = temp_uwnd[i][j]
                    self.cut_vwnd[lat_index][lon_index] = temp_vwnd[i][j]
                    self.cut_wspd[lat_index][lon_index] = math.sqrt(temp_uwnd[i][j]**2 + temp_vwnd[i][j]**2)
                    self.cut_wdir[lat_index][lon_index] = math.atan(temp_vwnd[i][j]/temp_uwnd[i][j])*180/math.pi
                    if self.cut_wdir[lat_index][lon_index] < 0:
                        self.cut_wdir[lat_index][lon_index] = self.cut_wdir[lat_index][lon_index] + 360
                    lon_index = lon_index + 1
                lat_index = lat_index + 1

        elif re.search(self.type, 'icoads'):
            index = 0
            for i in self.dataset.variables['lat'][::-1]:
                if i >= lats[0] and i <= lats[1]:
                    self.cut_latitude.append(i)
                    lat_indices.append(index)
                index = index + 1
            index = 0
            for i in self.dataset.variables['lon']:
                if i >= lons[0] and i <= lons[1]:
                    self.cut_longitude.append(i)
                    lon_indices.append(index)
                index = index + 1
            if self.type == 'icoads-wspd':
                self.cut_wspd = np.zeros((len(lat_indices), len(lon_indices)))
                temp_wspd = np.array(self.dataset.variables['wspd'][572+self.time])

                lat_index = 0
                for i in lat_indices:
                    lon_index = 0
                    for j in lon_indices:
                        self.cut_wspd[lat_index][lon_index] = temp_wspd[i][j]
                        lon_index = lon_index + 1
                    lat_index = lat_index + 1

            elif self.type == 'icoads-u':
                self.cut_uwnd = np.zeros((len(lat_indices), len(lon_indices)))
                temp_uwnd = np.array(self.dataset.variables['uwnd'][572+self.time])

                lat_index = 0
                for i in lat_indices:
                    lon_index = 0
                    for j in lon_indices:
                        self.cut_uwnd[lat_index][lon_index] = temp_uwnd[i][j]
                        lon_index = lon_index + 1
                    lat_index = lat_index + 1

            elif self.type == 'icoads-v':
                self.cut_vwnd = np.zeros((len(lat_indices), len(lon_indices)))
                temp_vwnd = np.array(self.dataset.variables['vwnd'][572+self.time])

                lat_index = 0
                for i in lat_indices:
                    lon_index = 0
                    for j in lon_indices:
                        self.cut_vwnd[lat_index][lon_index] = temp_vwnd[i][j]
                        lon_index = lon_index + 1
                    lat_index = lat_index + 1


def match(dataset_x, dataset_y):
    """Match two datasets. Note: the resolution of x should be larger than y's"""
    match_list_x = []
    match_list_y = []
    for i in dataset_x.lat:
        for j in dataset_x.lon:
            if dataset_x.wspd[dataset_x.lat.index(i)][dataset_x.lon.index(j)] == 32766:
                continue
            tmp_match_list_y = []
            last_lat = False
            for k in dataset_y.lat:
                dist_lat = abs(k-i)
                if last_lat and dist_lat >= 1: break
                elif dist_lat >= 1: continue
                last_lon = False
                for l in dataset_y.lon:
                    dist_lon = abs(l-j)
                    if last_lon and dist_lon >= 1: break
                    elif dist_lon >= 1: continue
                    # tmp_match_list_x.append(dataset_x.wspd[dataset_x.lat.index(i)][dataset_x.lon.index(j)])
                    tmp_match_list_y.append(dataset_y.wspd[dataset_y.lat.index(k)][dataset_y.lon.index(l)])
                    last_lon = True
                last_lat = True
            match_list_y.append(np.mean(np.array(tmp_match_list_y)))
            match_list_x.append(dataset_x.wspd[dataset_x.lat.index(i)][dataset_x.lon.index(j)])

    return match_list_x, match_list_y

class CuttedData:
    pass


if __name__ == '__main__':
    lats = (0, 45)
    lons = (99, 132)
    # dataset = ReadNetcdf('month_20070301_v11l35flk.nc', summary=True)
    dataset1 = ReadNetcdf('datasets/era/200812.nc')
    print ('ok')
    # dataset1.cut(lats=lats, lons=lons)
    # ccmp = CuttedData()
    # ccmp.lat = dataset1.cut_latitude
    # ccmp.lon = dataset1.cut_longitude
    # ccmp.wspd = dataset1.cut_wspd
    # ccmp.uwnd = dataset1.cut_uwnd
    # ccmp.vwnd = dataset1.cut_vwnd
    # dataset2 = ReadNetcdf('wspd.mean.nc', type='ICOADS-wspd', time=1)
    # dataset2.cut(lats=lats, lons=lons)
    # icoads = CuttedData()
    # icoads.lat = dataset2.cut_latitude
    # icoads.lon = dataset2.cut_longitude
    # icoads.wspd = dataset2.cut_wspd
    # # icoads.uwnd = dataset1.cut_uwnd
    # # icoads.vwnd = dataset1.cut_vwnd
    # match_list_x, match_list_y = match(icoads, ccmp)
    # # print match_list
    #
    # fig = plt.figure(1)
    # plt.scatter(match_list_x, match_list_y)
    # fig.savefig('scatter.png')
    # fig = plt.figure()
    # m = Basemap(projection='mill', lat_ts=10, llcrnrlon=lons[0], urcrnrlon=lons[1], llcrnrlat=lats[0], urcrnrlat=lats[1],
    #             resolution='c')
    #
    # m.drawcoastlines(linewidth=0.25)
    # m.drawcountries(linewidth=0.25)
    # m.fillcontinents(color='coral', lake_color='aqua')
    # aximg = m.imshow(ccmp.wspd, cmap=plt.cm.jet, vmin=0, vmax=10)
    #
    # cbar = fig.colorbar(aximg, orientation='vertical')
    # cbar.set_label('m s-1')
    #
    # # x, y = m(lon, lat)
    # # cs = m.contour(x,y,data,15,linewidths=1.5)
    #
    #
    # # print 'lon:', lon, 'x:', x
    # # print 'lat:', lat, 'y:', y
    # m.drawmeridians(np.arange(99, 132, 3))
    #
    # m.drawparallels(np.arange(0, 45, 3))
    # plt.title('10 metre wind speed')
    # # plt.xticks(np.arange(100, 131, 3))
    # # plt.yticks(np.arange(10, 45, 3))
    # # plt.xlim(lons)
    # # plt.ylim(lats)
    # fig.savefig('test_compare1.png')
    #
    # plt.close()

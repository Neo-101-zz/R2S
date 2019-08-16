from ascat_averaged import ASCATAveraged
from quikscat_averaged_v4 import QuikScatAveraged
from windsat_averaged_v7 import WindSatAveraged
from amsre_averaged_v7 import AMSREaveraged

import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

class ReadBytemap:

    def __init__(self, *args, **kwargs):
        """Read the dataset."""

        """
            Arguments:
            filename (required)

            Keyword arguments (optional):
            summary = True, if you want a full report of the file contents

            Returns:
            a read bytemap dataset instance
        """
        self.filename = args[0]

        self.type = kwargs.get('type')
        if self.type == 'ascat':
            self.dataset = ASCATAveraged(self.filename)
            self.wspd = self.dataset.variables['windspd']
            self.wdir = self.dataset.variables['winddir']

        elif self.type == 'qscat':
            self.dataset = QuikScatAveraged(self.filename)
            self.wspd = self.dataset.variables['windspd']
            self.wdir = self.dataset.variables['winddir']
        elif self.type == 'windsat':
            self.dataset = WindSatAveraged(self.filename)
            self.wspd = self.dataset.variables['w-aw']
            self.wspd_lf = self.dataset.variables['w-lf']
            self.wspd_mf = self.dataset.variables['w-mf']
            self.wdir = self.dataset.variables['wdir']
        elif self.type == 'amsre':
            self.dataset = AMSREaveraged(self.filename)
            self.wspd_lf = self.dataset.variables['windLF']
            self.wspd_mf = self.dataset.variables['windMF']
        else:
            print 'Wrong type selected.'
            print

        self.missing = self.dataset.variables['nodata']
        self.latitude = self.dataset.variables['latitude']
        self.longitude = self.dataset.variables['longitude']

        print 'Dataset read successfully.'
        print

        self._print_dataset_info(**kwargs)

    def _print_dataset_info(self, **kwargs):
        """ Print a list of important dataset and provenance information."""

        if kwargs.get('summary'):
            print 'Dataset summary:'
            print self.dataset
            self._show_dimensions()
            self._show_variables()
            self._show_validrange()

    def _show_dimensions(self):
        print('')
        print('Dimensions')
        for dim in self.dataset.dimensions:
            aline = ' '.join([' ' * 3, dim, ':', str(self.dataset.dimensions[dim])])
            print(aline)

    def _show_variables(self):
        print('')
        print('Variables:')
        for var in self.dataset.variables:
            aline = ' '.join([' ' * 3, var, ':', self.dataset.variables[var].long_name])
            print(aline)

    def _show_validrange(self):
        print('')
        print('Valid min and max and units:')
        for var in self.dataset.variables:
            aline = ' '.join([' ' * 3, var, ':',
                              str(self.dataset.variables[var].valid_min), 'to',
                              str(self.dataset.variables[var].valid_max),
                              '(', self.dataset.variables[var].units, ')'])
            print(aline)

    def cut(self, lats, lons):
        """Cut out a sub-dataset according to lats and lons."""

        self.cut_latitude = []
        self.cut_longitude = []
        lat_indices = []
        lon_indices = []
        index = 0
        for i in self.latitude:
            if i >= lats[0] and i <= lats[1]:
                self.cut_latitude.append(i)
                lat_indices.append(index)
            index = index + 1
        index = 0
        for i in self.longitude:
            if i >= lons[0] and i <= lons[1]:
                self.cut_longitude.append(i)
                lon_indices.append(index)
            index = index + 1

        if self.type == 'ascat' or self.type == 'qscat' or self.type == 'windsat':
            self.cut_wdir = np.zeros((len(lat_indices), len(lon_indices)))
            self.cut_wspd = np.zeros((len(lat_indices), len(lon_indices)))
            self.cut_missing = np.zeros((len(lat_indices), len(lon_indices)))
            lat_index = 0
            for i in lat_indices:
                lon_index = 0
                for j in lon_indices:
                    self.cut_missing[lat_index][lon_index] = self.missing[i][j]
                    if self.cut_missing[lat_index][lon_index]:
                        self.cut_wspd[lat_index][lon_index] = 999
                        self.cut_wdir[lat_index][lon_index] = 999
                    else:
                        self.cut_wspd[lat_index][lon_index] = self.wspd[i][j]
                        self.cut_wdir[lat_index][lon_index] = self.wdir[i][j]

                    lon_index = lon_index + 1
                lat_index = lat_index + 1

        elif self.type == 'amsre':
            temp_cut_wspd_lf = np.zeros((len(lat_indices), len(lon_indices)))
            temp_cut_wspd_mf = np.zeros((len(lat_indices), len(lon_indices)))
            self.cut_wspd = np.zeros((len(lat_indices), len(lon_indices)))
            self.cut_missing = np.zeros((len(lat_indices), len(lon_indices)))
            lat_index = 0
            for i in lat_indices:
                lon_index = 0
                for j in lon_indices:
                    self.cut_missing[lat_index][lon_index] = self.missing[i][j]
                    if self.cut_missing[lat_index][lon_index]:
                        temp_cut_wspd_lf[lat_index][lon_index] = 999
                        temp_cut_wspd_mf[lat_index][lon_index] = 999
                    else:
                        temp_cut_wspd_lf[lat_index][lon_index] = self.wspd_lf[i][j]
                        temp_cut_wspd_mf[lat_index][lon_index] = self.wspd_mf[i][j]

                    lon_index = lon_index + 1
                lat_index = lat_index + 1
            self.cut_wspd = (temp_cut_wspd_lf + temp_cut_wspd_mf) / 2

class CuttedData:
    pass

if __name__ == '__main__':
    lats = (0, 45)
    lons = (99, 132)
    dataset = ReadBytemap('datasets/WindSat/20079.gz', type='windsat', summary=True)
    # dataset.cut(lats=lats, lons=lons)
    # # print dataset.dataset.variables['windspd'].shape
    # ascat = CuttedData()
    # ascat.lat = dataset.cut_latitude
    # ascat.lon = dataset.cut_longitude
    # ascat.wspd = dataset.cut_wspd
    # ascat.wdir = dataset.cut_wdir

    # fig = plt.figure()
    # m = Basemap(projection='mill', lat_ts=10, llcrnrlon=lons[0], urcrnrlon=lons[1], llcrnrlat=lats[0],
    #             urcrnrlat=lats[1],
    #             resolution='c')
    #
    # m.drawcoastlines(linewidth=0.25)
    # m.drawcountries(linewidth=0.25)
    # m.fillcontinents(color='coral', lake_color='aqua')
    # aximg = m.imshow(ascat.wspd, cmap=plt.cm.jet, vmin=0, vmax=10)

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
    plt.xticks(np.arange(100, 131, 3))
    plt.yticks(np.arange(10, 45, 3))
    # # plt.xlim(lons)
    # # plt.ylim(lats)
    # fig.savefig('test_ascat.png')

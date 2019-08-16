import pickle
import os

base_dir = '/Users/zhangdongxiang/PycharmProjects/data4all/'

def match_with_ndbc(month, day, dataset, ndbc, space_d, time_d):
    """Match ASCAT/QSCAT data with NDBC data.

    Parameters
    ----------
    dataset : list of dict

    ndbc : list of dict

    Returns
    -------

    Questions
    ---------
    Why not include field 'scatflag' which stands for 'Scattermeter
    Rain Flag' to cover rain information like function that matchs
    Windsat data with NDBC data ?
    """
    wspd_list_dataset = []
    wspd_list_ndbc = []
    wdir_list_dataset = []
    wdir_list_ndbc = []
    if len(dataset) and len(ndbc):
        lat = ndbc[0]['lat']
        lon = ndbc[0]['lon']
        cnt = 1
        print(month, day)
        for x in ndbc:
            if x['month'] != month or x['day'] != day:
                continue
            for y in dataset:
                # Spatial filter
                if abs(lat-y['lat']) >= space_d or abs(lon-y['lon']) >= space_d:
                    continue
                # Temporal filter
                if abs(x['time'] - y['time']) < time_d:
                    print('ok. ', cnt)
                    wspd_list_ndbc.append(x['wspd'])
                    wspd_list_dataset.append(y['wspd'])
                    wdir_list_ndbc.append(x['wdir'])
                    wdir_list_dataset.append(y['wdir'])
                    # rain_list.append(y['rain'])
                    # match_point1 = (x['wspd'], y['wspd'])
                    # match_point2 = (x['wdir'], y['wdir'])
                    # match_list.append((match_point1, match_point2))
                    cnt += 1
    return wspd_list_ndbc, wspd_list_dataset, wdir_list_ndbc, wdir_list_dataset

def match(year, month, satellite, ndbc, space_d, time_d):
    """Match Windsat data with NDBC data.

    """
    match_list = []
    if len(satellite) and len(ndbc):
        lat = ndbc[0]['lat']
        lon = ndbc[0]['lon']
        cnt = 1
        for x in ndbc:
            if x['year'] != year or x['month'] != month:
                continue
            for y in satellite:
                # print('lat: '+str(lat)+';'+'lon: '+str(lon))
                # print('y_lat: '+str(y['lat'])+';'+'y_lon: '+str(y['lon']))

                # Spatial filter
                if abs(lat-y['lat']) >= space_d or abs(lon-y['lon']) >= space_d:
                    continue
                # Temporal filter
                if x['day'] == y['day'] and abs(x['time'] - y['time']) < time_d:
                    print('ok. ', cnt)
                    match_point = {}
                    match_point['a_wspd'] = x['wspd']
                    # w-aw WSPD_AW All-weather 10-meter wind speed
                    match_point['b_wspd'] = y['w-aw']
                    match_point['a_wdir'] = x['wdir']
                    match_point['b_wdir'] = y['wdir']
                    match_point['rain'] = y['rain']
                    match_list.append(match_point)
                    cnt += 1
    return match_list


if __name__ == '__main__':
    # month_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    # files_ndbc = os.listdir('./ndbc/pickle')
    # for file in files_ndbc:
    #     if not os.path.isdir(file):
    #         print('read ndbc '+file)
    #         pickle_ndbc = open('./ndbc/pickle/'+file, 'rb')
    #         ndbc = pickle.load(pickle_ndbc)
    #         print('read satellite')
    #         wspd_lists_ndbc = []
    #         wspd_lists_ascat = []
    #         wdir_lists_ndbc = []
    #         wdir_lists_ascat = []
    #         for month in month_list:
    #             pickle_ascat = open('./satellite/pickle/ascat/ascat_'+month+'_data.pkl', 'rb')
    #             ascat = pickle.load(pickle_ascat)
    #             if month == '02':
    #                 days = 29
    #             elif month == '04' or month == '06' or month == '09' or month == '11':
    #                 days = 30
    #             else:
    #                 days = 31
    #             for day in range(1, days+1):
    #                 wspd_list_ndbc, wspd_list_ascat, wdir_list_ndbc, wdir_list_ascat = match_with_ndbc(month, day, ascat[day], ndbc, 0.25/2, 5)
    #                 wspd_lists_ndbc = wspd_lists_ndbc + wspd_list_ndbc
    #                 wspd_lists_ascat = wspd_lists_ascat + wspd_list_ascat
    #                 wdir_lists_ndbc = wdir_lists_ndbc + wdir_list_ndbc
    #                 wdir_lists_ascat = wdir_lists_ascat + wdir_list_ascat
    #         match_dataset = {}
    #         match_dataset['wspd_ndbc'] = wspd_lists_ndbc
    #         match_dataset['wspd_ascat'] = wspd_lists_ascat
    #         match_dataset['wdir_ndbc'] = wdir_lists_ndbc
    #         match_dataset['wdir_ascat'] = wdir_lists_ascat
    #         pickle_file = open('./pickle/ascat/'+file[0:5]+'_ascat.pkl', 'wb')
    #         pickle.dump(match_dataset, pickle_file)
    #         pickle_file.close()

    # month_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    # files_ndbc = os.listdir('./ndbc/pickle')
    # for file in files_ndbc:
    #     if not os.path.isdir(file):
    #         print('read ndbc ' + file)
    #         pickle_ndbc = open('./ndbc/pickle/' + file, 'rb')
    #         ndbc = pickle.load(pickle_ndbc)
    #         print('read satellite')
    #         wspd_lists_ndbc = []
    #         wspd_lists_qscat = []
    #         wdir_lists_ndbc = []
    #         wdir_lists_qscat = []
    #         for month in month_list:
    #             pickle_qscat = open('./satellite/pickle/qscat/qscat_' + month + '_data.pkl', 'rb')
    #             qscat = pickle.load(pickle_qscat)
    #             if month == '02':
    #                 days = 29
    #             elif month == '04' or month == '06' or month == '09' or month == '11':
    #                 days = 30
    #             else:
    #                 days = 31
    #             for day in range(1, days + 1):
    #                 wspd_list_ndbc, wspd_list_qscat, wdir_list_ndbc, wdir_list_qscat = match_with_ndbc(month, day,
    #                                                                                                    qscat[day], ndbc,
    #                                                                                                    0.25 / 2, 5)
    #                 wspd_lists_ndbc = wspd_lists_ndbc + wspd_list_ndbc
    #                 wspd_lists_qscat = wspd_lists_qscat + wspd_list_qscat
    #                 wdir_lists_ndbc = wdir_lists_ndbc + wdir_list_ndbc
    #                 wdir_lists_qscat = wdir_lists_qscat + wdir_list_qscat
    #         match_dataset = {}
    #         match_dataset['wspd_ndbc'] = wspd_lists_ndbc
    #         match_dataset['wspd_qscat'] = wspd_lists_qscat
    #         match_dataset['wdir_ndbc'] = wdir_lists_ndbc
    #         match_dataset['wdir_qscat'] = wdir_lists_qscat
    #         pickle_file = open('./pickle/qscat/' + file[0:5] + '_qscat.pkl', 'wb')
    #         pickle.dump(match_dataset, pickle_file)
    #         pickle_file.close()


    """match for all period"""
    # qscat_year_list = ['2003', '2004', '2005', '2006', '2007', '2008', '2009']
    # for year in qscat_year_list:
    #     ndbc_dir = base_dir+'ndbc/pickle/'+year
    #     ndbc_files = os.listdir(ndbc_dir)
    #     ndbc_files.sort()
    #     dir = base_dir + 'qscat/pickle/' + year
    #     files = os.listdir(dir)
    #     files.sort()
    #     for ndbc_file in ndbc_files:
    #         id = ndbc_file[0:5]
    #         pickle_ndbc = open(ndbc_dir + '/' + ndbc_file, 'rb')
    #         ndbc_data = pickle.load(pickle_ndbc)
    #         for file in files:
    #             month = int(file[6:-4])
    #             pickle_qscat = open(dir + '/' + file, 'rb')
    #             qscat_data = pickle.load(pickle_qscat)
    #             print('start to match. '+id+' and month: '+str(month))
    #             match_list = match(year, month, qscat_data, ndbc_data, 0.25/2, 5)
    # 
    #             pickle_file = open(base_dir+'match/qscat/'+id+'_'+year+'_'+str(month)+'.pkl', 'wb')
    #             pickle.dump(match_list, pickle_file)
    #             pickle_file.close()

    windsat_year_list = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
    for year in windsat_year_list:
        ndbc_dir = base_dir + 'ndbc/pickle/' + year
        ndbc_files = os.listdir(ndbc_dir)
        ndbc_files.sort()
        dir = base_dir + 'windsat/pickle/' + year
        files = os.listdir(dir)
        files.sort()
        for ndbc_file in ndbc_files:
            id = ndbc_file[0:5]
            pickle_ndbc = open(ndbc_dir + '/' + ndbc_file, 'rb')
            ndbc_data = pickle.load(pickle_ndbc)
            for file in files:
                month = int(file[8:-4])
                pickle_windsat = open(dir + '/' + file, 'rb')
                windsat_data = pickle.load(pickle_windsat)
                print('start to match. ' + id + ' and month: ' + str(month))
                match_list = match(year, month, windsat_data, ndbc_data, 0.25 / 2, 5)

                pickle_file = open(base_dir + 'match/windsat/' + year+'/' + id + '_' + year + '_' + str(month) + '.pkl', 'wb')
                pickle.dump(match_list, pickle_file)
                pickle_file.close()





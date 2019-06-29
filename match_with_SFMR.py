import pickle
import os

base_dir = '/Users/zhangdongxiang/PycharmProjects/data4all/'

def match_with_sfmr_ascat(month, day, dataset, sfmr, space_d, time_d):
    match_list = []
    sfmr_len = len(sfmr['wspd'])
    for i in range(sfmr_len):
        lat = sfmr['lat'][i]
        lon = sfmr['lon'][i]
        cnt = 1
        for y in dataset:
            if abs(lat - y['lat']) >= space_d or abs(lon - y['lon']) >= space_d:
                continue
            if abs(sfmr['time'][i] - y['time']) < time_d:
                print('ok. ', cnt)
                match_point = {}
                match_point['a_wspd'] = sfmr['wspd'][i]
                match_point['b_wspd'] = y['wspd']
                match_list.append(match_point)
                cnt += 1

    return match_list


def match_with_sfmr_windsat(month, day, dataset, sfmr, space_d, time_d):
    match_list = []
    sfmr_len = len(sfmr['wspd'])
    for i in range(sfmr_len):
        lat = sfmr['lat'][i]
        lon = sfmr['lon'][i]
        cnt = 1
        for y in dataset:
            if abs(lat - y['lat']) >= space_d or abs(lon - y['lon']) >= space_d:
                continue
            if abs(sfmr['time'][i] - y['time']) < time_d:
                print('ok. ', cnt)
                match_point = {}
                match_point['a_wspd'] = sfmr['wspd'][i]
                match_point['b_wspd'] = y['w-aw']
                match_list.append(match_point)
                cnt += 1

    return match_list


def match(year, month, satellite, ndbc, space_d, time_d):
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
                if abs(lat-y['lat']) >= space_d or abs(lon-y['lon']) >= space_d:
                    continue
                if x['day'] == y['day'] and abs(x['time'] - y['time']) < time_d:
                    print('ok. ', cnt)
                    match_point = {}
                    match_point['a_wspd'] = x['wspd']
                    match_point['b_wspd'] = y['w-aw']
                    match_point['a_wdir'] = x['wdir']
                    match_point['b_wdir'] = y['wdir']
                    match_point['rain'] = y['rain']
                    match_list.append(match_point)
                    cnt += 1
    return match_list

dir = base_dir + 'SFMR/pickle/'
files = os.listdir(dir)
files.sort()
for file in files:
    if file == '.DS_Store':
        continue
    year = file[0:4]
    month = int(file[4:6])
    day = int(file[6:8])
    print(str(month))
    pickle_sfmr = open(dir + file, 'rb')
    sfmr = pickle.load(pickle_sfmr)
    pickle_ascat = open(base_dir + 'ascat/pickle/' + year + '/ascat_' + str(month) + '.pkl', 'rb')
    ascat_temp = pickle.load(pickle_ascat)
    ascat = []
    for a in ascat_temp:
        if a['day'] == day:
            ascat.append(a)
    pickle_windsat = open(base_dir + 'windsat/pickle/' + year + '/windsat_' + str(month) + '.pkl', 'rb')
    windsat_temp = pickle.load(pickle_windsat)
    windsat = []
    for w in windsat_temp:
        if w['day'] == day:
            windsat.append(w)

    print('ASCAT start to match with SMFR. ' + file)
    print(len(sfmr['time']))
    print(len(ascat))
    match_list_ascat = match_with_sfmr_ascat(month, day, ascat, sfmr, 0.25/2, 180)
    pickle_file = open(base_dir + 'match/SFMR/ascat_' + file + '.pkl', 'wb')
    pickle.dump(match_list_ascat, pickle_file)
    pickle_file.close()
    print('WINDSAT start to match with SMFR. ' + file)
    print(len(windsat))
    match_list_windsat = match_with_sfmr_windsat(month, day, windsat, sfmr, 0.25 / 2, 180)
    pickle_file = open(base_dir + 'match/SFMR/windsat_' + file + '.pkl', 'wb')
    pickle.dump(match_list_windsat, pickle_file)
    pickle_file.close()

    

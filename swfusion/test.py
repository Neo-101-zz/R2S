import pickle

from netCDF4 import Dataset

data = \
        Dataset('/Users/lujingze/Programming/SWFusion/data/ibtracs/IBTrACS_since1980_v04r00.nc')
vars = data.variables
name = vars['name']
basin = vars['basin']
season = vars['season']

storm_num, date_time = basin.shape[0], basin.shape[1]

names_2013 = []
for i in range(storm_num):
    if season[i] != 2013:
        continue
    basin_ = basin[i][0][basin[i][0].mask == False].tostring().decode('utf-8')
    if basin_ != 'NA':
        continue
    name_ = name[i][name[i].mask == False].tostring().decode('utf-8')
    if name_ != 'NOT_NAMED':
        names_2013.append(name_)

for name_ in sorted(names_2013):
    print(name_)

with open('/Users/lujingze/Programming/SWFusion/data/hwind/variable/year_tc.pkl', 'rb') as file:
    year_tc = pickle.load(file)

for year in year_tc.keys():
    print(year, end='\n')
    for tc_info in year_tc[year]:
        print(f'{tc_info.basin} {tc_info.name}')
    print()

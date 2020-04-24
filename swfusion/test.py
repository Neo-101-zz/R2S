import datetime
import os
import pickle

import pandas as pd

with open('../compare_info_na_sfmr_smap.txt', 'r') as f:
    infos = list(f)

tc_names = []
tc_dts = []
match = []
for info in infos:
    tc_name = info.split(' TC ')[1].split(' on ')[0].replace(' ', '')
    dt_str = info.split(' on ')[1].strip()
    dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

    tc_names.append(tc_name)
    tc_dts.append(dt)

    if info.startswith('Comparing'):
        match.append(True)
    elif info.startswith('Skiping'):
        match.append(False)

hit_dict = {'TC_name': tc_names,
            'datetime': tc_dts,
            'match': match,
           }
hit_df = pd.DataFrame(hit_dict)

dir = '../statistic/match_of_data_sources/sfmr_vs_smap/na/'
os.makedirs(dir, exist_ok=True)
hit_df.to_pickle(f'{dir}na_match_sfmr_vs_smap.pkl')

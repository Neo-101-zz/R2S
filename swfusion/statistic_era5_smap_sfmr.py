import logging
import math
import pickle

import pandas as pd
from sklearn.metrics import mean_squared_error

class Statisticer(object):

    def __init__(self, CONFIG, period, basin):
        self.logger = logging.getLogger(__name__)
        self.CONFIG = CONFIG
        self.period = period
        self.basin = basin

        self.bias_root_dir = self.CONFIG['result']['dirs']['statistic'][
            'windspd_bias_to_sfmr']
        self.sources = ['era5', 'smap']

        self.comapre_era5_smap_by_sfmr()

    def comapre_era5_smap_by_sfmr(self):
        bias = dict()
        for src in self.sources:
            name = f'{self.basin}'
            for dt in self.period:
                name = f'{name}_{dt.strftime("%Y%m%d%H%M%S")}'
            dir = f'{self.bias_root_dir}{src}_vs_sfmr/{name}/'
            with open(f'{dir}{name}.pkl', 'rb') as f:
                bias[src] = pickle.load(f)

        overlay_era5_indices = []
        for i in range(len(bias['era5'])):
            if (bias['era5']['sfmr_dt'].values[i]
                not in bias['smap']['sfmr_dt'].values):
                continue
            else:
                overlay_era5_indices.append(i)

        if len(overlay_era5_indices) != len(bias['smap']):
            self.logger.error('Not all SMAP bias overlay with ERA5 bias')
            breakpoint()
            exit()

        df_list = []
        for idx in overlay_era5_indices:
            one_row_df = bias['era5'].loc[[idx]]
            df_list.append(one_row_df)

        overlay_era5_bias = pd.concat(df_list).reset_index(drop=True)

        name_list = ['All ERA5', 'Overlay SMAP',
                     'Overlay ERA5']
        bias_list = [bias['era5'], bias['smap'], overlay_era5_bias]

        for name, df in zip(name_list, bias_list):
            print(name)
            print('-' * len(name))
            windspd_bias = df['windspd_bias']
            print(f'Count: {len(df)}')
            print(f'Max bias: {windspd_bias.max()}')
            print(f'Min bias: {windspd_bias.min()}')
            print(f'Mean bias: {windspd_bias.mean()}')
            print(f'Median bias: {windspd_bias.median()}')

            truth = df['sfmr_windspd']
            observation = df['tgt_windspd']
            mse = mean_squared_error(truth, observation)
            print(f'RMSE: {math.sqrt(mse)}')
            print()

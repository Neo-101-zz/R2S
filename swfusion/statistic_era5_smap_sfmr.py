import logging
import math
import pickle

import pandas as pd
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import utils

class Statisticer(object):

    def __init__(self, CONFIG, period, basin, sources):
        self.logger = logging.getLogger(__name__)
        self.CONFIG = CONFIG
        self.period = period
        self.basin = basin

        self.bias_root_dir = self.CONFIG['result']['dirs']['statistic'][
            'windspd_bias_to_sfmr']
        self.sources = sources

        if set(sources) == set(['smap', 'era5']):
            self.predict_smap = False
            self.compare_era5_smap_by_sfmr()
        elif set(sources) == set(['smap_prediction', 'era5']):
            self.predict_smap = True
            self.compare_era5_smap_prediction_by_sfmr()

    def compare_era5_smap_by_sfmr(self):
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

        # self.show_simple_statistic(name_list, bias_list)
        self.plot_scatter_regression(bias_list[1:])

    def compare_era5_smap_prediction_by_sfmr(self):
        bias = dict()
        sfmr_dt = dict()
        sfmr_dt_name = 'sfmr_dt'
        for src in self.sources:
            name = f'{self.basin}'
            for dt in self.period:
                name = f'{name}_{dt.strftime("%Y%m%d%H%M%S")}'
            dir = f'{self.bias_root_dir}{src}_vs_sfmr/{name}/'
            with open(f'{dir}{name}.pkl', 'rb') as f:
                bias[src] = pickle.load(f)
                sfmr_dt[src] = bias[src][sfmr_dt_name]

        sfmr_dt_intersection = set(sfmr_dt['smap_prediction']) & \
                set(sfmr_dt['era5'])
        tmp = set()
        for ts in sfmr_dt_intersection:
            tmp.add(ts.to_datetime64())
        sfmr_dt_intersection = tmp

        for src in self.sources:
            drop_row_indices = []
            for i in range(len(bias[src])):
                if (bias[src][sfmr_dt_name].values[i]
                    not in sfmr_dt_intersection):
                    #
                    drop_row_indices.append(i)
            bias[src].drop(drop_row_indices, inplace=True)

        len_smap_p = len(bias['smap_prediction'])
        len_era5 = len(bias['era5'])
        if len_smap_p != len_era5:
            self.logger.error((f"""Fail intersecting: """
                               f"""smap_prediction({len_smap_p}) """
                               f"""!= era5({len_era5})"""))
            breakpoint()
            exit(1)

        name_list = ['SMAP prediction', 'ERA5']
        bias_list = [bias['smap_prediction'], bias['era5']]

        self.show_simple_statistic(name_list, bias_list)
        self.plot_scatter_regression(bias_list)

    def plot_scatter_regression(self, bias_list):
        sns.set(style="ticks", color_codes=True)
        df = pd.concat(bias_list, ignore_index=True)
        sfmr_min = bias_list[0]['sfmr_windspd'].min()
        sfmr_max = bias_list[0]['sfmr_windspd'].max()

        era5_name = 'era5'
        if self.predict_smap:
            smap_x_name = 'smap_prediction'
        else:
            smap_x_name = 'smap'

        slope = dict()
        interpect = dict()
        r_value = dict()
        p_value = dict()
        std_err = dict()
        # get coeffs of linear fit
        for name in [smap_x_name, era5_name]:
            df_part = df.loc[df['tgt_name'] == name]
            slope[name], interpect[name], r_value[name], p_value[name],\
                    std_err[name] = stats.linregress(
                        df_part['sfmr_windspd'], df_part['tgt_windspd'])

        smap_x_equation = (f"""y={slope[smap_x_name]:.1f}x"""
                           f"""{interpect[smap_x_name]:+.1f}""")
        era5_equation = (f"""y={slope[era5_name]:.1f}x"""
                         f"""{interpect[era5_name]:+.1f}""")

        kws = dict(s=50, linewidth=.5, edgecolor="w")
        if self.predict_smap:
            pal = dict(era5="red", smap_prediction="blue")
            labels=[f'SMAP prediction: {smap_x_equation}']
        else:
            pal = dict(era5="red", smap="blue")
            labels=[f'SMAP: {smap_x_equation}']
        labels.append(f'ERA5: {era5_equation}')

        g = sns.lmplot(x="sfmr_windspd", y="tgt_windspd", hue="tgt_name",
                       data=df, markers=["o", "x"], palette=pal,
                       legend=False)
        g.despine(top=False, bottom=False, left=False, right=False)
        utils.const_line(sfmr_min, sfmr_max, 1, 0, 'green', 4, 'dashed')

        # g.add_legend()
        # leg = g._legend
        # leg.set_bbox_to_anchor([0.9, 0.125])
        # leg._loc = 4
        # leg._title = None
        # leg.edgecolor = 'black'
        plt.legend(title=None, loc='upper left', labels=labels)
        plt.xlabel('resampled SFMR wind speed (m/s)')
        plt.ylabel('matched wind speed (m/s)')

        plt.show()

    def plot_scatter_regression_old(self, bias_list):
        sns.set(style="ticks", color_codes=True)
        df = pd.concat(bias_list, ignore_index=True)
        sfmr_min = bias_list[0]['sfmr_windspd'].min()
        sfmr_max = bias_list[0]['sfmr_windspd'].max()

        kws = dict(s=50, linewidth=.5, edgecolor="w")
        if self.predict_smap:
            pal = dict(era5="red", smap_prediction="blue")
            labels=['SMAP prediction', 'ERA5']
        else:
            pal = dict(era5="red", smap="blue")
            labels=['SMAP', 'ERA5']

        g = sns.FacetGrid(df, hue='tgt_name', palette=pal, size=7,
                          despine=False, legend_out=False)
        g.map(plt.scatter, "sfmr_windspd", "tgt_windspd", **kws)
        g.map(sns.regplot, "sfmr_windspd", "tgt_windspd",
              ci=None, robust=1)
        utils.const_line(sfmr_min, sfmr_max, 1, 0, 'green', 'dashed')

        # g.add_legend()
        # leg = g._legend
        # leg.set_bbox_to_anchor([0.9, 0.125])
        # leg._loc = 4
        # leg._title = None
        # leg.edgecolor = 'black'
        plt.legend(title=None, loc='upper left', labels=labels)
        plt.xlabel('resampled SFMR wind speed (m/s)')
        plt.ylabel('matched wind speed (m/s)')

        plt.show()

    def show_simple_statistic(self, name_list, bias_list):
        for name, df in zip(name_list, bias_list):
            print(name)
            print('-' * len(name))
            windspd_bias = df['windspd_bias']
            print(f'Count: {len(df)}')
            print(f'Max bias: {windspd_bias.max()}')
            print(f'Min bias: {windspd_bias.min()}')
            print(f'Mean bias: {windspd_bias.mean()}')
            print(f'Median bias: {windspd_bias.median()}')
            print(f'Mean absolute bias: {windspd_bias.abs().mean()}')

            truth = df['sfmr_windspd']
            observation = df['tgt_windspd']
            mse = mean_squared_error(truth, observation)
            print(f'RMSE: {math.sqrt(mse)}')
            print()

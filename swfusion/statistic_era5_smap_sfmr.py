import logging
import math

from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

import utils

Base = declarative_base()


class Statisticer(object):

    def __init__(self, CONFIG, period, basin, sources, passwd):
        self.logger = logging.getLogger(__name__)
        self.CONFIG = CONFIG
        self.period = period
        self.basin = basin
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None

        self.logger = logging.getLogger(__name__)
        utils.setup_database(self, Base)

        self.period_str = []
        for dt in self.period:
            self.period_str.append(dt.strftime('%Y-%m-%d %H:%M:%S'))

        src_name_priority = {
            'smap_prediction': 1,
            'smap': 2,
            'era5': 3,
        }
        if len(sources) != 2:
            self.logger.error('Sources length should be 2')
            exit(1)
        self.sources = sorted(sources, key=src_name_priority.get)
        self.src_1 = self.sources[0]
        self.src_2 = self.sources[1]

        src_plot_names_mapper = {
            'smap_prediction': 'SMAP Prediction',
            'smap': 'SMAP',
            'era5': 'ERA5',
        }
        self.src_plot_names = [src_plot_names_mapper[src]
                               for src in self.sources]

        # self.show_hist_of_matchup()
        # breakpoint()
        self.compare_two_sources()

    def show_hist_of_matchup(self):
        # df = pd.read_sql('SELECT * FROM tc_sfmr_era5_na',
        #                  self.engine)
        # df.hist(column=['sfmr_windspd'], figsize = (12,10))
        # plt.show()

        all_basins = self.CONFIG['ibtracs']['urls'].keys()
        for basin in all_basins:
            table_name = f'tc_smap_era5_{basin}'
            if (utils.get_class_by_tablename(self.engine, table_name)
                    is not None):
                df = pd.read_sql(f'SELECT * FROM {table_name}',
                                 self.engine)
                df.hist(column=['smap_windspd'], figsize=(12, 10))
                plt.show()

    def compare_two_sources(self):
        bias = dict()
        sfmr_dt = dict()
        sfmr_dt_name = 'sfmr_datetime'

        for src in self.sources:
            tablename = utils.gen_validation_tablename(self, 'sfmr',
                                                       src)
            df = pd.read_sql(
                (f"""SELECT * FROM {tablename} """
                 f"""where sfmr_datetime > "{self.period_str[0]}" """
                 f"""and sfmr_datetime < "{self.period_str[1]}" """),
                self.engine)

            bias[src] = df
            sfmr_dt[src] = bias[src][sfmr_dt_name]

        sfmr_dt_intersection = set(sfmr_dt[self.src_1]) & \
            set(sfmr_dt[self.src_2])
        tmp = set()
        for ts in sfmr_dt_intersection:
            tmp.add(ts.to_datetime64())
        sfmr_dt_intersection = tmp

        for src in self.sources:
            drop_row_indices = []
            for i in range(len(bias[src])):
                if (bias[src][sfmr_dt_name].values[i]
                        not in sfmr_dt_intersection):
                    drop_row_indices.append(i)
            bias[src].drop(drop_row_indices, inplace=True)
            bias[src].drop_duplicates(subset='sfmr_datetime',
                                      inplace=True, ignore_index=True)

        len_src1 = len(bias[self.src_1])
        len_src2 = len(bias[self.src_2])
        if len_src1 != len_src2:
            self.logger.error((f"""Fail intersecting: """
                               f"""{self.src_1}({len_src1})"""
                               f""" != """
                               f"""{self.src_2}({len_src2})"""))
            breakpoint()
            exit(1)

        bias_list = [bias[src] for src in self.sources]
        bias_list = self.unify_df_colnames(bias_list)

        self.show_simple_statistic(bias_list)
        self.show_detailed_statistic(bias_list)
        self.plot_scatter_regression(bias_list)
        val_dir = self.CONFIG['results']['dirs']['fig'][
            'validation_by_sfmr']
        # utils.box_plot_windspd(val_dir

    def unify_df_colnames(self, bias_list):
        for i in range(len(bias_list)):
            name = self.sources[i]
            bias_list[i]['tgt_name'] = name
            bias_list[i].rename(columns={
                f'{name}_datetime': 'tgt_datetime',
                f'{name}_lon': 'tgt_lon',
                f'{name}_lat': 'tgt_lat',
                f'{name}_windspd': 'tgt_windspd'
            }, inplace=True)

        return bias_list

    def plot_scatter_regression(self, bias_list):
        sns.set(style="ticks", color_codes=True)
        sfmr_min = bias_list[0]['sfmr_windspd'].min()
        sfmr_max = bias_list[0]['sfmr_windspd'].max()

        df = pd.concat(bias_list, ignore_index=True)

        slope = dict()
        interpect = dict()
        r_value = dict()
        p_value = dict()
        std_err = dict()
        r2 = dict()
        equation_str = dict()

        def linefitline(src, x):
            return slope[src] * x + interpect[src]

        # get coeffs of linear fit
        for src in self.sources:
            df_part = df.loc[df['tgt_name'] == src]
            slope[src], interpect[src], r_value[src], \
                p_value[src], std_err[src] = stats.linregress(
                    df_part['sfmr_windspd'],
                    df_part['tgt_windspd'])
            r2[src] = r2_score(df_part['tgt_windspd'],
                               linefitline(src, df_part['tgt_windspd']))
            equation_str[src] = (f"""y={slope[src]:.1f}x"""
                                 f"""{interpect[src]:+.1f}""")

        print(f'R2: {r2}')

        pal = dict()
        labels = []
        colors = ['blue', 'red']
        for idx, src in enumerate(self.sources):
            pal[src] = colors[idx]
            labels.append((f"""{self.src_plot_names[idx]}: """
                           f"""{equation_str[src]} """
                           f"""(Corr Coeff: {r2[src]:.3f})"""))

        plt.rcParams['figure.figsize'] = (50.0, 30.0)

        g = sns.lmplot(x="sfmr_windspd", y="tgt_windspd",
                       hue="tgt_name", data=df, markers=["o", "x"],
                       palette=pal, legend=False,
                       size=7, aspect=1)
        g.despine(top=False, bottom=False, left=False, right=False)
        utils.const_line(sfmr_min, sfmr_max, 1, 0, 'green',
                         4, 'dashed')

        font_size = 15
        plt.ylim(0, 80)
        plt.legend(title=None, loc='upper left', labels=labels)
        plt.xlabel('resampled SFMR wind speed (m/s)',
                   fontsize=font_size)
        plt.ylabel('matched wind speed (m/s)', fontsize=font_size)
        plt.tick_params(labelsize=font_size)

        plt.show()

    def show_detailed_statistic(self, bias_list):
        windspd_split = [15, 25, 35, 45]
        for idx, val in enumerate(windspd_split):
            left = val
            if idx == len(windspd_split) - 1:
                right = 999
                interval_str = f'>{left}'
            else:
                right = windspd_split[idx + 1]
                interval_str = f'{left}-{right}'
            print(interval_str)
            print('=' * len(interval_str))
            print()

            for name, df in zip(self.src_plot_names, bias_list):
                print(name)
                print('-' * len(name))
                df_part = df.loc[(df['sfmr_windspd'] >= left)
                                 & (df['sfmr_windspd'] < right)]
                windspd_bias = df_part['windspd_bias']

                print(f'Count: {len(df_part)}')
                print(f'Max bias: {windspd_bias.max()}')
                print(f'Min bias: {windspd_bias.min()}')
                print(f'Mean bias: {windspd_bias.mean()}')
                print(f'Median bias: {windspd_bias.median()}')
                print((f"""Mean absolute bias: """
                       f"""{windspd_bias.abs().mean()}"""))

                truth = df_part['sfmr_windspd']
                observation = df_part['tgt_windspd']
                mse = mean_squared_error(truth, observation)
                print(f'RMSE: {math.sqrt(mse)}')
                print('\n\n')

    def show_simple_statistic(self, bias_list):
        for name, df in zip(self.src_plot_names, bias_list):
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

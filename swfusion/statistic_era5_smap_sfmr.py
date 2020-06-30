import logging
import math
import os
import string
import sys

from mpl_toolkits.mplot3d import Axes3D
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
import numpy as np

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
        # if len(sources) != 2:
        #     self.logger.error('Sources length should be 2')
        #     exit(1)
        self.sources = sorted(sources, key=src_name_priority.get)
        self.src_1 = self.sources[0]
        if len(sources) == 2:
            self.src_2 = self.sources[1]

        src_plot_names_mapper = {
            'smap_prediction': 'HYBRID',
            'smap': 'SMAP',
            'era5': 'ERA5',
        }
        self.src_plot_names = [src_plot_names_mapper[src]
                               for src in self.sources]

        # self.show_hist_of_matchup()
        # breakpoint()
        # if 'smap_prediction' in self.sources:
        #     self.add_dist2coast()
        self.compare_two_sources()
        # self.show_2d_diff_between_sources()


    def show_hist_of_matchup(self):
        # df = pd.read_sql('SELECT * FROM tc_sfmr_era5_na',
        #                  self.engine)
        # df.hist(column=['sfmr_windspd'], figsize = (12,10))
        # plt.show()

        all_basins = self.CONFIG['ibtracs']['urls'].keys()
        for basin in all_basins:
            table_name = f'tc_smap_era5_{basin}'
            if (utils.get_class_by_table_name(self.engine, table_name)
                    is not None):
                df = pd.read_sql(f'SELECT * FROM {table_name}',
                                 self.engine)
                df.hist(column=['smap_windspd'], figsize=(12, 10))
                plt.show()

    def show_2d_diff_between_sources(self):
        try:
            if ('smap_prediction' in self.sources
                    and 'smap' in self.sources):
                src_1 = 'smap'
                src_2 = 'smap_prediction'
            else:
                self.logger.error(('Have not considered this '
                                   'combination of sources'))
                sys.exit(1)

            CompareTable = utils.create_2d_compare_table(
                self, src_1, src_2)
            table_name = utils.get_2d_compare_table_name(
                self, src_1, src_2)
            # Get max and min of x and y
            df = pd.read_sql(f'SELECT * FROM {table_name}', self.engine)
            x_min = df['x'].min()
            x_max = df['x'].max()
            y_min = df['y'].min()
            y_max = df['y'].max()
            x_length = int(x_max - x_min) + 1
            y_length = int(y_max - y_min) + 1

            degree = u'\N{DEGREE SIGN}'
            xticklabels = [f'{int(0.25*(x+x_min))}{degree}'
                           if not x%4 else ''
                           for x in range(x_length)]
            yticklabels = [f'{int(0.25*(y+y_min))}{degree}'
                           if not y%4 else ''
                           for y in range(y_length)]

            count_matrix = np.zeros(shape=(y_length, x_length))
            mean_bias_matrix = np.zeros(shape=(y_length, x_length))
            rmse_matrix = np.zeros(shape=(y_length, x_length))

            for y in range(y_length):
                for x in range(x_length):
                    pixel_df = df.loc[(df['y'] == y_min + y)
                                      & (df['x'] == x_min + x)]
                    count_matrix[y][x] = len(pixel_df)
                    mean_bias_matrix[y][x] = pixel_df[
                        f'{src_2}_minus_{src_1}_windspd'].mean()
                    rmse_matrix[y][x] = np.sqrt(
                        mean_squared_error(pixel_df[f'{src_1}_windspd'],
                                           pixel_df[f'{src_2}_windspd']))

            fig, axes = plt.subplots(1, 3, figsize=(22, 4))

            sns.heatmap(count_matrix, ax=axes[0], square=True, fmt='g')
            sns.heatmap(mean_bias_matrix, ax=axes[1], square=True,
                        fmt='g')
            sns.heatmap(rmse_matrix, ax=axes[2], square=True, fmt='g')

            subplot_titles = ['Count', 'Mean bias (m/s)', 'RMSE (m/s)']
            for index, ax in enumerate(axes):
                ax.invert_yaxis()
                ax.text(-0.1, 1.025,
                        f'({string.ascii_lowercase[index]})',
                        transform=ax.transAxes, fontsize=16,
                        fontweight='bold', va='top', ha='right')
                ax.set_title(subplot_titles[index], size=15)

                ax.set_xticklabels(xticklabels)
                ax.set_yticklabels(yticklabels, rotation=0)

            fig.tight_layout()
            # plt.xlabel('x')
            # plt.ylabel('y')
            # ax.legend(title=None, loc='upper left', labels=labels)
            plt.show()
        except Exception as msg:
            breakpoint()
            sys.exit(msg)

    def add_dist2coast(self):
        lons = [round(x * 0.04 - 179.98, 2) for x in range(9000)]
        lats = [round(y * 0.04 - 89.98, 2) for y in range(4500)]

        dist2coast_table_name = 'dist2coast_na_sfmr'
        Dist2Coast = utils.get_class_by_tablename(
            self.engine, dist2coast_table_name)

        validation_tablename = utils.gen_validation_tablename(
            self, 'sfmr', 'smap_prediction')
        Validation = utils.get_class_by_tablename(
            self.engine, validation_tablename)

        validation_query = self.session.query(Validation).filter(
            Validation.sfmr_datetime > self.period[0],
            Validation.sfmr_datetime < self.period[1]
        )
        validation_count = validation_query.count()

        for validation_idx, validation_row in enumerate(validation_query):
            print(f'\r{validation_idx+1}/{validation_count}', end='')

        indices_to_drop = []
        for src in self.sources:
            length = len(bias[src])

            for i in range(length):
                print(f'\r{i+1}/{length}', end='')

                lookup_lon, lookup_lon_idx = \
                    utils.get_nearest_element_and_index(
                        lons, bias[src]['sfmr_lon'][i]-360)
                lookup_lat, lookup_lat_idx = \
                    utils.get_nearest_element_and_index(
                        lats, bias[src]['sfmr_lat'][i])
                dist_query = self.session.query(Dist2Coast).filter(
                    Dist2Coast.lon > lookup_lon - 0.01,
                    Dist2Coast.lon < lookup_lon + 0.01,
                    Dist2Coast.lat > lookup_lat - 0.01,
                    Dist2Coast.lat < lookup_lat + 0.01,
                )
                if dist_query.count() != 1:
                    self.logger.error('Dist not found')
                    breakpoint()
                    exit(1)

                if dist_query[0].dist2coast > distance_to_land_threshold:
                    indices_to_drop.append(i)

            utils.delete_last_lines()
            print('Done')

            bias[src].drop(indices_to_drop, inplace=True)

    def compare_two_sources(self):
        bias = dict()
        sfmr_dt = dict()
        sfmr_dt_name = 'sfmr_datetime'

        # center_border_range = 1
        # dis_minutes_threshold = 20
        # cbr = center_border_range
        # dmt = dis_minutes_threshold

        # not_center_clause = (f"""and not (x > -{cbr} and x < {cbr} """
        #                      f"""and y > -{cbr} and y < {cbr}) """)
        # tight_time_clause = (f"""and dis_minutes > -{dmt} """
        #                      f"""and dis_minutes < {dmt} """)
        not_center_clause = 'and not (x = 0 and y = 0) '
        tight_time_clause = ''
        # not_low_outilier_clause = 'and smap_prediction_windspd > 1 '
        not_low_outilier_clause = ''
        dist2coast_threshold = 0
        far_away_from_coast_clause = (f""" and dist2coast > """
                                      f"""{dist2coast_threshold} """)

        for src in self.sources:
            table_name = utils.gen_validation_tablename(self, 'sfmr',
                                                        src)
            table_name += '_invalid'
            if src == 'smap_prediction' and 'smap' in self.sources:
                table_name = ('smap_prediction_validation_by_sfmr'
                             '_na_aligned_with_smap')
            df = pd.read_sql(
                (f"""SELECT * FROM {table_name} """
                 f"""where sfmr_datetime > "{self.period_str[0]}" """
                 f"""and sfmr_datetime < "{self.period_str[1]}" """
                 f"""{not_center_clause} {tight_time_clause} """
                 f"""{not_low_outilier_clause} """
                 f"""{far_away_from_coast_clause} """),
                self.engine)

            bias[src] = df
            sfmr_dt[src] = bias[src][sfmr_dt_name]

        if len(self.sources) > 1:
            sfmr_dt_intersection = set(sfmr_dt[self.src_1]) & \
                set(sfmr_dt[self.src_2])
            tmp = set()
            for ts in sfmr_dt_intersection:
                tmp.add(ts.to_datetime64())
            sfmr_dt_intersection = tmp
            print((f"""Length of sfmr_dt_intersection: """
                   f"""{len(sfmr_dt_intersection)}"""))

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
            print(f'Length of {self.src_1}: {len_src1}')
            print(f'Length of {self.src_2}: {len_src2}')
        else:
            bias[src].drop_duplicates(subset='sfmr_datetime',
                                      inplace=True, ignore_index=True)



        bias_list = [bias[src] for src in self.sources]
        bias_list = self.unify_df_colnames(bias_list)

        tmp = bias['smap_prediction']
        threshold = 15
        validation = tmp[tmp['sfmr_windspd'] > threshold]
        # validation = tmp.loc[(tmp['dis_minutes'] > -threshold)
        #                      & (tmp['dis_minutes'] < threshold)]
        self.dig_into_validation(validation)

        self.export_statistic_csv(bias_list, 'rmse')
        self.export_statistic_csv(bias_list, 'mae')
        self.export_statistic_csv(bias_list, 'mean_bias')
        # self.show_simple_statistic(bias_list)
        # self.show_detailed_statistic_splitted_by_windspd(bias_list)
        # self.show_detailed_statistic_splitted_by_datetime(bias_list)

        # Pre-process bias list
        # sfmr_windspd = np.zeros(shape=(bias[self.sources[0]].shape[0],))
        sfmr_windspd = bias_list[0]['sfmr_windspd'].to_numpy()
        pred_windspd = dict()
        for idx, src in enumerate(self.sources):
            name = self.src_plot_names[idx]
            pred_windspd[name] = bias_list[idx][
                'tgt_windspd'].to_numpy()
        out_dir = self.CONFIG['result']['dirs'][
            'fig']['validation_by_sfmr']
        os.makedirs(out_dir, exist_ok=True)
        utils.scatter_plot_pred(out_dir, sfmr_windspd, pred_windspd,
                                statistic=True,
                                x_label='resampled SFMR wind speed (m/s)',
                                y_label='wind speed (m/s)',
                                palette_start=2,
                                range_min=15)
        # self.plot_scatter_regression(bias_list)
        # val_dir = self.CONFIG['results']['dirs']['fig'][
        #     'validation_by_sfmr']
        # utils.box_plot_windspd(val_dir

    def dig_into_validation(self, df):
        """
        df['dis2coast_range'] = ''
        sorted_order = []
        split_values = [0, 50, 100, 200, 400, 800, 1400, 9999]
        for idx, val in enumerate(split_values):
            if idx == len(split_values) - 1:
                break
            left = val
            right = split_values[idx + 1]
            indices = df.loc[(df['dist2coast'] >= left)
                             & (df['dist2coast'] < right)].index
            if idx + 1 < len(split_values) - 1:
                label = f'{left} - {right}'
            else:
                label = f'> {left}'
            df.loc[indices, ['dis2coast_range']] = label
            sorted_order.append(label)

        sns.violinplot(x="dis2coast_range", y="windspd_bias",
                        data=df, order=sorted_order,
                        inner='box')
        plt.show()
        breakpoint()
        """
        fig_dir = self.CONFIG['result']['dirs']['fig'][
            'validation_by_sfmr']
        utils.grid_rmse_and_bias(True,
                                 fig_dir,
                                 df['x'],
                                 df['y'],
                                 df['tgt_windspd'],
                                 df['sfmr_windspd'])

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
        breakpoint()
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
            # r2[src] = r2_score(df_part['tgt_windspd'],
            #                    linefitline(src, df_part['tgt_windspd']))
            equation_str[src] = (f"""y={slope[src]:.2f}x"""
                                 f"""{interpect[src]:+.2f}""")


        pal = dict()
        labels = []
        colors = ['blue', 'red']
        for idx, src in enumerate(self.sources):
            pal[src] = colors[idx]
            labels.append((f"""{self.src_plot_names[idx]}: """
                           f"""{equation_str[src]} """
                           f"""(Corr Coeff: {r_value[src]:.3f})"""))

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

    def show_detailed_statistic_splitted_by_windspd(self, bias_list):
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
                rmse = math.sqrt(mse)
                print(f'RMSE: {rmse}')
                nrmse = rmse / truth.mean()
                print(f'Normalized-RMSE: {nrmse}')
                std_dev = np.std(observation)
                print(f'Standard deviation: {std_dev}')
                print('\n\n')

    def export_statistic_csv(self, bias_list, metric):
        try:
            if len(bias_list) != 1:
                self.logger.error('Only support bias from one sources')
                exit(1)

            df = bias_list[0]
            year_split = [2015 + x for x in range(5)]
            windspd_split = [15, 25, 35, 45]

            csv_rows = []
            csv_index = []

            for y_i, y_v in enumerate(windspd_split):
                left = y_v
                if y_i < len(windspd_split) - 1:
                    right = windspd_split[y_i + 1]
                    interval_str = f'{left}-{right}'
                else:
                    right = 999
                    interval_str = f'>{left}'

                csv_index.append(interval_str)

                df_windspd = df.loc[
                    (df['sfmr_windspd'] >= left)
                    & (df['sfmr_windspd'] < right)]

                one_csv_row = dict()
                for x_i, x_v in enumerate(year_split):
                    df_windspd_year = df_windspd.loc[
                        (df_windspd['sfmr_datetime']
                         >= f'{x_v}-01-01 00:00:00')
                        & (df_windspd['sfmr_datetime']
                           <= f'{x_v}-12-31 23:59:59')
                    ]
                    tmp_df = df_windspd_year
                    windspd_bias = tmp_df['windspd_bias']

                    one_csv_row[f'{x_v}_count'] = len(tmp_df)

                    if metric == 'rmse':
                        truth = tmp_df['sfmr_windspd']
                        pred = tmp_df['tgt_windspd']
                        mse = mean_squared_error(truth, pred)
                        rmse = math.sqrt(mse)
                        one_csv_row[f'{x_v}_rmse'] = rmse

                    elif metric == 'mae':
                        mae = windspd_bias.abs().mean()
                        one_csv_row[f'{x_v}_mae'] = mae

                    elif metric == 'mean_bias':
                        mean_bias = windspd_bias.mean()
                        one_csv_row[f'{x_v}_mean_bias'] = mean_bias

                    else:
                        self.logger.error('Invalid metric')
                        exit(1)


                csv_rows.append(one_csv_row)

            csv_df = pd.DataFrame(data=csv_rows, index=csv_index)

            out_dir = self.CONFIG['result']['dirs']['fig'][
                'validation_by_sfmr']
            name = f'sta_{metric}_{year_split[0]}_{year_split[-1]}.csv'

            csv_df.to_csv(f'{out_dir}{name}')

        except Exception as msg:
            breakpoint()
            exit(msg)


    def show_detailed_statistic_splitted_by_datetime(self, bias_list):
        year_split = [2015 + x for x in range(5)]
        for idx, val in enumerate(year_split):
            print(val)
            print('=' * 3*len(str(val)))
            print()

            for name, df in zip(self.src_plot_names, bias_list):
                print(name)
                print('-' * len(name))
                df_part = df.loc[
                    (df['sfmr_datetime'] >= f'{val}-01-01 00:00:00')
                    & (df['sfmr_datetime'] <= f'{val}-12-31 23:59:59')]
                windspd_bias = df_part['windspd_bias']

                print(f'Count: {len(df_part)}')
                if not len(df_part):
                    print('\n')
                    continue

                print(f'Max bias: {windspd_bias.max()}')
                print(f'Min bias: {windspd_bias.min()}')
                print(f'Mean bias: {windspd_bias.mean()}')
                print(f'Median bias: {windspd_bias.median()}')
                print((f"""Mean absolute bias: """
                       f"""{windspd_bias.abs().mean()}"""))

                truth = df_part['sfmr_windspd']
                observation = df_part['tgt_windspd']
                mse = mean_squared_error(truth, observation)
                rmse = math.sqrt(mse)
                print(f'RMSE: {rmse}')
                nrmse = rmse / truth.mean()
                print(f'Normalized-RMSE: {nrmse}')
                std_dev = np.std(observation)
                print(f'Standard deviation: {std_dev}')
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

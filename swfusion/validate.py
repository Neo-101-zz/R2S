import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.axes_grid1 import make_axes_locatable

import utils

class ValidationManager(object):

    def __init__(self, CONFIG, period, basin):
        self.logger = logging.getLogger(__name__)
        self.CONFIG = CONFIG
        self.period = period
        self.basin = basin

        self.bias_root_dir = self.CONFIG['result']['dirs']['statistic'][
            'windspd_bias_to_sfmr']
        self.find_bias_dir_and_file_path()
        self.bias_df = pd.read_pickle(self.bias_file_path)

        # self.draw_histogram()
        self.draw_2d_scatter()
        # self.draw_3d_scatter()

    def find_bias_dir_and_file_path(self):
        save_name = (f"""{self.basin}_"""
                     f"""{self.period[0].strftime('%Y%m%d%H%M%S')}_"""
                     f"""{self.period[1].strftime('%Y%m%d%H%M%S')}""")
        file_path = f'{self.bias_root_dir}{save_name}/{save_name}.pkl'

        if os.path.exists(file_path):
            self.bias_dir = f'{self.bias_root_dir}{save_name}/'
            self.bias_file_path = file_path
        else:
            self.logger.error(f"""No such file: {file_path}""")
            exit(1)

    def draw_histogram(self):
        cols_name = ['dis_minutes', 'dis_kms', 'windspd_bias']

        for name in cols_name:
            self.bias_df.hist(column=name)
            fig_name = f'histogram_{name}.png'
            plt.savefig(f'{self.bias_dir}{fig_name}')

    def draw_single_2d_scatter_count_heatmap(self, fig, ax, x_col_name,
                                             y_col_name):
        # Get the integer bound of x and y
        interval = self.CONFIG['plot']['scatter_count']['interval']
        x_col = self.bias_df[x_col_name]
        y_col = self.bias_df[y_col_name]

        xlims = utils.get_bound_of_multiple_int(
            (x_col.min(), x_col.max()), interval)
        ylims = utils.get_bound_of_multiple_int(
            (y_col.min(), y_col.max()), interval)

        xticks = [xlims[0] + interval * i for i in range(
            int((xlims[1] - xlims[0]) / interval) + 1)]
        yticks = [ylims[0] + interval * i for i in range(
            int((ylims[1] - ylims[0]) / interval) + 1)]

        x_intervals = len(xticks) - 1
        y_intervals = len(yticks) - 1
        count = np.zeros(shape=(y_intervals, x_intervals), dtype=int)

        for i in range(len(self.bias_df)):
            row_y = self.bias_df[y_col_name][i]
            row_x = self.bias_df[x_col_name][i]

        x_intervals = len(xticks) - 1
        y_intervals = len(yticks) - 1
        count = np.zeros(shape=(y_intervals, x_intervals), dtype=int)

        for i in range(len(self.bias_df)):
            row_y = self.bias_df[y_col_name][i]
            row_x = self.bias_df[x_col_name][i]

            row_y_idx = None
            row_x_idx = None
            # Search interval of y
            for y_idx, y_val in enumerate(yticks):
                if y_idx < y_intervals:
                    if y_val <= row_y and row_y < yticks[y_idx + 1]:
                        row_y_idx = y_idx
                        break

            # Search interval of x
            for x_idx, x_val in enumerate(xticks):
                if x_idx < x_intervals:
                    if x_val <= row_x and row_x < xticks[x_idx + 1]:
                        row_x_idx = x_idx
                        break

            count[row_y_idx][row_x_idx] += 1

        # Make the y axis of count array large-end on the top
        count = np.flip(count, 0)

        left = min(xticks)
        right = max(xticks)
        bottom = min(yticks)
        top = max(yticks)
        count_heatmap = ax.imshow(count, cmap='Reds',
                                  extent=[left, right, bottom, top])
        ax.grid()
        # fig.colorbar(count_heatmap, ax=ax)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        axis_label_mapper = self.CONFIG['plot']['axis_label_mapper']
        xlabel = axis_label_mapper[x_col_name]
        ylabel = axis_label_mapper[y_col_name]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig.colorbar(count_heatmap, cax=cax,
                           orientation='vertical', format='%d')

    def draw_single_2d_scatter_plot(self, ax, x_col_name, y_col_name):
        fig_name = f'2d_scatter_plot_{x_col_name}_{y_col_name}.png'
        ax.scatter(self.bias_df[x_col_name], self.bias_df[y_col_name])
        ax.set_aspect('equal', 'box')
        ax.grid()

        axis_label_mapper = self.CONFIG['plot']['axis_label_mapper']
        xlabel = axis_label_mapper[x_col_name]
        ylabel = axis_label_mapper[y_col_name]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def draw_2d_scatter_pair(self, x_col_name, y_col_name):
        fig, axs = plt.subplots(1, 2, figsize=(24, 12))
        self.draw_single_2d_scatter_plot(axs[0], x_col_name, y_col_name)
        self.draw_single_2d_scatter_count_heatmap(fig, axs[1], x_col_name,
                                                  y_col_name)
        fig_name = f'2d_scatter_{x_col_name}_{y_col_name}.png'
        plt.savefig(f'{self.bias_dir}{fig_name}')
        plt.clf()

    def draw_2d_scatter(self):
        x_y_col_name_pairs = [
            ('dis_minutes', 'windspd_bias'),
            ('dis_kms', 'windspd_bias'),
            ('sfmr_windspd', 'windspd_bias')
        ]
        for name_pair in x_y_col_name_pairs:
            self.draw_2d_scatter_pair(name_pair[0], name_pair[1])


    def draw_3d_scatter(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = self.bias_df['dis_minutes']
        ys = self.bias_df['dis_kms']
        zs = self.bias_df['windspd_bias']
        sc = ax.scatter(xs, ys, zs, c=zs, cmap='tab20')
        plt.colorbar(sc)

        ax.set_xlabel('dis_minutes')
        ax.set_ylabel('dis_kms')
        ax.set_zlabel('windspd_bias')

        plt.show()

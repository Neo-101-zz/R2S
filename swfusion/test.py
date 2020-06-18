import pickle
import os
import pandas as pd

import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import smogn
import numpy as np

ori_path = ('/Users/lujingze/Programming/SWFusion/'
            'regression/tc/dataset/original/')


def split_validset_from_trainset():
    train_path = f'{ori_path}train.pkl'
    with open(train_path, 'rb') as f:
        train = pickle.load(f)

    y_full = train['smap_windspd']
    indices_to_delete = []
    bins = np.linspace(0, y_full.max(), int(y_full.max() / 5))
    y_binned = np.digitize(y_full, bins)

    unique, counts = np.unique(y_binned, return_counts=True)
    for idx, val in enumerate(counts):
        if val < 2:
            indices_to_delete.append(idx)
    bins = np.delete(bins, indices_to_delete)
    y_binned = np.digitize(y_full, bins)

    train_splitted, valid = train_test_split(train, test_size=0.2,
                                             stratify=y_binned)
    train_splitted.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)

    train_splitted.to_pickle(f'{ori_path}train_splitted.pkl')
    valid.to_pickle(f'{ori_path}valid.pkl')

# split_validset_from_trainset()

def remove_index_col_from_original_dataset():
    zero_paths = [f'{ori_path}train.pkl',
                  f'{ori_path}test.pkl']
    for idx, path in enumerate(zero_paths):
        with open(path, 'rb') as f:
            df = pickle.load(f)
        df.drop(columns='index', inplace=True)
        df.to_pickle(path)


def remove_zero_windspd_from_original_dataset():
    zero_paths = [f'{ori_path}train_with_zero_windspd.pkl',
                  f'{ori_path}test_with_zero_windspd.pkl']
    nonzero_fnames = ['train.pkl', 'test.pkl']
    for path, fname in zip(zero_paths, nonzero_fnames):
        with open(path, 'rb') as f:
            df = pickle.load(f)
        indices = df.index[df['smap_windspd'] == 0].tolist()
        new_df = df.drop(index=indices)
        new_df.reset_index(inplace=True, drop=True)
        breakpoint()
        new_df.to_pickle(f'{ori_path}{fname}')

# remove_index_col_from_original_dataset()

dataset_paths = {
    'ori': {
        'train': ('/Users/lujingze/Programming/SWFusion/regression/'
                  'tc/dataset/original/train.pkl'),
        'test': ('/Users/lujingze/Programming/SWFusion/regression/'
                 'tc/dataset/original/test.pkl'),
        'train_splitted': ('/Users/lujingze/Programming/SWFusion/regression/'
                  'tc/dataset/original/train_splitted.pkl'),
        'valid': ('/Users/lujingze/Programming/SWFusion/regression/'
                 'tc/dataset/original/valid.pkl'),
    },
    'smogn_final_on_all': {
        'train': ('/Users/lujingze/Programming/SWFusion/regression/'
                  'tc/dataset/smogn_final/smogn_on_all_data/'
                  'train_smogn_on_all_data.pkl'),
        'test': ('/Users/lujingze/Programming/SWFusion/regression/'
                 'tc/dataset/smogn_final/smogn_on_all_data/'
                 'test_smogn_on_all_data.pkl'),
    },
    'smogn_final_on_train': {
        'train': ('/Users/lujingze/Programming/SWFusion/regression/'
                  'tc/dataset/smogn_final/smogn_on_train/'
                  'train_smogn_on_train.pkl'),
        'test': ('/Users/lujingze/Programming/SWFusion/regression/'
                 'tc/dataset/original/test.pkl'),
    },
    'smogn_final_on_train_splitted': {
        'train': ('/Users/lujingze/Programming/SWFusion/regression/'
                  'tc/dataset/smogn_final/smogn_on_train_splitted/'
                  'train_splitted_smogn.pkl'),
        'test': ('/Users/lujingze/Programming/SWFusion/regression/'
                 'tc/dataset/original/test.pkl'),
    },
}

dataset = dict()
for idx_1, (key_1, val_1) in enumerate(dataset_paths.items()):
    dataset[key_1] = dict()
    for idx_2, (key_2, val_2) in enumerate(val_1.items()):
        try:
            with open(val_2, 'rb') as f:
                dataset[key_1][key_2] = pickle.load(f)
        except Exception as msg:
            breakpoint()

def plot_dist():
    with open(('/Users/lujingze/Programming/SWFusion/regression/tc/'
               'dataset/smogn_hyperopt/'
               'k_7_pert_0.02_samp_extreme_0.9_auto_high_3.5/train_smogn.pkl'),
              'rb') as f:
        auto_train_smogn = pickle.load(f)

    sns.set_style("whitegrid")
    # plot y distribution
    # sns.kdeplot(dataset['ori']['train']['smap_windspd'], clip=(0, 99),
    #             label='Original train')
    # sns.kdeplot(dataset['ori']['test']['smap_windspd'], clip=(0, 99),
    #             label='Original test')
    sns.kdeplot(dataset['ori']['train_splitted']['smap_windspd'], clip=(0, 99),
                label='Original train_splitted')
    sns.kdeplot(dataset['ori']['valid']['smap_windspd'], clip=(0, 99),
                label='Original valid')
    sns.kdeplot(dataset['smogn_final_on_train_splitted']['train'][
        'smap_windspd'], clip=(0, 99), label='SMOGN train_splitted')
    # sns.kdeplot(dataset['smogn_final_on_all']['train']['smap_windspd'],
    #             clip=(0, 99), label='SMOGN_on_all train')
    # sns.kdeplot(dataset['smogn_final_on_all']['test']['smap_windspd'],
    #             clip=(0, 99), label='SMOGN_on_all test')
    # sns.kdeplot(dataset['smogn_final_on_train']['train']['smap_windspd'],
    #             clip=(0, 99), label='SMOGN_on_train train')
    # sns.kdeplot(auto_train_smogn['smap_windspd'],
    #             clip=(0, 99), label='auto_SMOGN train')
    # sns.kdeplot(dataset['smogn_final_on_train']['test']['smap_windspd'],
    #             clip=(25, 99), label='SMOGN_on_train test')
    # add labels of x and y axis
    plt.xlabel('SMAP wind speed (m/s)')
    plt.ylabel('Probability')
    # plt.savefig((f"""{the_class.smogn_setting_dir}"""
    #              f"""dist_of_trainset_comparison.png"""))
    plt.show()

def smogn_on_train_splitted():
    smogn_params = dict(
        # main arguments
        data=dataset['ori']['train_splitted'],
        y='smap_windspd',        # string ('header name')
        k=7,                     # positive integer (k < n)
        pert=0.02,               # real number (0 < R < 1)
        samp_method='extreme',   # string ('balance' or 'extreme')
        drop_na_col=True,        # boolean (True or False)
        drop_na_row=True,        # boolean (True or False)
        replace=False,           # boolean (True or False)

        # phi relevance arguments
        rel_thres=0.9,          # real number (0 < R < 1)
        rel_method='manual',     # string ('auto' or 'manual')
        # rel_xtrm_type='high',  # unused (rel_method='manual')
        # rel_coef=3.525,         # unused (rel_method='manual')
        rel_ctrl_pts_rg = [
            [5, 0, 0],
            [20, 0, 0],
            [35, 0, 0],
            [50, 1, 0]
        ]
    )

    save_dir = ('/Users/lujingze/Programming/SWFusion/regression/'
                'tc/dataset/smogn_final/smogn_on_train_splitted/')
    train_splitted_smogn = smogn.smoter(**smogn_params)
    os.makedirs(save_dir, exist_ok=True)
    train_splitted_smogn.to_pickle(f'{save_dir}train_splitted_smogn.pkl')

# smogn_on_train_splitted()
plot_dist()

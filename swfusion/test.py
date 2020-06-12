import pickle
import os

import smogn

test_path = ('/Users/lujingze/Programming/SWFusion/regression/'
             'tc/dataset/original/test.pkl')
with open(test_path, 'rb') as f:
    test = pickle.load(f)

# specify phi relevance values
rg_mtrx = [
    [5, 0, 0],  # over-sample ("minority")
    [20, 0, 0],  # under-sample ("majority")
    [35, 0, 0],  # under-sample
    [50, 1, 0],  # under-sample
]

smogn_params = dict(
    # main arguments
    data=test,                 # pandas dataframe
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
    # rel_xtrm_type='both',  # unused (rel_method='manual')
    # rel_coef=1.50,         # unused (rel_method='manual')
    rel_ctrl_pts_rg=rg_mtrx  # 2d array (format: [x, y])
)

test_smogn = smogn.smoter(**smogn_params)
out_path = ('/Users/lujingze/Programming/SWFusion/regression/tc/'
            'dataset/comparison_smogn/test_smogn.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(test_smogn, f)

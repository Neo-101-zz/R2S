import pickle
import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel


model_dirs = {
    'mse': ('/Users/lujingze/Programming/SWFusion/regression/'
            'tc/lightgbm/model/na_valid_2.496193/'),
    'smogn_tcl': ('/Users/lujingze/Programming/SWFusion/regression/tc/'
                  'lightgbm/model/na_valid_2557.909583_fl_smogn_final_'
                  'thre_50_power_3_under_maxeval_100/'),
    'fl': ('/Users/lujingze/Programming/SWFusion/classify/tc/lightgbm/'
           'model/na_valid_0.560000_45_fl_smogn_final_unb_maxeval_2/'),
    }
models = dict()

for idx, (name, dir_path) in enumerate(model_dirs.items()):
    model_name = [f for f in os.listdir(dir_path)
                  if f.endswith('.pkl')
                  and f.startswith('na')]
    if len(model_name) != 1:
        exit(1)

    with open(f'{dir_path}{model_name[0]}', 'rb') as f:
        models[name] = pickle.load(f).model

with open('/Users/lujingze/Programming/SWFusion/regression/tc/dataset/original/train_splitted.pkl', 'rb') as f:
    train = pickle.load(f)
with open('/Users/lujingze/Programming/SWFusion/regression/tc/dataset/original/test.pkl', 'rb') as f:
    test = pickle.load(f)

y_name = 'smap_windspd'
y_train = getattr(train, y_name).reset_index(drop=True)
y_test = getattr(test, y_name).reset_index(drop=True)
X_train = test.drop([y_name], axis=1).reset_index(drop=True)
X_test = test.drop([y_name], axis=1).reset_index(drop=True)

interpreter = Interpretation(X_test, feature_names=X_test.columns)

for idx, (name, lgb) in enumerate(models.items()):
    model = InMemoryModel(lgb.predict, examples=X_test)
    print("Number of classes: {}".format(model.n_classes))
    print("Input shape: {}".format(model.input_shape))
    print("Model Type: {}".format(model.model_type))
    print("Output Shape: {}".format(model.output_shape))
    print("Output Type: {}".format(model.output_type))
    print("Returns Probabilities: {}".format(model.probability))
    fea_importances = interpreter.feature_importance.feature_importance(model)
    fea_importances.sort_values(ascending=False, inplace=True)

    top_fea_importance = fea_importances[:5]
    top_fea_importance.to_csv(f'{model_dirs[name]}top_5_features.csv')

    print(f'\n\n-----{name}-----')
    print(fea_importances[:5])

    # interpreter.feature_importance.plot_feature_importance(model, ascending=False)
    # plt.show()
    # 
    # fea_importances.sort_values(ascending=False, inplace=True)
    # pdp_features = ['mean_square_slope_of_waves', '10_metre_wind_speed']
    # r = interpreter.partial_dependence.plot_partial_dependence(
    #     pdp_features, model, grid_resolution=30, n_jobs=1)
breakpoint()

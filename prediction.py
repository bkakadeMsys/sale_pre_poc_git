import pandas.api.types
import os
import shutil
import joblib
import warnings
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import ElasticNetCV, ElasticNet, GammaRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import BayesianRidge
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
import plotly.graph_objects as go
from sklearn import preprocessing
from lazypredict.Supervised import LazyRegressor
import lazypredict
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from flask import jsonify


def training(model_path, dataset, target):
    if pandas.api.types.is_numeric_dtype(dataset[target]):
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset['month'] = [i.month for i in dataset['date']]
        dataset['year'] = [i.year for i in dataset['date']]
        dataset['day_of_week'] = [i.dayofweek for i in dataset['date']]
        dataset['day_of_year'] = [i.dayofyear for i in dataset['date']]
        dataset['day'] = [i.day for i in dataset['date']]
        le = preprocessing.LabelEncoder()
        dataset['category'] = le.fit_transform(dataset['category'])
        joblib.dump(le, model_path + 'product_encoder.joblib', compress=9)
        # load it when test
        X = dataset.drop([target, 'date'], axis=1)
        y = dataset[target]
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
        d = {'BaggingRegressor': BaggingRegressor(), 'ElasticNetCV': ElasticNetCV(), 'ElasticNet': ElasticNet(),
             'ExtraTreeRegressor': ExtraTreeRegressor(), 'GammaRegressor': GammaRegressor(),
             'ExtraTreesRegressor': ExtraTreesRegressor(), 'DecisionTreeRegressor': DecisionTreeRegressor(),
             'DummyRegressor': DummyRegressor(), 'AdaBoostRegressor': AdaBoostRegressor(),
             'BayesianRidge': BayesianRidge()}
        models, _ = reg.fit(train_x, test_x, train_y, test_y)
        model_names = []
        mae_list = []
        r2_list = []
        adj_r2_list = []
        n = len(test_x)
        p = len(test_x.columns)
        for i in range(0, 3):
            model = d[models.index[i]]
            model.fit(train_x, train_y)
            predictions = model.predict(test_x)
            r2_value = r2_score(test_y, predictions)
            mae_value = mean_absolute_error(test_y, predictions)
            adj_r2 = 1 - (1 - r2_value) * (n - 1) / (n - p - 1)
            model_names.append(models.index[i])
            mae_list.append(mae_value)
            r2_list.append(r2_value)
            adj_r2_list.append(adj_r2)
            joblib.dump(model, model_path + 'models/' + models.index[i] + ".pkl")
        modelDetails = {'Models': model_names, 'MAE': mae_list,
                        'r2 score': r2_list, 'Adjusted r2 score': adj_r2_list}
    else:
        return jsonify('Target Feature name should be Numeric')
    return modelDetails

def testing(model_path, data, data_test, model_name):
        data['date'] = pd.to_datetime(data['date'])
        data['month'] = [i.month for i in data['date']]
        data['year'] = [i.year for i in data['date']]
        data['day_of_week'] = [i.dayofweek for i in data['date']]
        data['day_of_year'] = [i.dayofyear for i in data['date']]
        data['day'] = [i.day for i in data['date']]
        le = joblib.load(model_path + 'product_encoder.joblib')
        data['category'] = le.transform(data['category'])
        X = data.drop(["date"], axis=1)
        path = os.listdir(model_path + 'models/')
        lis = []
        for i in range(len(path)):
            lis.append(path[i][:-4])
        available_models = {"Models": lis}
        if model_name in available_models['Models']:
            model = joblib.load(model_path + 'models/' + model_name + '.pkl')
        else:
            return jsonify("Model not found!")
        data_test['sales'] = model.predict(X)
        data_test['sales'] = data_test['sales'].apply(np.ceil)
        data_test['sales'] = data_test['sales'].astype(int)
        if os.path.isfile(model_path + "Output.csv"):
            os.remove(model_path + "Output.csv")
        data_test.to_csv(model_path + "Output.csv", index=False)
        return data_test.to_dict()



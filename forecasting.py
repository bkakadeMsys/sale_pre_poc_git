import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
import datetime
from tsprial.forecasting import *
import pickle


def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0


def train(data_file, if_train, model_path):
    sales = pd.read_csv(data_file)

    sales['date'] = pd.to_datetime(sales.date, format='%Y-%m-%d')
    sales = sales.set_index('date')
    sales['month'] = sales.index.month
    sales['year'] = sales.index.year
    sales['day'] = sales.index.day
    
    cat_1 = sales[sales.category == 'Beauty & Personal Care'].item.unique()
    cat_2 = sales[sales.category == 'Grocery & Gourmet Food'].item.unique()
    cat_3 = sales[sales.category == 'Clothing, Shoes and Jewelry'].item.unique()
    
    print(len(sales))
    sales = sales.drop(['category'], 1)
    sales = sales.drop(['store'], 1)

    if if_train == 'True' or if_train == 'true' or if_train == 'T':
        print("MODEL IS TRAINING.....................................")
        items_list = []
        for i in range(1, 51):
            item_train = sales[sales.item == i]
            items_list.append(item_train)

            X_train = item_train.drop(columns='sales')
            y_train = item_train['sales']

            model = ForecastingChain(
                Ridge(),
                n_estimators=24 * 3,
                lags=range(1, 24 * 7 + 1),
                use_exog=True,
                accept_nan=False
            )
            model.fit(X_train, y_train)

            filename = model_path + '/model_item_' + str(i) + '.pkl'
            pickle.dump(model, open(filename, 'wb'))
    last_date = sales.index[-1] + datetime.timedelta(days=1)
    return last_date, cat_1, cat_2, cat_3


def test(last_date, type_of_data, number_of, pass_cat, model_path):

    if type_of_data == 'W':
        totalday = number_of * 7
    if type_of_data == 'M':
        totalday = number_of * 30

    test_date = datetime.datetime.strptime(last_date.strftime('%Y-%m-%d'), '%Y-%m-%d')

    date_generated = pd.date_range(test_date, periods=totalday)

    final_test_data = pd.DataFrame()

    for i in pass_cat:
        item_test = pd.DataFrame()
        item_test['date'] = date_generated
        item_test = item_test.set_index('date')
        item_test['item'] = i
        item_test['month'] = item_test.index.month
        item_test['year'] = item_test.index.year
        item_test['day'] = item_test.index.day

        filename = model_path + '/model_item_' + str(i) + '.pkl'

        final_test_data['day'] = item_test.day
        final_test_data['month'] = item_test.month
        final_test_data['year'] = item_test.year

        loaded_model = pickle.load(open(filename, 'rb'))

        final_test_data['item' + str(i)] = np.round_(loaded_model.predict(item_test))

    final_test_data = final_test_data.drop(columns=['day', 'month', 'year'])
    final_test_data = final_test_data.resample(type_of_data).sum()
    print(final_test_data.head())

    final_test_data.index = final_test_data.index.strftime('%Y-%m-%d')
    return final_test_data.to_dict('index')

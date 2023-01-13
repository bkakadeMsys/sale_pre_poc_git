from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
import os
import shutil
import joblib
import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import lazypredict
import plotly.express as px
import datetime

from forecasting import train, test
from prediction import training, testing

warnings.warn('ignore', category=FutureWarning)
np.random.seed(1)

app = Flask(__name__)
cors = CORS(app)
ALLOWED_EXTENSIONS = (['csv'])
app.secret_key = "abcdef"
lazypredict.Supervised.REGRESSORS = lazypredict.Supervised.REGRESSORS[:10]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/project_list', methods=["GET", "POST"])
def project_list():
    if request.method == 'GET':
        UPLOAD_FOLDER_1 = 'projects/'
        if not os.path.isdir(UPLOAD_FOLDER_1):
            os.mkdir(UPLOAD_FOLDER_1)
        app.config['UPLOAD_FOLDER_1'] = UPLOAD_FOLDER_1

        path = os.listdir('projects/')
        lis = []
        for i in range(len(path)):
            lis.append(path[i])
        available_projects = {"Projects": lis}
        return available_projects


@app.route('/load_project', methods=["GET", "POST"])
def select_project():
    if request.method == 'POST':
        project_name = request.form['project_name']
        session['project_name'] = project_name
        return jsonify("Project Loaded Successfully!")


@app.route('/create_project', methods=["GET", "POST"])
def create_project():
    if request.method == 'POST':
        project_name = request.form['project_name']
        session['project_name'] = project_name
        UPLOAD_FOLDER_1 = 'projects/' + project_name + '/'
        if not os.path.isdir('projects/'):
            os.mkdir('projects/')

        if os.path.isdir(UPLOAD_FOLDER_1):
            return jsonify("Project " + project_name + " already exists")
        if not os.path.isdir(UPLOAD_FOLDER_1):
            os.mkdir(UPLOAD_FOLDER_1)
        app.config['UPLOAD_FOLDER_1'] = UPLOAD_FOLDER_1
        return jsonify("Project " + project_name + " Created Successfully")


@app.route('/load_data', methods=["GET", "POST"])
def data_load():
    if request.method == 'POST':
        session['project_name'] = 'project1'
        UPLOAD_FOLDER_1 = 'projects/' + session['project_name'] + '/' + 'data/'
        UPLOAD_FOLDER_2 = 'projects/' + session['project_name'] + '/' + 'models/'
        if os.path.isdir(UPLOAD_FOLDER_1):
            shutil.rmtree('projects/' + session['project_name'] + '/' + 'data')
        if os.path.isdir(UPLOAD_FOLDER_2):
            shutil.rmtree('projects/' + session['project_name'] + '/' + 'models')
        if not os.path.isdir(UPLOAD_FOLDER_1):
            os.mkdir(UPLOAD_FOLDER_1)
        if not os.path.isdir(UPLOAD_FOLDER_2):
            os.mkdir(UPLOAD_FOLDER_2)
        app.config['UPLOAD_FOLDER_1'] = UPLOAD_FOLDER_1
        if 'file' not in request.files:
            return jsonify('No file part')
        file = request.files['file']
        if file.filename == '':
            return jsonify('No selected file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER_1'], file.filename)
            file.save(filepath)
        else:
            return jsonify('Unsupported File Format')
        df = pd.read_csv(filepath)
        session['filepath'] = filepath
        cols = list(df.columns)
        session['columns'] = cols
        cols_response = {'columns': session['columns']}
        return cols_response


@app.route('/train', methods=["GET", "POST"])
def model_training():
    if request.method == 'POST':
        session['project_name'] = 'project1'
        file = os.listdir('projects/' + session['project_name'] + '/' + 'data/')
        dataset = pd.read_csv('projects/' + session['project_name'] + '/' + 'data/' + file[0])
        target = request.form['target']

        model_path = 'projects/' + session['project_name'] + '/'
        modelDetails = training(model_path, dataset, target)

        return modelDetails


@app.route('/model_list', methods=["GET", "POST"])
def model_list():
    path = os.listdir('projects/' + session['project_name'] + '/' + 'models/')
    lis = []
    for i in range(len(path)):
        lis.append(path[i][:-4])
    available_models = {"Models": lis}
    return available_models


@app.route('/test', methods=['GET', 'POST'])
def model_test():
    if request.method == 'POST':
        session['project_name'] = 'project1'
        UPLOAD_FOLDER = 'projects/' + session['project_name'] + '/' + 'test_data/'
        file = request.files['test_file']
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree('projects/' + session['project_name'] + '/' + 'test_data')
        if not os.path.isdir(UPLOAD_FOLDER):
            os.mkdir(UPLOAD_FOLDER)
        if 'test_file' not in request.files:
            return jsonify('No file part')
        if file.filename == '':
            return jsonify('No selected file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
        else:
            return jsonify('Unsupported File Format')
        model_name = request.form['model_name']
        data_test = pd.read_csv(filepath)
        data = data_test.copy()
        model_path = 'projects/' + session['project_name'] + '/'

        data_out = testing(model_path, data, data_test, model_name)
        return data_out
    else:
        return jsonify('GET method is not supported')


@app.route('/download', methods=['GET', 'POST'])
def download():
    if request.method == 'GET':
        if os.path.isfile('projects/' + session['project_name'] + '/' + "Output.csv"):
            return send_from_directory(os.getcwd(), path='projects/' + session['project_name'] + '/' + "Output.csv",
                                       as_attachment=True)
        else:
            return jsonify("No Predictions")


@app.route('/plot_chart', methods=['GET', 'POST'])
def plot_chart():
    if os.path.isfile('projects/' + session['project_name'] + '/' + "Output.csv"):
        df = pd.read_csv('projects/' + session['project_name'] + '/' + "Output.csv")
        df['date'] = pd.to_datetime(df['date'])
        tdf1 = df
        tdf_ = tdf1.groupby('item')['sales'].sum()
        tdf_.sort_values(ascending=False, inplace=True)
        ind = list(tdf_.index)[:8]
        fig = go.Figure()
        d = {}
        for i in ind:
            tdf = tdf1[tdf1['item'] == i]
            tdf.set_index("date", inplace=True)
            tdf = tdf.resample("W")['sales'].sum()
            tdf = tdf.reset_index()
            tdf['date'] = tdf['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            d["item " + str(i)] = {"date": list(tdf['date']), "sales": list(tdf['sales'])}
            fig.add_trace(go.Scatter(x=tdf["date"], y=tdf["sales"], name="item " + str(i), mode="lines"))
        fig.show()
        return d
    else:
        return jsonify("No Predictions to plot")


@app.route('/kpi', methods=['GET', 'POST'])
def kpi_data():
    if os.path.isfile('projects/' + session['project_name'] + '/' + "Output.csv"):
        df = pd.read_csv('projects/' + session['project_name'] + '/' + "Output.csv")
        print(df)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index("date")
        grouped = df.groupby('item').resample('W')['sales'].sum()
        grouped = grouped.reset_index()
        total_sale = grouped.groupby('date').agg({'sales': "sum"}).reset_index()
        total_sale['date'] = total_sale['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        total_weekly_sales = dict(zip(total_sale.date, total_sale.sales))
        df_max = grouped[grouped.groupby('date').sales.transform('max') == grouped.sales]
        df_max = df_max.sort_values(by='date')
        df_max['date'] = df_max['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        top_selling = df_max.groupby('date').apply(
            lambda df_max: df_max[['sales', 'item']].to_dict(orient='list')).to_dict()
        df_item = grouped.groupby("item").sum()
        df_item = df_item.reset_index()
        df_item['labels'] = df_item.item.apply(lambda x: "item " + str(x))
        item_total_sales = dict(zip(df_item.labels, df_item.sales))
        fig = px.pie(df_item, values="sales", names="labels", labels="labels")
        fig.show()
        return jsonify(total_sales_weekly=total_weekly_sales, top_selling=top_selling,
                       item_total_sales=item_total_sales)


@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate_performance():
    path = os.listdir('projects/' + session['project_name'] + '/' + 'data/')
    df = pd.read_csv('projects/' + session['project_name'] + '/' + 'data/' + path[0])
    df['date'] = pd.to_datetime(df['date'])
    end_date = df['date'].max()
    range_date = datetime.timedelta(days=60)
    split_date = end_date - range_date
    data_test = df[df['date'] >= split_date]
    data_test.reset_index(drop=True, inplace=True)
    data = data_test.copy()
    # data.reset_index(drop=True, inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = [i.month for i in data['date']]
    data['year'] = [i.year for i in data['date']]
    data['day_of_week'] = [i.dayofweek for i in data['date']]
    data['day_of_year'] = [i.dayofyear for i in data['date']]
    data['day'] = [i.day for i in data['date']]
    le = joblib.load('projects/' + session['project_name'] + '/' + 'product_encoder.joblib')
    data['category'] = le.transform(data['category'])
    X = data.drop(["date", "sales"], axis=1)
    path = os.listdir('projects/' + session['project_name'] + '/' + 'models/')
    lis = []
    d = {}
    for i in path:
        model = joblib.load('projects/' + session['project_name'] + '/' + 'models/' + i)
        data_test[i[:-4]] = model.predict(X)
        data_test[i[:-4]] = data_test[i[:-4]].apply(np.ceil)
        data_test[i[:-4]] = data_test[i[:-4]].astype(int)
    ind = data['item'].unique()[:5]
    for i in ind:
        fig = go.Figure()
        tdf = data_test[data_test['item'] == i]
        tdf['date'] = tdf['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        d["item " + str(i)] = {"date": list(tdf['date']), "actual_sales": list(tdf['sales']),
                               path[0][:-4]: list(tdf[path[0][:-4]]), path[1][:-4]: list(tdf[path[1][:-4]]), path[2][:-4]: list(tdf[path[2][:-4]])}
        # d["item " + str(i)] = {"date": list(tdf['date']), "sales": list(tdf['sales'])}
        fig.add_trace(go.Scatter(x=tdf["date"], y=tdf["sales"], name="actual", mode="lines", line=dict(color='red')))
        fig.add_trace(
            go.Scatter(x=tdf["date"], y=tdf[path[0][:-4]], name=path[0][:-4], mode="lines", line=dict(color='blue')))
        fig.add_trace(
            go.Scatter(x=tdf["date"], y=tdf[path[1][:-4]], name=path[1][:-4], mode="lines", line=dict(color='orange')))
        fig.add_trace(
            go.Scatter(x=tdf["date"], y=tdf[path[2][:-4]], name=path[2][:-4], mode="lines", line=dict(color='green')))
        fig.show()
    data_test.to_csv('projects/' + session['project_name'] + "/evaluation.csv", index=False)
    return jsonify(d)


@app.route('/forecasting', methods=["GET", "POST"])
def forecasting():
    if request.method == 'POST':
        UPLOAD_FOLDER_1 = 'data/'
        UPLOAD_FOLDER_2 = 'trained_models/'

        type_of_data = request.form['W_or_M']
        number_of = request.form['Nos']
        if_train = str(request.form['Train'])
        Category = str(request.form['Category'])

        print("train flag ______________", if_train)
        if os.path.isdir(UPLOAD_FOLDER_1):
            shutil.rmtree('data')
        if not os.path.isdir(UPLOAD_FOLDER_1):
            os.mkdir(UPLOAD_FOLDER_1)
        if if_train == 'True' or if_train == 'true' or if_train == 'T':
            print("REMOVE OLD MODELS........................")
            if os.path.isdir(UPLOAD_FOLDER_2):
                shutil.rmtree('trained_models')
            if not os.path.isdir(UPLOAD_FOLDER_2):
                os.mkdir(UPLOAD_FOLDER_2)

        app.config['UPLOAD_FOLDER_1'] = UPLOAD_FOLDER_1
        if 'file' not in request.files:
            return jsonify('No file part')
        file = request.files['file']
        if file.filename == '':
            return jsonify('No selected file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER_1'], file.filename)
            file.save(filepath)
        else:
            return jsonify('Unsupported File Format')

        last_date, cat_1, cat_2, cat_3 = train(filepath, if_train, UPLOAD_FOLDER_2)

        if Category == 'Beauty & Personal Care':
            pass_cat = cat_1
        elif Category == 'Grocery & Gourmet Food':
            pass_cat = cat_2
        elif Category == 'Clothing, Shoes and Jewelry':
            pass_cat = cat_3
        else:
            return jsonify('select given category')

        out = test(last_date, str(type_of_data), int(number_of), pass_cat, UPLOAD_FOLDER_2)

        return out

if __name__ == '__main__':
    app.run(debug=True)
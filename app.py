from flask import Flask, render_template, url_for,request,redirect,session, flash
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
import pickle
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from imblearn.over_sampling import SMOTE
import os
import matplotlib
from matplotlib.cm import viridis
from datetime import datetime


plt.switch_backend('Agg')
matplotlib.use('Agg')

app = Flask(__name__)

# Load the data
data = pd.read_csv("C:/data/heart_disease_data.csv")

# Drop duplicates and handle missing values
data.drop_duplicates(inplace=True)

predictions = []

@app.route("/")
def start():
    return redirect(url_for("home"))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route("/regressor")
def home():
    return render_template("index.html")

#Regression Models go here:-
@app.route("/main")
def main():
    return render_template("main.html")

@app.route("/info")
def info():
    return render_template("info.html")

@app.route("/idea")
def idea():
    return render_template("idea.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/datamenu')
def datamenu():
    return render_template('datamenu.html', columns=data.columns)

@app.route('/heatmap')
def heatmap():
    plt.figure(figsize=(15, 8))
    sns.heatmap(data.corr(), annot=True, cmap='viridis')
    plt.title("Correlation Heatmap")
    plt.savefig('static/heatmap.png')
    plt.close()
    return render_template('heatmap.html', heatmap_url='static/heatmap.png')

@app.route('/bargraph', methods=['GET', 'POST'])
def bargraph():
    columns = data.columns.tolist()

    if request.method == 'POST':
        x_column = request.form['x_column']
        y_column = request.form['y_column']
        plt.figure(figsize=(10, 6))
        sns.barplot(x=data[x_column], y=data[y_column], palette="viridis")
        plt.title(f'Bar Graph: {x_column} vs {y_column}')
        plt.xticks(rotation=45, ha='right')

        graph_path = 'static/bargraph.png'
        plt.savefig(graph_path)
        plt.close()

        return render_template('bargraph.html', columns=columns, graph_url=graph_path)

    return render_template('bargraph.html', columns=columns, graph_url=None)

@app.route('/piechart', methods=["GET","POST"])
def piechart():
    columns = data.columns.tolist()

    if request.method == "POST":
        x_column = request.form['x_column']
        plt.figure(figsize=(10, 6))
        data[x_column].value_counts().plot(kind='pie', autopct='%1.1f%%', colormap='viridis')
        plt.title(f'Pie Chart: {x_column}')
        plt.ylabel('')
        plt.axis('equal')
        plt.gca().set_facecolor('#0b0c10')
        graph_path = 'static/piechart.png'
        plt.savefig(graph_path)
        plt.close()

        return render_template('piechart.html', columns=columns, graph_url=graph_path)
    
    return render_template('piechart.html', columns=columns, graph_url=None)

@app.route('/boxplot', methods=['GET', 'POST'])
def boxplot():
    columns = data.columns.tolist()

    if request.method == 'POST':
        x_column = request.form['x_column']
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=data[x_column], data=data, palette='viridis')
        plt.title(f"Box Plot of {x_column}")
        plt.xticks(rotation=45, ha='right')

        graph_path = 'static/boxplot.png'
        plt.savefig(graph_path)
        plt.close()

        return render_template('boxplot.html', columns=columns, graph_url=graph_path)

    return render_template('boxplot.html', columns=columns, graph_url=None)


@app.route('/lineplot', methods=['GET', 'POST'])
def lineplot():
    columns = data.columns.tolist()

    if request.method == 'POST':
        x_column = request.form['x_column']
        y_column = request.form['y_column']
        
        plt.figure(figsize=(10, 6))
        plt.plot(data[x_column], data[y_column], marker='o', color=viridis(0.5))  # Using Viridis colormap
        plt.title(f'Line plot of {x_column} against {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.xticks(rotation=45, ha='right')

        graph_path = 'static/lineplot.png'
        plt.savefig(graph_path)
        plt.close()

        return render_template('lineplot.html', columns=columns, graph_url=graph_path)

    return render_template('lineplot.html', columns=columns, graph_url=None)


@app.route('/scatterplot', methods=['GET', 'POST'])
def scatterplot():
    columns = data.columns.tolist()

    if request.method == 'POST':
        x_column = request.form['x_column']
        y_column = request.form['y_column']
        z_column = request.form['z_column']

        plt.figure(figsize=(10, 6))

        if z_column == "Scatter":
            sns.scatterplot(data=data, x=data[x_column], y=data[y_column], hue='target')
            plt.title(f"Scatter Plot of {x_column} against {y_column}")
        else:
            sns.regplot(data=data, x=data[x_column], y=data[y_column], scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
            plt.title(f"Regression Plot of {x_column} against {y_column}")

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        graph_path = 'static/scatterplot.png'
        plt.savefig(graph_path)
        plt.close()

        return render_template('scatterplot.html', columns=columns, graph_url=graph_path)

    return render_template('scatterplot.html', columns=columns, graph_url=None)


@app.route('/link')
def link():
    return render_template('link.html')


@app.route('/prediction-menu')
def predmenu():
    return render_template('predmenu.html')



def logisticreg(x, y, solver):
    smote = SMOTE(random_state=0)
    x_res, y_res = smote.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=0)

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)

    model = LogisticRegression(solver=solver, max_iter=1000)
    model.fit(x_train_scaled, y_train)

    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(LogisticRegression(solver=solver), x_train_scaled, y_train, cv=kfold)
    
    return results.mean() * 100, model, scaler

def find_best_solver_log(x, y):
    solvers = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    best_solver = None
    best_score = 0
    best_model = None
    best_scaler = None
    solver_scores = []

    for solver in solvers:
        try:
            score, model, scaler = logisticreg(x, y, solver)
            solver_scores.append((solver, score))
            if score > best_score:
                best_score = score
                best_solver = solver
                best_model = model
                best_scaler = scaler
        except Exception as e:
            solver_scores.append((solver, f"failed with error: {e}"))

    return best_solver, best_score, best_model, best_scaler, solver_scores

def train_and_save_best_model_log(data):
    x = data.drop(columns=['target'])
    y = data['target']
    best_solver, best_score, best_model, best_scaler, solver_scores = find_best_solver_log(x, y)

    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(best_scaler, 'best_scaler.pkl')
    return list(x.columns), solver_scores, best_solver, best_score

def predict_and_store(model, scaler, data_columns, form_data):
    input_values = [float(form_data[col]) for col in data_columns]
    input_values_scaled = scaler.transform([input_values])
    prediction = model.predict(input_values_scaled)[0]
    return prediction

@app.route('/logistic', methods=['GET', 'POST'])
def logistic():
    columns, solver_scores, best_solver, best_score = train_and_save_best_model_log(data)

    if request.method == 'POST':
        form_data = request.form.to_dict()
        data_columns = [col for col in form_data.keys()]

        model = joblib.load('best_model.pkl')
        scaler = joblib.load('best_scaler.pkl')
        prediction = predict_and_store(model, scaler, data_columns, form_data)

        input_data = {col: form_data[col] for col in data_columns}
        input_data['predicted target value'] = prediction
        input_data['date and time'] = datetime.now().strftime("%d%m%Y %H:%M:%S")
        predictions.append(input_data)

    return render_template('logistic.html', columns=columns, solver_scores=solver_scores, best_solver=best_solver, best_score=best_score, predictions=predictions)















def lassoreg(x, y, alpha):
    smote = SMOTE(random_state=0)
    x_res, y_res = smote.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=0)

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)

    model = Lasso(alpha=alpha, max_iter=1000)
    model.fit(x_train_scaled, y_train)

    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(Lasso(alpha=alpha), x_train_scaled, y_train, cv=kfold)
    
    return results.mean() * 100, model, scaler

def find_best_alpha_lasso(x, y):
    alphas = np.logspace(-4, 0, 10)
    best_alpha = None
    best_score = 0
    best_model = None
    best_scaler = None
    alpha_scores = []

    for alpha in alphas:
        try:
            score, model, scaler = lassoreg(x, y, alpha)
            alpha_scores.append((alpha, score))
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_model = model
                best_scaler = scaler
        except Exception as e:
            alpha_scores.append((alpha, f"failed with error: {e}"))

    return best_alpha, best_score, best_model, best_scaler, alpha_scores

def train_and_save_best_model_lasso(data):
    x = data.drop(columns=['target'])
    y = data['target']
    best_alpha, best_score, best_model, best_scaler, alpha_scores = find_best_alpha_lasso(x, y)

    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(best_scaler, 'best_scaler.pkl')
    return list(x.columns), alpha_scores, best_alpha, best_score

def predict_and_store(model, scaler, data_columns, form_data):
    input_values = [float(form_data[col]) for col in data_columns]
    input_values_scaled = scaler.transform([input_values])
    prediction = model.predict(input_values_scaled)[0]
    return prediction

@app.route('/lasso', methods=['GET', 'POST'])
def lasso():
    columns, alpha_scores, best_alpha, best_score = train_and_save_best_model_lasso(data)

    if request.method == 'POST':
        form_data = request.form.to_dict()
        data_columns = [col for col in form_data.keys()]

        model = joblib.load('best_model.pkl')
        scaler = joblib.load('best_scaler.pkl')
        prediction = predict_and_store(model, scaler, data_columns, form_data)

        input_data = {col: form_data[col] for col in data_columns}
        input_data['predicted target value'] = prediction
        input_data['date and time'] = datetime.now().strftime("%d%m%Y %H:%M:%S")
        predictions.append(input_data)

    return render_template('lasso.html', columns=columns, alpha_scores=alpha_scores, best_alpha=best_alpha, best_score=best_score, predictions=predictions)















def ridgereg(x, y, alpha):
    smote = SMOTE(random_state=0)
    x_res, y_res = smote.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=0)

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)

    model = Ridge(alpha=alpha, max_iter=1000)
    model.fit(x_train_scaled, y_train)

    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(Ridge(alpha=alpha), x_train_scaled, y_train, cv=kfold)
    
    return results.mean() * 100, model, scaler

def find_best_alpha_ridge(x, y):
    alphas = np.logspace(-2, 2, 10)
    best_alpha = None
    best_score = 0
    best_model = None
    best_scaler = None
    alpha_scores = []

    for alpha in alphas:
        try:
            score, model, scaler = ridgereg(x, y, alpha)
            alpha_scores.append((alpha, score))
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_model = model
                best_scaler = scaler
        except Exception as e:
            alpha_scores.append((alpha, f"failed with error: {e}"))

    return best_alpha, best_score, best_model, best_scaler, alpha_scores

def train_and_save_best_model_ridge(data):
    x = data.drop(columns=['target'])
    y = data['target']
    best_alpha, best_score, best_model, best_scaler, alpha_scores = find_best_alpha_ridge(x, y)

    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(best_scaler, 'best_scaler.pkl')
    return list(x.columns), alpha_scores, best_alpha, best_score

def predict_and_store(model, scaler, data_columns, form_data):
    input_values = [float(form_data[col]) for col in data_columns]
    input_values_scaled = scaler.transform([input_values])
    prediction = model.predict(input_values_scaled)[0]
    return prediction

@app.route('/ridge', methods=['GET', 'POST'])
def ridge():
    columns, alpha_scores, best_alpha, best_score = train_and_save_best_model_ridge(data)

    if request.method == 'POST':
        form_data = request.form.to_dict()
        data_columns = [col for col in form_data.keys()]

        model = joblib.load('best_model.pkl')
        scaler = joblib.load('best_scaler.pkl')
        prediction = predict_and_store(model, scaler, data_columns, form_data)

        input_data = {col: form_data[col] for col in data_columns}
        input_data['predicted target value'] = prediction
        input_data['date and time'] = datetime.now().strftime("%d%m%Y %H:%M:%S")
        predictions.append(input_data)

    return render_template('ridge.html', columns=columns, alpha_scores=alpha_scores, best_alpha=best_alpha, best_score=best_score, predictions=predictions)















def svr_model(x, y, kernel):
    smote = SMOTE(random_state=0)
    x_os, y_os = smote.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x_os, y_os, test_size=0.3, random_state=0)

    scaler = StandardScaler().fit(x_train)
    x_train_sc = scaler.transform(x_train)

    model = SVR(kernel=kernel)
    model.fit(x_train_sc, y_train)

    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(model, x_train_sc, y_train, cv=kfold)

    return results.mean() * 100, model, scaler

def find_best_kernel_svr(x, y):
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    best_kernel = None
    best_score = 0
    best_model = None
    best_scaler = None
    kernel_scores = []

    for kernel in kernels:
        try:
            score, model, scaler = svr_model(x, y, kernel)
            kernel_scores.append((kernel, score))
            if score > best_score:
                best_score = score
                best_kernel = kernel
                best_model = model
                best_scaler = scaler
        except Exception as e:
            kernel_scores.append((kernel, f"failed with error: {e}"))

    return best_kernel, best_score, best_model, best_scaler, kernel_scores

def train_and_save_best_svr_model(data):
    x = data.drop(columns=['target'])
    y = data['target']
    best_kernel, best_score, best_model, best_scaler, kernel_scores = find_best_kernel_svr(x, y)

    joblib.dump(best_model, 'best_svr_model.pkl')
    joblib.dump(best_scaler, 'best_svr_scaler.pkl')

    return list(x.columns), kernel_scores, best_kernel, best_score

def predict_and_store(model, scaler, data_columns, form_data):
    input_values = [float(form_data[col]) for col in data_columns]
    input_values_scaled = scaler.transform([input_values])
    prediction = model.predict(input_values_scaled)[0]
    return prediction


@app.route('/svr', methods=['GET', 'POST'])
def svr():
    columns, kernel_scores, best_kernel, best_score = train_and_save_best_svr_model(data)

    if request.method == 'POST':
        form_data = request.form.to_dict()
        data_columns = [col for col in form_data.keys()]

        model = joblib.load('best_svr_model.pkl')
        scaler = joblib.load('best_svr_scaler.pkl')
        prediction = predict_and_store(model, scaler, data_columns, form_data)

        input_data = {col: form_data[col] for col in data_columns}
        input_data['predicted target value'] = prediction
        input_data['date and time'] = datetime.now().strftime("%d%m%Y %H:%M:%S")
    
        predictions.append(input_data)

    return render_template('svr.html', columns=columns, kernel_scores=kernel_scores, best_kernel=best_kernel, best_score=best_score, predictions=predictions)
















def rforestreg(x, y, n_estimators):
    smote = SMOTE(random_state=0)
    x_res, y_res = smote.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=0)

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)

    model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=0)
    model.fit(x_train_scaled, y_train)

    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(model, x_train_scaled, y_train, cv=kfold, n_jobs=-1)
    
    return results.mean() * 100, model, scaler

def find_best_n_estimators_rf(x, y):
    n_estimators_range = range(50, 501, 50)
    best_n_estimators, best_score, best_model, best_scaler = None, 0, None, None
    n_estimators_scores = []

    for n_estimators in n_estimators_range:
        score, model, scaler = rforestreg(x, y, n_estimators)
        n_estimators_scores.append((n_estimators, score))
        if score > best_score:
            best_score = score
            best_n_estimators = n_estimators
            best_model = model
            best_scaler = scaler

    return best_n_estimators, best_score, best_model, best_scaler, n_estimators_scores

def train_and_save_best_model_rf(data):
    x = data.drop(columns=['target'])
    y = data['target']
    best_n_estimators, best_score, best_model, best_scaler, n_estimators_scores = find_best_n_estimators_rf(x, y)

    joblib.dump(best_model, 'best_rf_model.pkl')
    joblib.dump(best_scaler, 'best_rf_scaler.pkl')
    return list(x.columns), n_estimators_scores, best_n_estimators, best_score

def predict_and_store(model, scaler, data_columns, form_data):
    input_values = [float(form_data[col]) for col in data_columns]
    input_values_scaled = scaler.transform([input_values])
    prediction = model.predict(input_values_scaled)[0]
    return prediction

@app.route('/rforest', methods=['GET', 'POST'])
def rforest():
    columns, n_estimators_scores, best_n_estimators, best_score = train_and_save_best_model_rf(data)

    if request.method == 'POST':
        form_data = request.form.to_dict()
        data_columns = [col for col in form_data.keys()]

        model = joblib.load('best_rf_model.pkl')
        scaler = joblib.load('best_rf_scaler.pkl')
        prediction = predict_and_store(model, scaler, data_columns, form_data)

        input_data = {col: form_data[col] for col in data_columns}
        input_data['predicted target value'] = prediction
        input_data['date and time'] = datetime.now().strftime("%d%m%Y %H:%M:%S")
        predictions.append(input_data)

    return render_template('rforest.html', columns=columns, n_estimators_scores=n_estimators_scores, best_n_estimators=best_n_estimators, best_score=best_score, predictions=predictions)















def mlp_model(x, y, activation, solver):
    print("Starting SMOTE")
    smote = SMOTE(random_state=0)
    x_os, y_os = smote.fit_resample(x, y)
    
    print("Splitting data")
    x_train, x_test, y_train, y_test = train_test_split(x_os, y_os, test_size=0.3, random_state=0)

    print("Scaling data")
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)

    print("Initializing MLPRegressor")
    model = MLPRegressor(hidden_layer_sizes=(100,), activation=activation, solver=solver, max_iter=500, random_state=0)
    print("Training model")
    model.fit(x_train_scaled, y_train)

    print("Performing cross-validation")
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(model, x_train_scaled, y_train, cv=kfold)

    return results.mean() * 100, model, scaler

def find_best_hyperparameters_mlp(x, y):
    activations = ['identity', 'logistic', 'tanh', 'relu']
    solvers = ['sgd', 'adam']
    best_activation = ""
    best_solver = ""
    best_score = 0
    best_model = None
    best_scaler = None
    param_scores = []

    for activation in activations:
        for solver in solvers:
            try:
                score, model, scaler = mlp_model(x, y, activation, solver)
                param_scores.append((activation, solver, score))
                if score > best_score:
                    best_score = score
                    best_activation = activation
                    best_solver = solver
                    best_model = model
                    best_scaler = scaler
            except Exception as e:
                param_scores.append((activation, solver, f"failed with error: {e}"))

    return best_activation, best_solver, best_score, best_model, best_scaler, param_scores

def train_and_save_best_model_mlp(data):
    x = data.drop(columns=['target'])
    y = data['target']
    best_activation, best_solver, best_score, best_model, best_scaler, param_scores = find_best_hyperparameters_mlp(x, y)

    joblib.dump(best_model, 'best_mlp_model.pkl')
    joblib.dump(best_scaler, 'best_mlp_scaler.pkl')
    return list(x.columns), param_scores, best_activation, best_solver, best_score

def predict_and_store(model, scaler, data_columns, form_data):
    input_values = [float(form_data[col]) for col in data_columns]
    input_values_scaled = scaler.transform([input_values])
    prediction = model.predict(input_values_scaled)[0]
    return prediction

@app.route('/mlp', methods=['GET', 'POST'])
def mlp():
    columns, param_scores, best_activation, best_solver, best_score = train_and_save_best_model_mlp(data)

    if request.method == 'POST':
        form_data = request.form.to_dict()
        data_columns = [col for col in form_data.keys()]

        model = joblib.load('best_mlp_model.pkl')
        scaler = joblib.load('best_mlp_scaler.pkl')
        prediction = predict_and_store(model, scaler, data_columns, form_data)

        input_data = {col: form_data[col] for col in data_columns}
        input_data['predicted target value'] = prediction
        input_data['date and time'] = datetime.now().strftime("%d%m%Y %H:%M:%S")
        predictions.append(input_data)

    return render_template('mlp.html', columns=columns, param_scores=param_scores, best_activation=best_activation, best_solver=best_solver, best_score=best_score, predictions=predictions)










if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)

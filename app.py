from re import template
import re
from flask import Flask, render_template, url_for, redirect, request
import pandas as pd
import csv
import matplotlib.pyplot as plt          # plotting
import numpy as np                       # dense matrices
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from flask_wtf import FlaskForm
from werkzeug.datastructures import Range
from wtforms import StringField, SubmitField, IntegerField
from wtforms.validators import DataRequired
import numpy as np


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/regressor")
def regressor():
    dataset = 'e-commerce.csv'
    data = pd.read_csv(dataset)
    df = data.drop("item_basic/name", axis=1)
    content = df.head()

    # mendefinisikan atribut (fitur/variabel independen) dan target (class/variabel dependen)
    X = df[['item_basic/stock', 'item_basic/sold',
            'item_basic/historical_sold']]
    y = df['item_basic/price']

    # merubah tipe data dari dataframe ke numpy array
    X = np.asarray(X)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # create a regressor object
    regressor = DecisionTreeRegressor(random_state=0)

    # fit the regressor with X and Y data
    regressor.fit(X, y)
    content2 = regressor.score(X_test, y_test)

    return render_template("regressor.html", content2=content2)


@app.route("/contents", methods=["POST", "GET"])
def contents():
    if request.method == "POST":
        stok = request.form["stok"]
        jual = request.form["sold"]
        terjual = request.form["histori"]
        # harga = request.form["price"]
        dataset = 'e-commerce.csv'
        data = pd.read_csv(dataset)
        df = data.drop("item_basic/name", axis=1)

        # mendefinisikan atribut (fitur/variabel independen) dan target (class/variabel dependen)
        X = df[['item_basic/stock', 'item_basic/sold',
                'item_basic/historical_sold']]
        y = df['item_basic/price']

        # # merubah tipe data dari dataframe ke numpy array
        X = np.asarray(X)
        y = np.asarray(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        # # create a regressor object
        regressor = DecisionTreeRegressor(random_state=0)

        # # fit the regressor with X and Y data
        regressor.fit(X, y)
        49  # mendefinisikan kolom yang ada pada dataset
        listColumns = ['item_basic/stock', 'item_basic/sold',
                       'item_basic/historical_sold']

        # # mendefinisikan list untuk menampung data yang akan diinput user
        singleData = []

        # # mendefinisikan list untuk menampung data
        dataTest = []

        data1 = (stok)
        tag1 = float(data1)
        data2 = (jual)
        tag2 = float(data2)
        data3 = (terjual)
        tag3 = float(data3)
        singleData.append(tag1)
        singleData.append(tag2)
        singleData.append(tag3)

        # # menggabungkan tiap single data ke dalam data test
        dataTest.append(singleData)
        # convert dari list ke numpy array
        dataTest = np.array(dataTest)
        # proses prediksi dari data yang diinput user
        prediksi = regressor.predict(dataTest).tolist()

        # print hasil prediksi
        hasil = ("Hasil prediksi yaitu harga:", prediksi[0])
        return render_template("models.html", usr=hasil)
    else:
        return render_template("contents.html")


# @ app.route("/user")
# def user(usr):
#     return f"<h1>{usr}</h1>"


if __name__ == "__main__":
    app.run(debug=True)

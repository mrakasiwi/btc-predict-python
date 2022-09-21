from unittest import result
from flask import Flask, render_template, redirect, url_for, request

import pandas as pd
import numpy as np
import locale
import datetime
import time
import math
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

df = pd.read_csv('bitcoin_data.csv', index_col=False)
dfs = df[:-7]
actual = df[-7:]

to_row = int(len(dfs)*0.8)

def train():
    training_data = list(df[:to_row]['Adj Close'])
    return training_data

def train2():
    training_data = list(dfs[:]['Adj Close'])
    return training_data

def test():
    test_data = list(df[-7:]['Adj Close'])
    return test_data

def last_date():
    last_date = list(dfs[-1:]["Date"])[0]
    return last_date

def percentage(n,predict):
    training_data =  train2()
    last_price = training_data[-1]
    percent = []
    for i in range(n):
        percentage = (predict[i]-last_price)/last_price*100
        percent.append(percentage)
    return percent

def get_pdate(n):
    pdate = [d.strftime('%Y-%m-%d') 
            for d in pd.date_range(start = last_date(), periods = n+1)]
    dates = pdate[1:]
    return dates

def to_currency(data):
    locale.setlocale( locale.LC_ALL, 'en_US' )
    currency = [locale.currency( cr, grouping=True ) 
         for cr in data]
    return currency

@app.route("/")
def main():
    data = [round(ds, 2) for ds in dfs[:]["Adj Close"]]
    data_us = to_currency(data)
    date = list(dfs[:]["Date"])
    return render_template('index.html', menu='home', data=data, data_us=data_us, date=date, len=len, list=list, round=round)

@app.route("/calculation")
def calculation():
    test = []
    j=0
    while j < len(dfs[to_row:]):
        test.append(dfs['Adj Close'][to_row+j])
        j+=1
    
    testing_data = list(test)
    training_data = train()
    
    model_prediction = []
    n_test_obser = len(testing_data)
    
    date_range = list(dfs[to_row:]['Date'])

    for i in range(n_test_obser):
        model = ARIMA(training_data, order = (4,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        #first prediction
        yhat = list(output[0])[0]
        model_prediction.append(yhat)
        actual_test_value = testing_data[i]
        training_data.append(actual_test_value)
    
    model = model_fit.summary()
    mape = np.mean(np.abs(np.array(model_prediction) - np.array(testing_data)) / np.abs(testing_data))*100

    return render_template('calculate.html', menu='calculation', data=dfs, training=training_data, testing=testing_data, train2=train(), model=model, mape=mape, prediction=model_prediction, date=date_range, len=len, list=list, round=round)

@app.route("/predictions", methods=['GET', 'POST'])
def predictions():
    model_prediction = []
    date = list(df[-7:]['Date'])
    training_data = train2()
    test_data = test()
    
    if request.method == 'POST':
        p = int(request.form['p'])
        d = int(request.form['d'])
        q = int(request.form['q'])
        day = int(request.form['day'])
   
        for i in range(day):
            model = ARIMA(training_data, order = (p,d,q))
            model_fit = model.fit()
            output = model_fit.forecast()
            #first prediction
            yhat = list(output[0])[0]
            model_prediction.append(yhat)
            actual_test_value = test_data[i]
            training_data.append(actual_test_value)

        #model = ARIMA(training_data, order = (p,d,q))
        #model_fit = model.fit()
        #predict_result = model_fit.forecast(day)
        #model_prediction = list(predict_result[0])
        prediction = [round(predic, 2) for predic in model_prediction]
        predic_us = to_currency(prediction)
        test_data = list(df[-day:]['Adj Close'])
        percentages = percentage(day,model_prediction)
        pdate = get_pdate(day)
        #mape = np.mean(np.abs(np.array(model_prediction) - np.array(test_data)) / np.abs(test_data))*100
        return render_template('predics.html', menu='predictions', data=dfs, predic=prediction, predic_us=predic_us, percents=percentages, list=list, pdate=pdate, len=len, date=date, round=round, p=p, d=d, q=q)

    elif request.method == 'GET':
        return render_template('predics.html', menu='predictions', len=len, predic=0)
    
    else:
        return render_template('predics.html', menu='predictions', len=len, predic=0)
    
if __name__ == "__main__":
    app.run(debug=True)
from unittest import result
from flask import Flask, render_template, redirect, url_for, request

import pandas as pd
import numpy as np
import locale
import datetime
import time
import math

from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

df = pd.read_csv('bitcoin_data.csv', index_col=False)
dfs = df[:-7]
actual = df[-7:]

to_row = int(len(dfs)*0.8)

str_trdates = 0
end_trdates = 0
mape = 0
ariman = 0
models = 0
arima_model = 0
ntest = 0

def train(start, end):
    training_data = df[start:end]['Adj Close']
    return training_data

def train2():
    training_data = list(dfs[:]['Adj Close'])
    return training_data

def test(n, start):
    test_data = list(df[start:start+n]['Adj Close'])
    return test_data

def get_date_range(start, end):
    date_range = df[start:end+1]["Date"]
    return date_range

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

def get_pdate(n, start):
    pdate = [d.strftime('%Y-%m-%d') 
            for d in pd.date_range(start = start, periods = n+1)]
    dates = pdate[1:]
    return dates

def to_currency(data):
    locale.setlocale( locale.LC_ALL, 'en_US' )
    currency = [locale.currency( cr, grouping=True ) 
         for cr in data]
    return currency

def get_mape(pred, test):
    mape = np.mean(np.abs(np.array(pred) - np.array(test)) / np.abs(test))*100
    return mape

def predics(n, result):
    model_prediction = result.predict(n)
    return model_prediction

def fit_model(training, p, d, q):
    model = ARIMA(training, order = (p,d,q))
    model_fit = model.fit()
    return model_fit

def get_index(Date):
    index = df[df["Date"]==Date].index.values
    return index[0]
    

@app.route("/")
def main():
    data = [round(ds, 2) for ds in df[:]["Adj Close"]]
    data_us = to_currency(data)
    date = list(df[:]["Date"])
    return render_template('index.html', menu='dashboard', data=data, data_us=data_us, date=date, len=len, list=list, round=round)

@app.route("/calculation", methods=['GET', 'POST'])
def calculation():
    global ntest
    global str_trdates
    global end_trdates
    global ariman
    global models
    global arima_model
    
    if request.method == 'POST':
        
        str_trdate = request.form['strdate_tr']
        end_trdate = request.form['enddate_tr']
            
        if str_trdate != str_trdates or end_trdate != end_trdates :
            str_trdates = str_trdate
            end_trdates = end_trdate
            training_data = train(get_index(str_trdates), get_index(end_trdates)+1)
            str_date = df.loc[get_index(str_trdates)]["Date"]
            end_date = df.loc[get_index(end_trdates)]["Date"]
            
            date_range = list(get_date_range(get_index(str_trdates), get_index(end_trdates)))
        
            arima_model = auto_arima(training_data,trace=True, error_action='ignore', start_p=1,start_q=1,max_p=4,max_q=4,
                    suppress_warnings=True,stepwise=False,seasonal=False)
            arima = arima_model.fit(training_data)
            
            ariman= str(arima)[0:13]
            
            models = arima_model.summary()
            tabel1 = models.tables[0].as_html()
            tabel2 = models.tables[1].as_html()
            tabel3 = models.tables[2].as_html()
            testing_data = 0
            model_prediction=0
            ts_date = 0
            mape = 0
            count_ts=len(df[get_index(end_trdate):])
            if count_ts > 7:
                count_ts = 8
            return render_template('calculate.html', menu='calculation', df=df, data=dfs, training=training_data, testing=testing_data,  model=models, mape=round(mape,4), prediction=model_prediction, date=date_range, ts_date = ts_date, len=len, list=list, round=round, arima_model = ariman, str_date = str_date, end_date = end_date, count_ts=count_ts, table1=tabel1, table2=tabel2, table3=tabel3)
        
        else:
            ntest = int(request.form['day'])

            str_date = df.loc[get_index(str_trdates)]["Date"]
            end_date = df.loc[get_index(end_trdates)]["Date"]
            training_data = df[get_index(str_trdates): get_index(end_trdates)+1]["Adj Close"]
            date_range = list(get_date_range(get_index(str_trdates), get_index(end_trdates)))
            
            count_ts=len(df[get_index(end_trdates):])
            testing_data = 0
            model_prediction=0
            ts_date = 0
            mape = 0
            tabel1 = models.tables[0].as_html()
            tabel2 = models.tables[1].as_html()
            tabel3 = models.tables[2].as_html()
            
            
            p = int(ariman[7:8])
            d = int(ariman[9:10])
            q = int(ariman[11:12])
            
            if ntest != 0 :
                testing_data = test(ntest, get_index(end_trdates)+1)
                testing_data = [round(test, 2) for test in testing_data]
                ts_date = list(get_date_range(get_index(end_trdates)+1, get_index(end_trdates)+ntest+1))
                model = ARIMA(training_data, order = (p,d,q))
                model_fit = model.fit()
                model_prediction = list(model_fit.forecast(ntest)[0])
                model_prediction = [round(predic, 2) for predic in model_prediction]
                
                #model_prediction = list(predics(n_test_obser, arima_model))
                mape = get_mape(model_prediction, testing_data)

            return render_template('calculate.html', menu='calculation', df=df, data=dfs, training=training_data, testing=testing_data, model=models, mape=round(mape,4), prediction=model_prediction, date=date_range, ts_date = ts_date, len=len, list=list, round=round, arima_model = ariman, str_date = str_date, end_date = end_date, count_ts=count_ts, table1=tabel1, table2=tabel2, table3=tabel3)
        
    else:
        training_data = train2()
        date_range = list(actual[:]['Date'])
        
        str_date = list(dfs[:1]["Date"])
        end_date = list(dfs[-1:]["Date"])
        
        ntest = 0
        str_trdates = 0
        end_trdates = 0
        ariman = 0
        models = 0
        arima_model = 0
        return render_template('calculate.html', menu='calculation', df=df, arima=ariman, list=list, date=date_range, str_date = str_date, end_date = end_date, len=len)

@app.route("/predictions", methods=['GET', 'POST'])
def predictions():
    model_prediction = []
    global str_trdates
    global end_trdates
    
    if request.method == 'POST':
        p = int(request.form['p'])
        d = int(request.form['d'])
        q = int(request.form['q'])
        day = int(request.form['day'])
        str_trdate = request.form['strdate_tr']
        end_trdate = request.form['enddate_tr']
   
        training_data = train(get_index(str_trdate), get_index(end_trdate)+1)
        model = ARIMA(training_data, order = (p,d,q))
        model_fit = model.fit()
        model_prediction = list(model_fit.forecast(day)[0])

        #model = ARIMA(training_data, order = (p,d,q))
        #model_fit = model.fit()
        #predict_result = model_fit.forecast(day)
        #model_prediction = list(predict_result[0])
        date = list(get_date_range(get_index(str_trdate), get_index(end_trdate)))
        prediction = [round(predic, 2) for predic in model_prediction]
        predic_us = to_currency(prediction)
        percentages = percentage(day,model_prediction)
        pdate = get_pdate(day, end_trdate)
        #mape = np.mean(np.abs(np.array(model_prediction) - np.array(test_data)) / np.abs(test_data))*100
        return render_template('predics.html', menu='predictions', df=df, data=dfs, predic=prediction, predic_us=predic_us, percents=percentages, list=list, pdate=pdate, len=len, date=date, round=round, p=p, d=d, q=q, str_date = str_trdate, end_date = end_trdate)

    elif request.method == 'GET':
        return render_template('predics.html', menu='predictions', len=len, predic=0, df = df, list=list, str_date = str_trdates, end_date = end_trdates)
    
    else:
        return render_template('predics.html', menu='predictions', len=len, predic=0, df = df, str_date = str_trdates, end_date = end_trdates, list=list)
    
if __name__ == "__main__":
    app.run(debug=True)

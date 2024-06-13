# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:25:27 2018

@author: Giorgos
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller,acf, pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pandas import datetime
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error



def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

data = pd.read_csv('ASE_Indices.csv', parse_dates=['Date'], index_col='Date')
series=data['Close'] 
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
validation = list()
forecast=list()

series_log = np.log(series)
series_diff = series_log - series_log.shift()
series_diff.dropna(inplace=True)



def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')  
    plt.figure()
       
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries,autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


def ACF_PACF():
    
#    ACF and PACF plots:
    lag_acf = acf(series, nlags=20)
    lag_pacf = pacf(series, nlags=20, method='ols')
    lag_acf = acf(series_diff, nlags=20)
    lag_pacf = pacf(series_diff, nlags=20, method='ols')
    
    #Plot ACF: 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(series_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(series_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(series_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(series_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    
    plot_acf(series, lags=30)
    pyplot.show()
    plot_pacf(series, lags=30)
    pyplot.show()
    
    plot_acf(series_diff, lags=30)
    pyplot.show()
    plot_pacf(series_diff, lags=30)
    pyplot.show()
      

def validation_proc(): 

    for t in range(len(test)):
        model = ARIMA(history, order=(1,1,1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        validation.append(yhat[0])
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    
    error = mean_absolute_error(test, validation)
    print(model_fit.summary())

    # plot
    li=train.tolist()+validation
    df=pd.DataFrame({'ExpPrice':history,'PredPrice':li})
    ax = df['ExpPrice'].plot(color='b')
    ax = df['PredPrice'].plot(color='b')
    df.loc[df.index >= len(train), 'PredPrice'].plot(color='r', ax=ax)
    plt.title('Overall Model Fit')
    pyplot.figure()
    
    pyplot.plot(test,label='Expected Price')
    pyplot.plot(validation, color='red',label='Predicted Price')
    plt.legend(loc='best') 
    plt.title('MAE:%f' %error)
    pyplot.show()
    

def forecast_proc():
    
 
     model = ARIMA(X, order=(1,1,1))
     model_fit = model.fit(disp=0)
     output = model_fit.forecast(steps=30)
     forecast = output[0]

            
     plt.plot(forecast, color='red',label='Predicted Price')
     plt.legend(loc='best')
     pyplot.show()
            
#plt.ylim(650,950)
#test_stationarity(series)
#
#plt.ylim(-0.05,0.055)
#test_stationarity(series_diff) 
#
#ACF_PACF()

validation_proc()

#forecast_proc()
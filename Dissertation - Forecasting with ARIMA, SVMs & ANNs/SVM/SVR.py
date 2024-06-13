# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 18:37:36 2018

@author: Giorgos
"""

from sklearn.svm import SVR
from pandas import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.metrics import mean_absolute_error




def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

data = pd.read_csv('data_norm_svr.csv', parse_dates=['Date'], index_col='Date')
X = data.drop(['Close','Day 8','Day 7','Day 6','Day 5','Day 4','Day 3'], axis=1)
#X= X.values
y = data['Close'].values.reshape((251,1))
size=200
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = X[0:size], X[size:], y[0:size], y[size:]

# #############################################################################
# Fit regression model


Cs = [2**-5,2**-4,2**-3,2**-2,2**-1,2**0,2**1,2**2,2**3,2**4,2**5,2**6,2**7,2**8,2**9,2**10,2**11,2**12,2**13,2**14,2**15]
gammas = [2**-15,2**-14,2**-13,2**-12,2**-11,2**-10,2**-9,2**-8,2**-7,2**-6,2**-5,2**-4,2**-3,2**-2,2**-1,2**0,2**1,2**2,2**3]
    
svr_rbf = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=10,
               param_grid={"C": Cs,
                           "gamma": gammas})
    
svr_lin = GridSearchCV(SVR(kernel='linear'), cv=10,
               param_grid={"C": Cs})

svr_poly = GridSearchCV(SVR(kernel='poly'), cv=10,
               param_grid={"C": Cs})


y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
y_poly = svr_poly.fit(X_train, y_train).predict(X_test)

print(svr_rbf.best_params_)
print(svr_lin.best_params_)
print(svr_poly.best_params_)
print(mean_absolute_error(y_test, y_rbf))
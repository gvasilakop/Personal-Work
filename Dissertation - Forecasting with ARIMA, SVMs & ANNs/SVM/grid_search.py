# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:27:10 2018

@author: Giorgos
"""


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
from sklearn import svm


data = pd.read_csv('data_norm.csv')
X = data.drop('Class', axis=1)  
y = data['Class'] 
kernels=['rbf','linear']

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

for svm_kernel in kernels:

    Cs = [2**-5,2**-4,2**-3,2**-2,2**-1,2**0,2**1,2**2,2**3,2**4,2**5,2**6,2**7,2**8,2**9,2**10,2**11,2**12,2**13,2**14,2**15]
    gammas = [2**-15,2**-14,2**-13,2**-12,2**-11,2**-10,2**-9,2**-8,2**-7,2**-6,2**-5,2**-4,2**-3,2**-2,2**-1,2**0,2**1,2**2,2**3]
    param_grid = {'C': Cs, 'gamma' : gammas}
    target_names = ['class 0', 'class 1']
    grid_search = GridSearchCV(svm.SVC(kernel=svm_kernel), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    y_true, y_pred = y_test, grid_search.predict(X_test)

    
    print('Kernel: ',svm_kernel,'Folds: 5')
    print(grid_search.best_params_)
    print(classification_report(y_true, y_pred,target_names=target_names))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
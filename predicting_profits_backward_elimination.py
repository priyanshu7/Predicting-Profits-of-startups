# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 00:28:35 2018

@author: priyanshumehta
"""

#Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Categorical Data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[: , 3 ] = labelencoder_X.fit_transform(X[: , 3 ])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Dummy Variable Trap

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training Set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)

#Predicting the Test Set Results

y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination

import statsmodels.formula.api as sm
#add columns of 1 as x0 
#X = np.append(arr= X , values = np.ones((50,1)).astype(int), axis = 1) because it will append at back, so append X to ones column
X = np.append(arr = np.ones((50,1)).astype(int) , values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()

#Predicting the Test Set Results

y_OPTpred = regressor_OLS.predict(X_opt)
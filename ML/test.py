#!/usr/bin/env python3
import main
from metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings

# Все выполнено в этом блоке, потому что это отключает ПРЕДУПРЕЖДЕНИЯ СКЛЕРН!
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Do stuff here
    data = pd.read_csv('./testing_files/class.txt')
    X = data.drop('0',axis=1).values
    y = data['0'].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)
    list_of_models = [main.LogisticRegression(),main.LinearRegression()]
    # Building our model!
    model = main.MultiLayerPerceptron(X_train,y_train)
    a = model.fit(100)
    prediction = model.predict(X_test)
    print(prediction)
    #metricss = confusion_matrix(y_test,prediction)
    #print(metricss)
    try:
        metricss = metrics.accuracy_score(y_test,prediction)
        print(metricss,'This is my accuracy score')
    except ValueError:
        metricsss = metrics.mean_squared_error(y_test,prediction)
        print(metricsss,'This is my MSE!')


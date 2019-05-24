import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Слздание маленького дата-сета
training_data = pd.read_csv('class.txt')
training_data = training_data.values
training_data,test_data = train_test_split(training_data,test_size=0.20)
# Заголовки
header = ['first','second','class']


list_of_models = [LinearRegression(),DecisionTreeClassifier(),SVR()]

for i in list_of_models:
    i.fit()
    i.predict()



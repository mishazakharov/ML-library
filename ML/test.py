from main import GradientBoostingClassifier
from metrics import f1_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings

# Все выполнено в этом блоке, потому что это отключает ПРЕДУПРЕЖДЕНИЯ СКЛЕРН!
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Do stuff here
    data = pd.read_csv('./testing_files/class.txt')
    X = data.drop('0',axis=1).values
    y = data['0'].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.50)
    list_of_models = [DecisionTreeClassifier(),LogisticRegression()]
    model =  GradientBoostingClassifier()
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    metrics = f1_score(y_test,prediction)
    print(metrics)


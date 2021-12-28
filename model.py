import pandas as pd

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection

raw_data = pd.read_excel(r"C:\Users\user\Downloads\DSCT\Health Prediction\dataGYM.xlsx")

raw_data.Prediction = LabelEncoder().fit_transform(raw_data.Prediction)

del raw_data['BMI']

del raw_data['Class']

x = raw_data.iloc[:, :-1]

y = raw_data.iloc[:, -1]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3)

"""RANDOM FOREST """

model1 = RandomForestClassifier(n_estimators=150)
model1.fit(x_train, y_train.values.ravel())

print(f"\nresult for {model1}")

expected1 = y_test
predicted1 = model1.predict(x_test)

print(metrics.classification_report(expected1, predicted1))

"""knn"""

model2 = KNeighborsClassifier(11)
model2.fit(x_train, y_train.values.ravel())

print(f"\nresult for {model2}")

expected2 = y_test
predicted2 = model2.predict(x_test)
print(metrics.classification_report(expected2, predicted2))

# Saving model to disk
pickle.dump(model1, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[40, 5.6, 70]]))

#
pickle.dump(model, open("model1.pkl", "wb"))

model = pickle.load(open("model1.pkl", "rb"))

print(model.predict([[40, 5.6, 70]]))

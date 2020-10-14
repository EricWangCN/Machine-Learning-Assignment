import pandas as pd
import numpy as np

data = pd.read_csv(
    '~/Desktop/Study/Assignment/机器学习/Machine-Learning-Assignment/Lab1/Assignment/data/wine_quality/winequality-white.csv',
    delimiter=";")

from sklearn.linear_model import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import *
from sklearn.model_selection import cross_val_predict

# Unary linear regression
features = ['alcohol']
x = data[features]
y = data['quality']
model1 = LinearRegression()
prediction1 = cross_val_predict(model1, x, y, cv=10)
accuracy1 = accuracy_score(y, prediction1.round())
print(accuracy1)

# Multiple linear regression
features = ['fixed acidity', 'fixed acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
x = data[features]
y = data['quality']
model2 = LinearRegression()
prediction2 = cross_val_predict(model2, x, y, cv=10)
accuracy2 = accuracy_score(y, prediction2.round())
print(accuracy2)

# Log linear regression
features = ['alcohol']
x = data[features]
y = data['quality']
model3 = LinearRegression()
prediction3 = np.exp(cross_val_predict(model3, x, y, cv=10))
accuracy3 = accuracy_score(y, prediction3.round())
print(accuracy3)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))
# plt.plot(data['alcohol'].values, data['quality'].values, '.', label='data')
# plt.plot(data['alcohol'].values, prediction3, '-', label='prediction')
# plt.legend()
# plt.show()

# Log odds regression
features = ['alcohol']
x = data[features]
y = data['quality']
model4 = LogisticRegression()
prediction4 = cross_val_predict(model4, x, y, cv=10)
accuracy4 = accuracy_score(y, prediction4.round())
print(accuracy4)

# Linear discriminant analysis
features = ['alcohol']
x = data[features]
y = data['quality']
model5 = LinearDiscriminantAnalysis()
prediction5 = cross_val_predict(model5, x, y, cv=10)
accuracy5 = accuracy_score(y, prediction5.round())
print(accuracy5)

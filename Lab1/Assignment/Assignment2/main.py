import numpy as np
import pandas as pd

# 读取数据
data = pd.read_csv('/Users/wangzilong/Desktop/Study/Assignment/机器学习/Machine-Learning-Assignment/Lab1/Assignment/data/kaggle_house_price_prediction/kaggle_hourse_price_train.csv')

# 丢弃有缺失值的特征（列）
data.dropna(axis = 1, inplace = True)

# 只保留整数的特征
data = data[[col for col in data.dtypes.index if data.dtypes[col] == 'int64']]

data.info()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict

model = LinearRegression()

features = ['LotArea']
x = data[features]
y = data['SalePrice']

prediction = cross_val_predict(model, x, y, cv = 10)

prediction.shape

mean_absolute_error(prediction, data['SalePrice'])

mean_squared_error(prediction, data['SalePrice']) ** 0.5

# YOUR CODE HERE
# Model1
features = ['LotArea', 'YearBuilt', 'GrLivArea']
x = data[features]
y = data['SalePrice']
prediction1 = cross_val_predict(model, x, y, cv = 10)
print('Model1, MAE:', mean_absolute_error(prediction1, data['SalePrice']))
print('Model1, SMSE:', mean_squared_error(prediction1, data['SalePrice']) ** 0.5)

# Model2
features = ['BsmtUnfSF', 'FullBath', 'BedroomAbvGr']
x = data[features]
y = data['SalePrice']
prediction2 = cross_val_predict(model, x, y, cv = 10)
print('Model2, MAE:', mean_absolute_error(prediction2, data['SalePrice']))
print('Model2, SMSE:', mean_squared_error(prediction2, data['SalePrice']) ** 0.5)

# Model3
features = ['GarageCars', 'ScreenPorch', 'KitchenAbvGr']
x = data[features]
y = data['SalePrice']
prediction3 = cross_val_predict(model, x, y, cv = 10)
print('Model3, MAE:', mean_absolute_error(prediction3, data['SalePrice']))
print('Model3, SMSE:', mean_squared_error(prediction3, data['SalePrice']) ** 0.5)
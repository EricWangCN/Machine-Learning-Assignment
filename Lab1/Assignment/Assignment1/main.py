import numpy as np
import pandas as pd

# 使用pandas读取csv数据
data = pd.read_csv(
    '/Users/wangzilong/Desktop/Study/Assignment/机器学习/Machine-Learning-Assignment/Lab1/Assignment/data/kaggle_house_price_prediction/kaggle_hourse_price_train.csv')
# 打印前5行
data.head()

# 丢弃有缺失值的特征（列）
data.dropna(axis=1, inplace=True)

# 只保留整数的特征
data = data[[col for col in data.dtypes.index if data.dtypes[col] == 'int64']]
# 上下等价
idx_list = []
for col in data.dtypes.index:
    if data.dtypes[col] == 'int64':
        idx_list.append(col)
data = data[idx_list]

# 地块尺寸 未完成的地下室平方英尺 车库的面积
features = ['LotArea', 'BsmtUnfSF', 'GarageArea']
target = 'SalePrice'
data = data[features + [target]]
data.head()

from sklearn.utils import shuffle

data_shuffled = shuffle(data, random_state=32)  # 这个32不要改变

data_shuffled.head()

num_of_samples = data_shuffled.shape[0]
split_line = int(num_of_samples * 0.7)
train_data = data.iloc[:split_line]
test_data = data.iloc[split_line:]


def get_w(x, y):
    '''
    这个函数是计算模型w的值的函数，
    传入的参数分别是x和y，表示数据与标记

    Parameter
    ----------
        x: np.ndarray，pd.Series，传入的特征数据

        y: np.ndarray, pd.Series，对应的标记

    Returns
    ----------
        w: float, 模型w的值
    '''

    # m表示样本的数量
    m = x.shape[0]

    # 求x的均值
    x_mean = np.mean(x)

    # 求w的分子部分
    numerator = np.dot(y, (x - x_mean).T)

    # 求w的分母部分
    denominator = np.dot(x, x.T) - (1 / m) * np.square(np.sum(x))

    # 求w
    w = numerator * 1.0 / denominator

    # 返回w
    return w


def get_b(x, y, w):
    '''
    这个函数是计算模型b的值的函数，
    传入的参数分别是x, y, w，表示数据，标记以及模型的w值

    Parameter
    ----------
        x: np.ndarray，pd.Series，传入的特征数据

        y: np.ndarray, pd.Series，对应的标记

        w: np.ndarray, pd.Series，模型w的值

    Returns
    ----------
        b: float, 模型b的值
    '''
    # 样本个数
    m = x.shape[0]

    # 求b
    b = (1 / m) * np.sum(y - w * x)

    # 返回b
    return b


class myLinearRegression:
    def __init__(self):
        '''
        类的初始化方法，不需要初始化的参数
        这里设置了两个成员变量，用来存储模型w和b的值
        '''
        self.w = None
        self.b = None

    def fit(self, x, y):
        '''
        这里需要编写训练的函数，也就是调用模型的fit方法，传入特征x的数据和标记y的数据
        这个方法就可以求解出w和b
        '''
        self.w = get_w(x, y)
        self.b = get_b(x, y, self.w)

    def predict(self, x):
        '''
        这是预测的函数，传入特征的数据，返回模型预测的结果
        '''
        if self.w == None or self.b == None:
            print("模型还未训练，请先调用fit方法训练")
            return

        return self.w * x + self.b


# 创建一个模型的实例
model1 = myLinearRegression()

# 使用训练集对模型进行训练，传入训练集的LotArea和标记SalePrice
model1.fit(train_data['LotArea'], train_data['SalePrice'])

# 对测试集进行预测，并将结果存储在变量prediction中
prediction1 = model1.predict(test_data['LotArea'])


def MAE(y_hat, y):
    # 请你完成MAE的计算过程
    # YOUR CODE HERE
    m = y.shape[0]
    mae = np.sum(np.abs(y_hat - y)) / m

    return mae


def RMSE(y_hat, y):
    # 请你完成RMSE的计算过程
    # YOUR CODE HERE
    m = y.shape[0]
    rmse = np.sqrt(np.dot(y_hat - y, (y_hat - y).T) / m)

    return rmse


mae1 = MAE(prediction1, test_data['SalePrice'])
rmse1 = RMSE(prediction1, test_data['SalePrice'])
print("模型1，特征：LotArea")
print("MAE:", mae1)
print("RMSE:", rmse1)

import matplotlib.pyplot as plt

# 创建新的图
plt.figure(figsize=(16, 6))

# 创建子图1
plt.subplot(121)  # 121分别代表，生成1行2列个图，这是第1个
# 其中的参数为横轴值，纵轴值，label为此条曲线的标签， '.' 表示画出的图的图形为散点图
plt.plot(train_data['LotArea'].values, train_data['SalePrice'].values, '.', label='training data')
# '-' 表示画出的图形为折线图
plt.plot(train_data['LotArea'].values, model1.predict(train_data['LotArea']), '-', label='prediction')
plt.xlabel("LotArea")
plt.ylabel('SalePrice')
plt.title("training set")
plt.legend()

# 创建子图2
plt.subplot(122)  # 121分别代表，生成1行2列个图，这是第2个
plt.plot(test_data['LotArea'].values, test_data['SalePrice'].values, '.', label='testing data')
plt.plot(test_data['LotArea'].values, prediction1, '-', label='prediction')
plt.xlabel("LotArea")
plt.ylabel('SalePrice')
plt.title("testing set")
plt.legend()

plt.show()

#################
# 创建一个模型的实例
model2 = myLinearRegression()

# 使用训练集对模型进行训练，传入训练集的BsmtUnfSF和标记SalePrice
model2.fit(train_data['BsmtUnfSF'], train_data['SalePrice'])

# 对测试集进行预测，并将结果存储在变量prediction中
prediction2 = model2.predict(test_data['BsmtUnfSF'])

mae2 = MAE(prediction2, test_data['SalePrice'])
rmse2 = RMSE(prediction2, test_data['SalePrice'])
print("模型2，特征：BsmtUnfSF")
print("MAE2:", mae2)
print("RMSE2:", rmse2)

import matplotlib.pyplot as plt

# 创建新的图
plt.figure(figsize=(16, 6))

# 创建子图1
plt.subplot(121)  # 121分别代表，生成1行2列个图，这是第1个
# 其中的参数为横轴值，纵轴值，label为此条曲线的标签， '.' 表示画出的图的图形为散点图
plt.plot(train_data['BsmtUnfSF'].values, train_data['SalePrice'].values, '.', label='training data')
# '-' 表示画出的图形为折线图
plt.plot(train_data['BsmtUnfSF'].values, model2.predict(train_data['BsmtUnfSF']), '-', label='prediction')
plt.xlabel("BsmtUnfSF")
plt.ylabel('SalePrice')
plt.title("training set")
plt.legend()

# 创建子图2
plt.subplot(122)  # 121分别代表，生成1行2列个图，这是第2个
plt.plot(test_data['BsmtUnfSF'].values, test_data['SalePrice'].values, '.', label='testing data')
plt.plot(test_data['BsmtUnfSF'].values, prediction2, '-', label='prediction')
plt.xlabel('BsmtUnfSF')
plt.ylabel('SalePrice')
plt.title("testing set")
plt.legend()

# YOUR CODE HERE
model3 = myLinearRegression()
model3.fit(train_data['GarageArea'], train_data['SalePrice'])
prediction3 = model3.predict(test_data['GarageArea'])
mae3 = MAE(prediction3, test_data['SalePrice'])
rmse3 = RMSE(prediction3, test_data['SalePrice'])
print('模型3，特征：GarageArea')
print('MAE:', mae3)
print('RMSE:', rmse3)

plt.figure(figsize=(16, 6))
plt.subplot(121)
plt.plot(train_data['GarageArea'].values, train_data['SalePrice'].values, '.', label='train data')
plt.plot(train_data['GarageArea'].values, model3.predict(train_data['GarageArea']), '-', label='prediction')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.title('train set')
plt.legend()

plt.subplot(122)
plt.plot(test_data['GarageArea'].values, test_data['SalePrice'].values, '.', label='testing data')
plt.plot(test_data['GarageArea'].values, prediction3, '-', label='prediction')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.title('testing set')
plt.legend()


# YOUR CODE HERE
train_data = train_data[(train_data['LotArea'] < 60000) & (train_data['LotArea'] > 0)] # 将训练集中LotArea小于60000的值存入t
train_data = train_data[train_data['SalePrice'] < 500000] # 将t中SalePrice小于500000的值保留

# 绘制处理后的数据
plt.figure(figsize = (8, 7))
plt.plot(train_data['LotArea'], train_data['SalePrice'], '.')

# 创建一个模型的实例
model1 = myLinearRegression()

# 使用训练集对模型进行训练，传入训练集的LotArea和标记SalePrice
model1.fit(train_data['LotArea'], train_data['SalePrice'])

# 对测试集进行预测，并将结果存储在变量prediction中
prediction1 = model1.predict(test_data['LotArea'])

mae1 = MAE(prediction1, test_data['SalePrice'])
rmse1 = RMSE(prediction1, test_data['SalePrice'])
print("模型1，特征：LotArea")
print("MAE:", mae1)
print("RMSE:", rmse1)

import matplotlib.pyplot as plt

# 创建新的图
plt.figure(figsize=(16, 6))

# 创建子图1
plt.subplot(121)  # 121分别代表，生成1行2列个图，这是第1个
# 其中的参数为横轴值，纵轴值，label为此条曲线的标签， '.' 表示画出的图的图形为散点图
plt.plot(train_data['LotArea'].values, train_data['SalePrice'].values, '.', label='training data')
# '-' 表示画出的图形为折线图
plt.plot(train_data['LotArea'].values, model1.predict(train_data['LotArea']), '-', label='prediction')
plt.xlabel("LotArea")
plt.ylabel('SalePrice')
plt.title("training set")
plt.legend()

# 创建子图2
plt.subplot(122)  # 121分别代表，生成1行2列个图，这是第2个
plt.plot(test_data['LotArea'].values, test_data['SalePrice'].values, '.', label='testing data')
plt.plot(test_data['LotArea'].values, prediction1, '-', label='prediction')
plt.xlabel("LotArea")
plt.ylabel('SalePrice')
plt.title("testing set")
plt.legend()

plt.show()

#################
# 创建一个模型的实例
model2 = myLinearRegression()

# 使用训练集对模型进行训练，传入训练集的BsmtUnfSF和标记SalePrice
model2.fit(train_data['BsmtUnfSF'], train_data['SalePrice'])

# 对测试集进行预测，并将结果存储在变量prediction中
prediction2 = model2.predict(test_data['BsmtUnfSF'])

mae2 = MAE(prediction2, test_data['SalePrice'])
rmse2 = RMSE(prediction2, test_data['SalePrice'])
print("模型2，特征：BsmtUnfSF")
print("MAE2:", mae2)
print("RMSE2:", rmse2)

import matplotlib.pyplot as plt

# 创建新的图
plt.figure(figsize=(16, 6))

# 创建子图1
plt.subplot(121)  # 121分别代表，生成1行2列个图，这是第1个
# 其中的参数为横轴值，纵轴值，label为此条曲线的标签， '.' 表示画出的图的图形为散点图
plt.plot(train_data['BsmtUnfSF'].values, train_data['SalePrice'].values, '.', label='training data')
# '-' 表示画出的图形为折线图
plt.plot(train_data['BsmtUnfSF'].values, model2.predict(train_data['BsmtUnfSF']), '-', label='prediction')
plt.xlabel("BsmtUnfSF")
plt.ylabel('SalePrice')
plt.title("training set")
plt.legend()

# 创建子图2
plt.subplot(122)  # 121分别代表，生成1行2列个图，这是第2个
plt.plot(test_data['BsmtUnfSF'].values, test_data['SalePrice'].values, '.', label='testing data')
plt.plot(test_data['BsmtUnfSF'].values, prediction2, '-', label='prediction')
plt.xlabel('BsmtUnfSF')
plt.ylabel('SalePrice')
plt.title("testing set")
plt.legend()

# YOUR CODE HERE
model3 = myLinearRegression()
model3.fit(train_data['GarageArea'], train_data['SalePrice'])
prediction3 = model3.predict(test_data['GarageArea'])
mae3 = MAE(prediction3, test_data['SalePrice'])
rmse3 = RMSE(prediction3, test_data['SalePrice'])
print('模型3，特征：GarageArea')
print('MAE:', mae3)
print('RMSE:', rmse3)

plt.figure(figsize=(16, 6))
plt.subplot(121)
plt.plot(train_data['GarageArea'].values, train_data['SalePrice'].values, '.', label='train data')
plt.plot(train_data['GarageArea'].values, model3.predict(train_data['GarageArea']), '-', label='prediction')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.title('train set')
plt.legend()

plt.subplot(122)
plt.plot(test_data['GarageArea'].values, test_data['SalePrice'].values, '.', label='testing data')
plt.plot(test_data['GarageArea'].values, prediction3, '-', label='prediction')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.title('testing set')
plt.legend()
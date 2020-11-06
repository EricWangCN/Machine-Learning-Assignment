import matplotlib.pyplot as plt
from time import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(load_digits()['data'], load_digits()['target'], test_size=0.4,
                                                random_state=32)

from sklearn.preprocessing import StandardScaler

s = StandardScaler()
trainX = s.fit_transform(trainX)
testX = s.transform(testX)

trainY_mat = np.zeros((len(trainY), 10))
trainY_mat[np.arange(0, len(trainY), 1), trainY] = 1

testY_mat = np.zeros((len(testY), 10))
testY_mat[np.arange(0, len(testY), 1), testY] = 1


def initialize(h, K):
    '''
    参数初始化

    Parameters
    ----------
    h: int: 隐藏层单元个数

    K: int: 输出层单元个数

    Returns
    ----------
    parameters: dict，参数，键是"W1", "b1", "W2", "b2"

    '''
    np.random.seed(32)
    W_1 = np.random.normal(size=(trainX.shape[1], h)) * 0.01
    b_1 = np.zeros((1, h))

    np.random.seed(32)
    W_2 = np.random.normal(size=(h, K)) * 0.01
    b_2 = np.zeros((1, K))

    parameters = {'W1': W_1, 'b1': b_1, 'W2': W_2, 'b2': b_2}

    return parameters


# 测试样例
parameterst = initialize(50, 10)
print(parameterst['W1'].shape)  # (64, 50)
print(parameterst['b1'].shape)  # (1, 50)
print(parameterst['W2'].shape)  # (50, 10)
print(parameterst['b2'].shape)  # (1, 10)


def linear_combination(X, W, b):
    '''
    计算Z，Z = XW + b

    Parameters
    ----------
    X: np.ndarray, shape = (n, m)，输入的数据

    W: np.ndarray, shape = (m, h)，权重

    b: np.ndarray, shape = (1, h)，偏置

    Returns
    ----------
    Z: np.ndarray, shape = (n, h)，线性组合后的值

    '''

    # Z = XW + b
    # YOUR CODE HERE
    Z = np.dot(X, W) + np.ones((X.shape[0], 1)) * b

    return Z


# 测试样例
parameterst = initialize(50, 10)
Zt = linear_combination(trainX, parameterst['W1'], parameterst['b1'])
print(Zt.shape)  # (1078, 50)
print(Zt.mean())  # -5.27304442123e-19


def ReLU(X):
    '''
    ReLU激活函数

    Parameters
    ----------
    X: np.ndarray，待激活的矩阵

    Returns
    ----------
    activations: np.ndarray, 激活后的矩阵

    '''

    # YOUR CODE HERE
    X[X < 0] = 0

    return X


# 测试样例
parameterst = initialize(50, 10)
Zt = linear_combination(trainX, parameterst['W1'], parameterst['b1'])
Ht = ReLU(Zt)
print(Ht.mean())  # 0.0304

Ot = linear_combination(Ht, parameterst['W2'], parameterst['b2'])
print(Ot.shape)  # (1078, 10)
print(Ot.mean())  # 0.0006


def my_softmax(O):
    '''
    softmax激活
    '''
    # YOUR CODE HERE
    sum = np.sum(np.exp(O))
    O1 = np.exp(O) / sum
    return O1


# 测试样例1
# print(my_softmax(np.array([[0.3, 0.3, 0.3]])))  # array([[ 0.33333333,  0.33333333,  0.33333333]])

# 测试样例2
# test1 = np.array([[-1e32, -1e32, -1e32]])
# test2 = np.array([[1e32, 1e32, 1e32]])
# print(my_softmax(test1))
# print(my_softmax(test2))


def softmax(O):
    '''
    softmax激活函数

    Parameters
    ----------
    O: np.ndarray，待激活的矩阵

    Returns
    ----------
    activations: np.ndarray, 激活后的矩阵

    '''

    # YOUR CODE HERE
    length = O.shape[0]
    activations = my_softmax(O - np.ones(O.shape) * np.max(O))

    for i in range(length):
        activations[i] = my_softmax(O[i] - np.ones(O.shape[1]) * np.max(O[i]))

    return activations


# 测试样例
parameterst = initialize(50, 10)
Zt = linear_combination(trainX, parameterst['W1'], parameterst['b1'])
Ht = ReLU(Zt)
Ot = linear_combination(Ht, parameterst['W2'], parameterst['b2'])
y_pred = softmax(Ot)

print(y_pred.shape)  # (1078, 10)
print(Ot.mean())  # 0.000600192658464
print(y_pred.mean())  # 0.1

softmax(np.array([[1e32, 0, -1e32]]))


def log_softmax(x):
    '''
    log softmax

    Parameters
    ----------
    x: np.ndarray，待激活的矩阵

    Returns
    ----------
    log_activations: np.ndarray, 激活后取了对数的矩阵

    '''
    # 获取每行的最大值
    max_ = np.max(x, axis=1, keepdims=True)

    # 指数运算
    exp_x = np.exp(x - max_)

    # 每行求和
    Z = np.sum(exp_x, axis=1, keepdims=True)

    # 求log softmax
    log_activations = x - max_ - np.log(Z)

    return log_activations


# 测试样例
parameterst = initialize(50, 10)
Zt = linear_combination(trainX, parameterst['W1'], parameterst['b1'])
Ht = ReLU(Zt)
Ot = linear_combination(Ht, parameterst['W2'], parameterst['b2'])
t = log_softmax(Ot)
print(t.shape)  # (1078, 10)
print(t.mean())  # -2.30259148717


def cross_entropy_with_softmax(y_true, O):
    '''
    求解交叉熵损失函数，这里需要使用log softmax，所以参数分别是真值和未经softmax激活的输出值

    Parameters
    ----------
    y_true: np.ndarray，shape = (n, K), 真值

    O: np.ndarray, shape = (n, K)，softmax激活前的输出层的输出值

    Returns
    ----------
    loss: float, 平均的交叉熵损失值

    '''

    # 平均交叉熵损失
    # YOUR CODE HERE
    loss = - np.sum(log_softmax(O) * y_true) / len(y_true)

    return loss


parameterst = initialize(50, 10)
Zt = linear_combination(trainX, parameterst['W1'], parameterst['b1'])
Ht = ReLU(Zt)
Ot = linear_combination(Ht, parameterst['W2'], parameterst['b2'])
losst = cross_entropy_with_softmax(trainY_mat, Ot)
print(losst.mean())  # 2.30266707958


def forward(X, parameters):
    '''
    前向传播，从输入一直到输出层softmax激活前的值

    Parameters
    ----------
    X: np.ndarray, shape = (n, m)，输入的数据

    parameters: dict，参数

    Returns
    ----------
    O: np.ndarray, shape = (n, K)，softmax激活前的输出层的输出值

    '''
    # 输入层到隐藏层
    Z = linear_combination(X, parameters['W1'], parameters['b1'])

    # 隐藏层的激活
    H = ReLU(Z)

    # 隐藏层到输出层
    O = linear_combination(H, parameters['W2'], parameters['b2'])

    return O

# 测试样例
parameterst = initialize(50, 10)
Ot = forward(trainX, parameterst)
print(Ot.mean()) # 0.000600192658464


def compute_gradient(y_true, y_pred, H, Z, X, parameters):
    '''
    计算梯度

    Parameters
    ----------
    y_true: np.ndarray，shape = (n, K), 真值

    y_pred: np.ndarray, shape = (n, K)，softmax激活后的输出层的输出值

    H: np.ndarray, shape = (n, h)，隐藏层激活后的值

    Z: np.ndarray, shape = (n, h), 隐藏层激活前的值

    X: np.ndarray, shape = (n, m)，输入的原始数据

    parameters: dict，参数

    Returns
    ----------
    grads: dict, 梯度

    '''

    # 计算W2的梯度
    dW2 = np.dot(H.T, (y_pred - y_true)) / len(y_pred)

    # 计算b2的梯度
    db2 = np.sum(y_pred - y_true, axis=0) / len(y_pred)

    # 计算ReLU的梯度
    relu_grad = Z.copy()
    relu_grad[relu_grad < 0] = 0
    relu_grad[relu_grad >= 0] = 1

    # 计算W1的梯度
    dW1 = np.dot(X.T, np.dot(y_pred - y_true, parameters['W2'].T) * relu_grad) / len(y_pred)

    # 计算b1的梯度
    db1 = np.sum((np.dot(y_pred - y_true, parameters['W2'].T) * relu_grad), axis=0) / len(y_pred)

    grads = {'dW2': dW2, 'db2': db2, 'dW1': dW1, 'db1': db1}

    return grads

# 测试样例
parameterst = initialize(50, 10)

Zt = linear_combination(trainX, parameterst['W1'], parameterst['b1'])
Ht = ReLU(Zt)
Ot = linear_combination(Ht, parameterst['W2'], parameterst['b2'])
y_predt = softmax(Ot)

gradst = compute_gradient(trainY_mat, y_predt, Ht, Zt, trainX, parameterst)

print(gradst['dW1'].sum()) # 0.0429186117668
print(gradst['db1'].sum()) # -5.05985151857e-05
print(gradst['dW2'].sum()) # -2.16840434497e-18
print(gradst['db2'].sum()) # -1.34441069388e-17


def update(parameters, grads, learning_rate):
    '''
    参数更新

    Parameters
    ----------
    parameters: dict，参数

    grads: dict, 梯度

    learning_rate: float, 学习率

    '''
    parameters['W2'] -= learning_rate * grads['dW2']
    parameters['b2'] -= learning_rate * grads['db2']
    parameters['W1'] -= learning_rate * grads['dW1']
    parameters['b1'] -= learning_rate * grads['db1']

# 测试样例
parameterst = initialize(50, 10)
print(parameterst['W1'].sum())  # 0.583495454481
print(parameterst['b1'].sum())  # 0.0
print(parameterst['W2'].sum())  # 0.1888716431
print(parameterst['b2'].sum())  # 0.0
print()

Zt = linear_combination(trainX, parameterst['W1'], parameterst['b1'])
Ht = ReLU(Zt)
Ot = linear_combination(Ht, parameterst['W2'], parameterst['b2'])
y_predt = softmax(Ot)

gradst = compute_gradient(trainY_mat, y_predt, Ht, Zt, trainX, parameterst)
update(parameterst, gradst, 0.1)

print(parameterst['W1'].sum())  # 0.579203593304
print(parameterst['b1'].sum())  # 5.05985151857e-06
print(parameterst['W2'].sum())  # 0.1888716431
print(parameterst['b2'].sum())  # 1.24683249836e-18


def backward(y_true, y_pred, H, Z, X, parameters, learning_rate):
    '''
    计算梯度，参数更新

    Parameters
    ----------
    y_true: np.ndarray，shape = (n, K), 真值

    y_pred: np.ndarray, shape = (n, K)，softmax激活后的输出层的输出值

    H: np.ndarray, shape = (n, h)，隐藏层激活后的值

    Z: np.ndarray, shape = (n, h), 隐藏层激活前的值

    X: np.ndarray, shape = (n, m)，输入的原始数据

    parameters: dict，参数

    learning_rate: float, 学习率

    '''
    grads = compute_gradient(y_true, y_pred, H, Z, X, parameters)
    update(parameters, grads, learning_rate)

# 测试样例
parameterst = initialize(50, 10)
print(parameterst['W1'].sum())  # 0.583495454481
print(parameterst['b1'].sum())  # 0.0
print(parameterst['W2'].sum())  # 0.1888716431
print(parameterst['b2'].sum())  # 0.0
print()

Zt = linear_combination(trainX, parameterst['W1'], parameterst['b1'])
Ht = ReLU(Zt)
Ot = linear_combination(Ht, parameterst['W2'], parameterst['b2'])
y_predt = softmax(Ot)

backward(trainY_mat, y_predt, Ht, Zt, trainX, parameterst, 0.1)

print(parameterst['W1'].sum())  # 0.579203593304
print(parameterst['b1'].sum())  # 5.05985151857e-06
print(parameterst['W2'].sum())  # 0.1888716431
print(parameterst['b2'].sum())  # 1.24683249836e-18


def train(trainX, trainY, testX, testY, parameters, epochs, learning_rate=0.01, verbose=False):
    '''
    训练

    Parameters
    ----------
    Parameters
    ----------
    trainX: np.ndarray, shape = (n, m), 训练集

    trainY: np.ndarray, shape = (n, K), 训练集标记

    testX: np.ndarray, shape = (n_test, m)，测试集

    testY: np.ndarray, shape = (n_test, K)，测试集的标记

    parameters: dict，参数

    epochs: int, 要迭代的轮数

    learning_rate: float, default 0.01，学习率

    verbose: boolean, default False，是否打印损失值

    Returns
    ----------
    training_loss_list: list(float)，每迭代一次之后，训练集上的损失值

    testing_loss_list: list(float)，每迭代一次之后，测试集上的损失值

    '''
    # 存储损失值
    training_loss_list = []
    testing_loss_list = []

    for i in range(epochs):

        # 这里要计算出Z和H，因为后面反向传播计算梯度的时候需要这两个矩阵
        Z = linear_combination(trainX, parameters['W1'], parameters['b1'])
        H = ReLU(Z)
        train_O = linear_combination(H, parameters['W2'], parameters['b2'])
        train_y_pred = softmax(train_O)
        training_loss = cross_entropy_with_softmax(trainY, train_O)

        test_O = forward(testX, parameters)
        testing_loss = cross_entropy_with_softmax(testY, test_O)

        if verbose == True:
            print('epoch %s, training loss:%s' % (i + 1, training_loss))
            print('epoch %s, testing loss:%s' % (i + 1, testing_loss))
            print()

        training_loss_list.append(training_loss)
        testing_loss_list.append(testing_loss)

        backward(trainY, train_y_pred, H, Z, trainX, parameters, learning_rate)
    return training_loss_list, testing_loss_list

# 测试样例
parameterst = initialize(50, 10)
print(parameterst['W1'].sum())  # 0.583495454481
print(parameterst['b1'].sum())  # 0.0
print(parameterst['W2'].sum())  # 0.1888716431
print(parameterst['b2'].sum())  # 0.0
print()

training_loss_list, testing_loss_list = train(trainX, trainY_mat, testX, testY_mat, parameterst, 1, 0.1, False)

print(parameterst['W1'].sum())  # 0.579203593304
print(parameterst['b1'].sum())  # 5.05985151857e-06
print(parameterst['W2'].sum())  # 0.1888716431
print(parameterst['b2'].sum())  # 1.24683249836e-18


def plot_loss_curve(training_loss_list, testing_loss_list):
    '''
    绘制损失值变化曲线

    Parameters
    ----------
    training_loss_list: list(float)，每迭代一次之后，训练集上的损失值

    testing_loss_list: list(float)，每迭代一次之后，测试集上的损失值

    '''
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss_list, label='training loss')
    plt.plot(testing_loss_list, label='testing loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def predict(X, parameters):
    '''
    预测，调用forward函数完成神经网络对输入X的计算，然后完成类别的划分，取每行最大的那个数的下标作为标记

    Parameters
    ----------
    X: np.ndarray, shape = (n, m), 训练集

    parameters: dict，参数

    Returns
    ----------
    prediction: np.ndarray, shape = (n, 1)，预测的标记

    '''
    # 用forward函数得到softmax激活前的值
    O = forward(X, parameters)

    # 计算softmax激活后的值
    y_pred = softmax(O)

    # 取每行最大的元素对应的下标
    prediction = np.argmax(y_pred, axis=1)

    return prediction

start_time = time()

h = 50
K = 10
parameters = initialize(h, K)
training_loss_list, testing_loss_list = train(trainX, trainY_mat, testX, testY_mat, parameters, 1000, 0.03, False)

end_time = time()
print('training time: %s s'%(end_time - start_time))

prediction = predict(testX, parameters)
accuracy_score(prediction, testY)

plot_loss_curve(training_loss_list, testing_loss_list)
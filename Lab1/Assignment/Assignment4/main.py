import numpy as np

spambase = np.loadtxt('/Users/wangzilong/Desktop/Study/Assignment/机器学习/Machine-Learning-Assignment/Lab1/Assignment/data/spambase/spambase.data', delimiter=",")
dota2results = np.loadtxt('/Users/wangzilong/Desktop/Study/Assignment/机器学习/Machine-Learning-Assignment/Lab1/Assignment/data/dota2Dataset/dota2Train.csv', delimiter=',')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

spamx = spambase[:, :57]
spamy = spambase[:, 57]

dota2x = dota2results[:, 1:]
dota2y = dota2results[:, 0]

spambaseModel = LinearDiscriminantAnalysis()
spambasePrediction = cross_val_predict(spambaseModel, spamx, spamy, cv=10)
dota2Model = LinearDiscriminantAnalysis()
dota2Prediction = cross_val_predict(dota2Model, dota2x, dota2y, cv=10)

spamyAcc = accuracy_score(spamy, spambasePrediction)
spamyPre = precision_score(spamy, spambasePrediction)
spamyRecall = recall_score(spamy, spambasePrediction)
spamyF1 = f1_score(spamy, spambasePrediction)

dota2Acc = accuracy_score(dota2y, dota2Prediction)
dota2Pre = precision_score(dota2y, dota2Prediction)
dota2Recall = recall_score(dota2y, dota2Prediction)
dota2F1 = f1_score(dota2y, dota2Prediction)

# YOUR CODE HERE
print("======= Spambase ======")
print("Accuracy: ", spamyAcc)
print("Precision:", spamyPre)
print("Recall:   ", spamyRecall)
print("F1"        , spamyF1)
print("=======  Dota2  ======")
print("Accuracy: ", dota2Acc)
print("Precision:", dota2Pre)
print("Recall:   ", dota2Recall)
print("F1"        , dota2F1)

'''
解题流程
'''
import os
import csv
import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

data_dir = './data/'


# 加载数据
def opencsv():
    data_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    data_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    train_data = data_train.values[0:, 1:]   # 读入全部训练数据， [行，列]
    train_label = data_train.values[0:, 0]   # 读取列表的第一列
    test_data = data_test.values[0:, 0:]     # 测试全部测试数据
    return train_data, train_label, test_data

# 加载数据
trainData, trainLabel, testData = opencsv()

# 以下开始特征工程
# 数据预处理-降维  PCA主成分分析
def dRPCA(x_train, x_test, COMPONENT_NUM):
    print('dimensionality reduction...')
    trainData = np.array(x_train)
    testData = np.array(x_test)
    '''
    使用说明：https://www.cnblogs.com/pinard/p/6243025.html
    n_components>=1
      n_components=NUM   设置占特征数量比
    0 < n_components < 1
      n_components=0.99  设置阈值总方差占比
    '''
    pca = PCA(n_components=COMPONENT_NUM, whiten=False)
    pca.fit(trainData)    # 用训练数据来拟合模型
    pcaTrainData = pca.transform(trainData)    # 在trainData上完成降维
    pcaTestData = pca.transform(testData)      # 在testData上完成降维

    # pca 方差大小、方差占比、 特征数量
    # print("方差大小:\n", pca.explained_variance_, "方差占比:\n", pca.explained_variance_ratio_)
    print("特征数量: %s" % pca.n_components_)
    print("总方差占比: %s" % sum(pca.explained_variance_ratio_))
    return pcaTrainData, pcaTestData


# 降维处理
trainDataPCA, testDataPCA = dRPCA(trainData, testData, 0.8)

'''
模型选择
    常用算法： knn、决策树、朴素贝叶斯、Logistic回归、SVM、集成方法（随机森林和AdaBoost）
'''
# 采用KNN
# def trainModel(trainData, trainLabel):
#     clf = KNeighborsClassifier()   # default: k=5, defined by yourself: KNeighborsClassifier(n_neighbors=10)
#     clf.fit(trainData, np.ravel(trainLabel))
#     return clf
# # 模型训练
# clf = trainModel(trainDataPCA, trainLabel)
# # 结果预测
# testLabel = clf.predict(testDataPCA)


# SVM  训练模型
def trainModel(trainData, trainLabel):
    print('Train SVM...')
    clf = SVC(C=4, kernel='rbf')
    clf.fit(trainData, trainLabel)  # 训练SVM
    return clf
# 模型训练
clf = trainModel(trainDataPCA, trainLabel)
# 结果预测
testLabel = clf.predict(testDataPCA)

# RF ---   Random Forest
# 训练模型
# def trainModel(X_train, y_train):
#     print('Train RF...')
#     clf = RandomForestClassifier(
#         n_estimators=10,
#         max_depth=10,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         random_state=34)
#     clf.fit(X_train, y_train)  # 训练rf
#     return clf
# # 模型训练
# clf = trainModel(trainDataPCA, trainLabel)
# # 结果预测
# testLabel = clf.predict(testDataPCA)
print('开始打印')

# 结果导出
def saveResult(result, csvName):
    with open(csvName, 'w', newline='') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(["ImageId", "Label"])
        index = 0
        # tmp = []
        for i in result:
            tmp = []
            index = index+1
            tmp.append(index)
            # tmp.append(i)
            tmp.append(int(i))
            myWriter.writerow(tmp)
# 结果输出
saveResult(testLabel, 'data/Result_SVM.csv')



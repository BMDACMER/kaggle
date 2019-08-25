from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
import pandas as pd
import time
import csv
import lightgbm as lgb
import numpy as np

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)  # 列对齐  testData的Label列 补NaN
data.drop(['label'], axis=1, inplace=True)
label = train_data.label

# 特征选择
pca = PCA(n_components=100, random_state=34)  # 定义PCA模型
data_pca = pca.fit_transform(data)            # 将所有数据传入

Xtrain,  xtest, Ytrain, ytest = train_test_split(data_pca[0:len(train_data)], label, test_size=0.1, random_state=34)

# 模型选择
clf = RandomForestClassifier(n_estimators=110, max_depth=5, min_samples_leaf=1, min_samples_split=2, random_state=34)
# 将RandomForestClassifier换为LGBMClassifier
# clf=lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
#
# param_test1 = {'n_estimators':[10,150,10],'max_depth':[1,11,1]}
# gsearch1 = GridSearchCV(estimator=clf, param_grid=param_test1, scoring='accuracy',iid=False,cv=5)
# gsearch1.fit(Xtrain,Ytrain)
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
# y_predict = gsearch1.predict(xtest)



clf.fit(Xtrain, Ytrain)
y_predict = clf.predict(xtest)

zeroLabel = ytest - y_predict  # 判断预测是否正确
rightCount = 0
for i in range(len(zeroLabel)):
    if list(zeroLabel)[i] == 0:
        rightCount += 1
print('the right rate is:', float(rightCount)/len(zeroLabel))

#统计预测时间
start = time.time()
# 做最后结果预测
result = clf.predict(data_pca[len(train_data):])
# result = gsearch1.predict(data_pca[len(train_data):])
print("预测时间为",(time.time() - start))


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
saveResult(result, 'data/version2/Result_RF2.csv')

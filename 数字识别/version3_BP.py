from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import csv
import time

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
data = pd.concat([train_data, test_data], axis=0, sort=True).reset_index(drop=True)  # 列对齐  testData的Label列 补NaN
data.drop(['label'], axis=1, inplace=True)
label = train_data.label


pca=PCA(n_components=100, random_state=34)
data_pca=pca.fit_transform(data)

Xtrain,Ytrain,xtest,ytest=train_test_split(data_pca[0:len(train_data)],label,test_size=0.1, random_state=34)

clf=MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', alpha=0.0001,learning_rate='constant', learning_rate_init=0.001,max_iter=200, shuffle=True, random_state=34)


clf.fit(Xtrain,xtest)
y_predict=clf.predict(Ytrain)


zeroLable=ytest-y_predict
rightCount=0
for i in range(len(zeroLable)):
    if list(zeroLable)[i]==0:
        rightCount+=1
print ('the right rate is:',float(rightCount)/len(zeroLable))

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
saveResult(result, 'data/version2/Result_BP.csv')

'''
使用sklearn中的IRIS（鸢尾花）数据集来对特征处理功能进行说明

IRIS数据集由Fisher在1936年整理，包含4个特征（Sepal.Length（花萼长度）、Sepal.Width（花萼宽度）、
Petal.Length（花瓣长度）、Petal.Width（花瓣宽度）），特征值都为正浮点数，单位为厘米。目标值为鸢尾花的分类
（Iris Setosa（山鸢尾）、Iris Versicolour（杂色鸢尾），Iris Virginica（维吉尼亚鸢尾））

参考：    https://www.cnblogs.com/jasonfreak/p/5448385.html
'''

from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer, OneHotEncoder, Imputer

# 导入IRIS数据集
iris = load_iris()

# 特征选择
# 无量纲化
StandardScaler.fit_transform(iris.data)   # 减均值除以标准差
MinMaxScaler.fit_transform(iris.data)   # 区间缩放法 --- (x - Min) / (Max - Min)
Normalizer.fit_transform(iris.data)   # 归一化区别

# 对定量特征二值化
Binarizer(threshold=3).fit_transform(iris.data)
# 对定性特征编码
OneHotEncoder().fit_transform(iris.target.reshape((-1, 1)))
#缺失值计算，返回值为计算缺失值后的数据
#参数missing_value为缺失值的表示形式，默认为NaN
#参数strategy为缺失值填充方式，默认为mean（均值）
Imputer().fit_transform(np.vstack((np.array([np.nan, np.nan, np.nan, np.nan]), iris.data)))
# 目标向量
iris.target



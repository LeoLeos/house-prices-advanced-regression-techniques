# 导入需要的模块
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 用来绘图的，封装了matplot
# 要注意的是一旦导入了seaborn，
# matplotlib的默认作图风格就会被覆盖成seaborn的格式
import seaborn as sns

from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import linear_model, svm, gaussian_process
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


data_train = pd.read_csv("train.csv")

#预览数据头
#print(data_train)

#数据总体信息描述
#print(data_train['SalePrice'].describe())

#画出某一属性，可视化
sns.distplot(data_train['SalePrice'])
plt.show()

#print(data_train["SalePrice"].kurt())
# 6.53628186006
# 查看偏度
print(data_train["SalePrice"].skew())
# 1.8828757597

sns.distplot(np.log(data_train["SalePrice"]))
plt.show()

#选取LotArea分析
varName = "LotArea"
data_train.plot.scatter(x=varName, y="SalePrice", ylim=(0, 800000))
plt.show()

#选取GrLivArea分析
varName = "GrLivArea"
data_train.plot.scatter(x=varName, y="SalePrice", ylim=(0, 800000))
plt.show()

#选取TotalBsmtSF分析
varName = "TotalBsmtSF"
data_train.plot.scatter(x=varName, y="SalePrice", ylim=(0, 800000))
plt.show()

#选取MiscVal，GarageArea，GarageCars分析
varName = ["MiscVal", "GarageArea", "GarageCars"]
for i in range(len(varName)):
    data_train.plot.scatter(x=varName[i], y="SalePrice", ylim=(0, 800000))
plt.show()

#类别特征分析
#选取CentralAir分析
varName = "CentralAir"
fig = sns.boxplot(x=varName, y="SalePrice", data=data_train)
fig.axis(ymin=0, ymax=800000)
plt.show()

#分析OverallQual
varName = "OverallQual"
fig = sns.boxplot(x=varName, y="SalePrice", data=data_train)
fig.axis(ymin=0, ymax=800000)
plt.show()

#选取YearBuilt
varName = "YearBuilt"
plt.subplots(figsize=(30, 15))
fig = sns.boxplot(x=varName, y="SalePrice", data=data_train)
fig.axis(ymin=0, ymax=800000)
# 将x轴的年份垂直显示
plt.xticks(rotation=90)
plt.show()

#选取YearBuilt
varName = "YearBuilt"
data_train.plot.scatter(x=varName, y="SalePrice", ylim=(0, 800000))
plt.show()

#选取Neighborhood
varName = "Neighborhood"
plt.subplots(figsize=(30, 15))
fig = sns.boxplot(x=varName, y="SalePrice", data=data_train)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
plt.show()

#关系矩阵,判断两个变量的相关性程度
corrmat = data_train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

#使用sklearn对数据进行预处理
f_names = ['CentralAir', 'Neighborhood']
for x in f_names:
    label = preprocessing.LabelEncoder()
    data_train[x] = label.fit_transform(data_train[x])
corrmat = data_train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

#房价关系矩阵
k  = 10 # 关系矩阵中将显示10个特征
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#绘制关系点图
sns.set()
cols = ['SalePrice','OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.pairplot(data_train[cols], size = 2.5)
plt.show()

#sklearn机器学习
# 获取数据
cols = ['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
x = data_train[cols].values
y = data_train['SalePrice'].values
x_scaled = preprocessing.StandardScaler().fit_transform(x)
y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))
X_train,X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=42)

#分类器训练
clfs = {
        'svm':svm.SVR(),
        'RandomForestRegressor':RandomForestRegressor(n_estimators=400),
        'BayesianRidge':linear_model.BayesianRidge()
       }
for clf in clfs:
    try:
        clfs[clf].fit(X_train, y_train)
        y_pred = clfs[clf].predict(X_test)
        print(clf + " cost:" + str(np.sum(y_pred-y_test)/len(y_pred)) )
    except Exception as e:
        print(clf + " Error:")
        print(str(e))

#选用随机森林
cols = ['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
x = data_train[cols].values
y = data_train['SalePrice'].values
X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = RandomForestRegressor(n_estimators=400)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#输出预测值
print(y_pred)

# 保存clf，共下面计算测试集数据使用
# rfr = clf

#误差值
error = sum(abs(y_pred - y_test))/len(y_pred)

#加载之前训练好的模型
# 之前训练的模型
rfr = clf
data_test = pd.read_csv("test.csv")
#判断数据内部是否有Null值，如果有，则改为本属性的均值
print(data_test[cols].isnull().sum())
#查看GarageCars总体描述
print(data_test['GarageCars'].describe())
#TotalBsmtSF总体描述
print(data_test['TotalBsmtSF'].describe())

#填充Null值
# 不知道为什么fillna函数对data_test[cols]总是不起作用，所以只好用最笨的办法了
#data_test[ ['GarageCars'] ].fillna(1.766118, inplace=True)
#data_test[ ['TotalBsmtSF']].fillna(1046.117970, inplace=True)
#data_test[cols].fillna(data_test[cols].mean())
#data_test[cols].isnull().sum()

cols2 = ['OverallQual','GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
cars = data_test['GarageCars'].fillna(1.766118)
bsmt = data_test['TotalBsmtSF'].fillna(1046.117970)
data_test_x = pd.concat( [data_test[cols2], cars, bsmt] ,axis=1)
data_test_x.isnull().sum()

#预测测试集中的值
x = data_test_x.values
y_te_pred = rfr.predict(x)
print(y_te_pred)

print(y_te_pred.shape)
print(x.shape)

#格式化预测出来的房价值
prediction = pd.DataFrame(y_te_pred, columns=['SalePrice'])
#合并
result = pd.concat([ data_test['Id'], prediction], axis=1)
# result = result.drop(resultlt.columns[0], 1)
#result.columns

#保存预测结果
# 保存预测结果
result.to_csv('./Predictions.csv', index=False)
# Logistic_Regression
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读入数据
data = pd.read_csv("data_2_test.csv")

# 提取特征

# 菌群
feature_DecisionTree = ['sample', 'g__Lachnospiraceae_UCG-001', 'g__Victivallis', 'g__Paraprevotella',
                        'g__Lachnospiraceae_UCG-003']
feature_XGBoost = ['sample', 'g__Ruminiclostridium_9', 'g__Parasutterella', 'g__Lachnospiraceae_UCG-001', 'g__Blautia',
                   'g__Ruminococcaceae_UCG-003']
feature_Forest = ['sample', 'g__Abiotrophia', 'g__Acetanaerobacterium', 'g__Acidaminococcus', 'g__Acinetobacter',
                  'g__Actinomyces', 'g__Adlercreutzia', 'g__Akkermansia', 'g__Aliihoeflea']

# 代谢物
feature_DecisionTree_2 = ['sample', '3-Dehydrocholic acid', 'Hydrocinnamic acid', 'Glycolithocholic acid']
feature_XGBoost_2 = ['sample', 'Phthalic acid', '3-Dehydrocholic acid', 'Nicotinic acid']
feature_Forest_2 = ['sample', 'Deoxycholic acid', 'Hyodeoxycholic acid', 'L-Lysine', 'L-Histidine', 'L-Arginine',
                    'Ornithine', 'L-Glutamine', 'L-Glutamic acid', 'Sarcosine']

# 划分测试集与训练集
X = data[feature_Forest_2[1:]]
Y = data[feature_Forest_2[0]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=0)

# 标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)  # 先拟合数据再进行标准化

# 构建模型并训练
lr = LogisticRegressionCV(multi_class="ovr", fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, penalty="l2",
                          solver="lbfgs", tol=0.01)
re = lr.fit(X_train, Y_train)

# 模型效果获取
r = re.score(X_train, Y_train)
print("Accuracy", r)
print("参数：", re.coef_)
print("截距", re.intercept_)
print("稀疏化特征比率：%.2f%% " % (np.mean(lr.coef_.ravel() == 0) * 100))
print("=========sigmoid函数转化的值，即：概率p=========")
print(re.predict_proba(X_test))  # sigmoid函数的值 概率p
X_test = ss.fit_transform(X_test)  # 数据标准化
Y_predict = lr.predict(X_test)  # 预测

# 训练结果可视化
x = range(len(X_test))
plt.figure(figsize=(14, 7), facecolor='w')
plt.ylim(0, 6)
plt.plot(x, Y_test, "ro", markersize=8, zorder=3, label=u"real_value")
plt.plot(x, Y_predict, "go", markersize=14, zorder=2, label=u"predictive_value$R^2$=%.3f" % lr.score(X_test, Y_test))
plt.legend(loc="upper left")
plt.xlabel(u"data number", fontsize=18)
plt.ylabel(u"illness", fontsize=18)
plt.title(u"Logistic_Regression", fontsize=20)
plt.savefig("Logistic_Regression.png")
plt.show()
print("=============Y_test==============")
print(Y_test.ravel())
print("============Y_predict============")
print(Y_predict)

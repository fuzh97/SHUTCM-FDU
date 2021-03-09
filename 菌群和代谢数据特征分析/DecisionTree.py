import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import graphviz

# 读入数据
data = pd.read_csv("data_2.csv")

# 划分特征值目标值
x = data.iloc[:, data.columns != "sample"]
y = data.iloc[:, data.columns == "sample"]

# 划分训练集测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, test_size=0.3)

# 重新按顺序编号
for i in [x_train, x_test, y_train, y_test]:
    i.index = range(i.shape[0])
# print(y_train.index)

# 训练一次
clf = DecisionTreeClassifier(random_state=25
                             , max_depth=3
                             , criterion='gini'
                             , min_impurity_decrease=0.030612244897959183
                             # , min_samples_leaf=4
                             , splitter='best'
                             )
clf = clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print("训练一次的分数：%f" % score)

# 决策树可视化
feature_name = x.columns
dot_data = export_graphviz(clf
                           , feature_names=feature_name
                           , class_names=["2", "1", "0"]
                           , filled=True
                           , rounded=True
                           , out_file=None
                           )
graph = graphviz.Source(dot_data)
graph.view()

# 每个特征的重要性
temp = list(zip(feature_name, clf.feature_importances_))
ans = []
for tmp in temp:
    if tmp[1] > 0:
        ans.append(tmp)
ans.sort(key=lambda x: x[1], reverse=True)
plt.figure()
plt.barh([tmp[0] for tmp in ans], [tmp[1] for tmp in ans])
plt.show()
weight = "\n".join(str(i) for i in ans)
print(weight)

# 交叉验证
clf = DecisionTreeClassifier(random_state=25)
score = cross_val_score(clf, x, y.values.reshape(y.shape[0], ), cv=3).mean()
print("交叉验证的分数： %f" % score)

# 尝试不同的高度并画曲线
tr = []
te = []
for i in range(10):
    clf = DecisionTreeClassifier(random_state=25
                                 , max_depth=i + 1
                                 , criterion="entropy"
                                 )
    clf = clf.fit(x_train, y_train)
    score_tr = clf.score(x_train, y_train)
    score_te = cross_val_score(clf, x, y.values.reshape(y.shape[0], ), cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)
print("不同高度能取得的最大分数：%f" % max(te))
plt.plot(range(1, 11), tr, color="red", label="train")
plt.plot(range(1, 11), te, color="blue", label="test")
plt.xticks(range(1, 11))
plt.legend()
plt.show()

# 网格搜索
parameters = {
    "criterion": ("gini", "entropy")
    , "splitter": ("best", "random")
    , "max_depth": [*range(1, 10)]
    , "min_samples_leaf": [*range(1, 12, 3)]
    , "min_impurity_decrease": np.linspace(0, 0.5, 50)
}
clf = DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf, parameters, cv=2)
GS = GS.fit(x_train, y_train.values.reshape(y_train.shape[0], ))
print(GS.best_params_)
print(GS.best_score_)

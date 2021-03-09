import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 数据读入
data = pd.read_csv("data_2.csv")
x = data.iloc[:, data.columns != "sample"]
y = data.iloc[:, data.columns == "sample"]

# 训练
clf = RandomForestClassifier()
clf.fit(x, y)
importance = clf.feature_importances_  # 特征重要性
indices = np.argsort(importance)[::-1]
features = x.columns
for f in range(x.shape[1]):
    print(("%2d) %-*s %f" % (f + 1, 30, features[f], importance[indices[f]])))

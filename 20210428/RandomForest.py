import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from openpyxl import load_workbook

data = pd.read_csv("/data/data_2_total.csv.csv")
x = data.iloc[:, data.columns != "sample"]
y = data.iloc[:, data.columns == "sample"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
clf.fit(x_train, y_train)

importance = clf.feature_importances_  # 特征重要性
indices = np.argsort(importance)[::-1]
features = x_train.columns
print(clf.score(x_test, y_test))

for f in range(x_train.shape[1]):
    print(("%2d) %-*s %f" % (f + 1, 30, features[indices[f]], importance[indices[f]])))
# 保存到文件
'''
workbook = load_workbook(filename="Feature_Importance_data2_total.xlsx")
sheet = workbook.active
sheet.append(["特征", "Feature Importance"])
for f in range(x_train.shape[1]):
    print(("%2d) %-*s %f" % (f + 1, 30, features[indices[f]], importance[indices[f]])))
    sheet.append([features[indices[f]], importance[indices[f]]])
workbook.save(filename="Feature_Importance_data2_total.xlsx")
'''

import pandas as pd
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("G:\\project\\中医药大学\\2021.12.29\\data_1_33.csv")

feature = []
i = 1
path = "G:\\project\\中医药大学\\2021.12.29\\feature1.csv"
with open(path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row)
        if (row == []):
            break
        feature.append(row[0])
feature[0] = 'g__Tyzzerella'

print(feature)

data = df[feature]

print(data)

import pandas as pd
import numpy as np
from openpyxl import load_workbook
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("F:\\project\\中医药大学\\2021.12.29\\data\\data_1_65.csv")

# 65例中的27个特征进行选取
# 65例"['g__unclassified_o__Rhizobiales', 'g__Sphingomonas'] not in index"
feature_1 = ['sample', 'g__Tyzzerella', 'g__Eggerthella', 'g__Papillibacter', 'g__Pseudobutyrivibrio',
             'g__Terrisporobacter',
             'g__norank_o__NB1-n', 'g__[Clostridium]_innocuum_group', 'g__Mitsuokella',
             'g__Caproiciproducens', 'g__Coprococcus_3', 'g__Erysipelatoclostridium', 'g__Scardovia', 'g__Odoribacter',
             'g__Flavonifractor', 'g__Haemophilus', 'g__Veillonella',
             'g__Subdoligranulum', 'g__Erysipelotrichaceae_UCG-003', 'g__Ruminococcaceae_NK4A214_group',
             'g__norank_f__Bacteroidales_S24-7_group', 'g__norank_o__Mollicutes_RF9', 'g__[Eubacterium]_rectale_group',
             'g__Ruminococcaceae_UCG-002', 'g__Phascolarctobacterium', 'g__Paraprevotella',
             'g__unclassified_o__Rhizobiales', 'g__Sphingomonas']

# 65例"['g__unclassified_o__Rhizobiales', 'g__Sphingomonas'] not in index"


# 78例中的34个特征进行选取
feature_2 = ['sample',
             'L-Arginine',
             'Sarcosine',
             'L-Alanine',
             'Gamma-Aminobutyric acid',
             'L-Asparagine',
             'L-Valine',
             'Phenylpyruvic acid',
             '3-Aminosalicylic acid',
             'Lithocholic acid 3-sulfate',
             'Apocholic Acid',
             'D-Gluconolactone',
             'Carnitine',
             'Azelaic acid',
             'Myristic acid',
             'Pentadecanoic acid',
             'Palmitoleic acid',
             'Citramalic acid',
             'Heptadecanoic acid',
             'Oleic acid',
             'Indoleacrylic acid',
             '3-Indolepropionic acid',
             'Alpha-Ketoisovaleric acid',
             'Ketoleucine',
             '3-Methyl-2-oxovaleric acid',
             '4-Hydroxyphenylpyruvic acid',
             'p-Hydroxyphenylacetic acid',
             '2-Phenylpropionate',
             'Hydrocinnamic acid',
             'Phenyllactic acid',
             'Cinnamic acid',
             'Ethylmethylacetic acid',
             'Propanoic acid',
             'Isobutyric acid',
             'Isovaleric acid']

data = df[feature_1]

x = data.iloc[:, data.columns != "sample"]
y = data.iloc[:, data.columns == "sample"]

# 训练随机森林模型

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)  # 22.0.9375
clf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
clf.fit(x_train, y_train)

importance = clf.feature_importances_  # 特征重要性
indices = np.argsort(importance)[::-1]
features = x_train.columns
print(clf.score(x_test, y_test))

for f in range(x_train.shape[1]):
    print(("%2d) %-*s %f" % (f + 1, 30, features[indices[f]], importance[indices[f]])))

# 保存到文件
workbook = load_workbook(filename="Feature_Importance_78.xlsx")
sheet = workbook.active
sheet.append(["特征", "Feature Importance"])
for f in range(x_train.shape[1]):
    print(("%2d) %-*s %f" % (f + 1, 30, features[indices[f]], importance[indices[f]])))
    sheet.append([features[indices[f]], importance[indices[f]]])
workbook.save(filename="Feature_Importance_78.xlsx")

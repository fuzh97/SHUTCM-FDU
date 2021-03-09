import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv("corrleation.csv")
# 提取数据
# 肠道菌群
feature_DecisionTree = ['g__Lachnospiraceae_UCG-001', 'g__Victivallis', 'g__Paraprevotella',
                        'g__Lachnospiraceae_UCG-003']
feature_XGBoost = ['g__Ruminiclostridium_9', 'g__Parasutterella', 'g__Lachnospiraceae_UCG-001', 'g__Blautia',
                   'g__Ruminococcaceae_UCG-003']
feature_Forest = ['g__Abiotrophia', 'g__Acetanaerobacterium', 'g__Acidaminococcus', 'g__Acinetobacter',
                  'g__Actinomyces', 'g__Adlercreutzia', 'g__Akkermansia', 'g__Aliihoeflea']
# 代谢物
feature_DecisionTree_2 = ['3-Dehydrocholic acid', 'Hydrocinnamic acid', 'Glycolithocholic acid']
feature_XGBoost_2 = ['Phthalic acid', '3-Dehydrocholic acid', 'Nicotinic acid']
feature_Forest_2 = ['Deoxycholic acid', 'Hyodeoxycholic acid', 'L-Lysine', 'L-Histidine', 'L-Arginine',
                    'Ornithine', 'L-Glutamine', 'L-Glutamic acid', 'Sarcosine']

# 计算相关性 pearson相关系数矩阵
a = feature_Forest
b = feature_Forest_2
new_list = a + b
corr1 = data[new_list].corr(method="pearson")
corr2 = data[new_list].corr(method="spearman")
corr3 = data[new_list].corr(method="kendall")

# 绘制热力图
corr = corr3.iloc[len(a):len(a) + len(b), 0:len(a)]
sns.heatmap(corr, annot=True)
plt.show()

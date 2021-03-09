import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc

# 读入数据
data = pd.read_csv("data_2_test.csv")

# 提取出来的特征
# 肠道菌群
feature_DecisionTree = ['g__Lachnospiraceae_UCG-001', 'g__Victivallis', 'g__Paraprevotella',
                        'g__Lachnospiraceae_UCG-003']  # 决策树
feature_XGBoost = ['g__Ruminiclostridium_9', 'g__Parasutterella', 'g__Lachnospiraceae_UCG-001', 'g__Blautia',
                   'g__Ruminococcaceae_UCG-003']  # XG-Boost
feature_Forest = ['g__Abiotrophia', 'g__Acetanaerobacterium', 'g__Acidaminococcus', 'g__Acinetobacter',
                  'g__Actinomyces', 'g__Adlercreutzia', 'g__Akkermansia', 'g__Aliihoeflea']  # 随机森林

# 代谢物
feature_DecisionTree_2 = ['3-Dehydrocholic acid', 'Hydrocinnamic acid', 'Glycolithocholic acid']
feature_XGBoost_2 = ['Phthalic acid', '3-Dehydrocholic acid', 'Nicotinic acid']
feature_Forest_2 = ['Deoxycholic acid', 'Hyodeoxycholic acid', 'L-Lysine', 'L-Histidine', 'L-Arginine', 'Ornithine',
                    'L-Glutamine', 'L-Glutamic acid', 'Sarcosine']

# 模型训练
train, test = train_test_split(data, test_size=0.2)
train_X = train[feature_Forest_2]
train_Y = train['sample']
test_X = test[feature_Forest_2]
test_Y = test['sample']
# 0均值标准化
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.fit_transform(test_X)

# 创建svm分类器
random_state = np.random.RandomState(0)
svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)
test_predict_label = svm.fit(train_X, train_Y).decision_function(test_X)
prediction = svm.predict(test_X)
print("准确率： ", metrics.accuracy_score(prediction, test_Y))

# 绘制ROC曲线
fpr, tpr, thershold = roc_curve(test_Y, test_predict_label)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

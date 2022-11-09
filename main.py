import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings('ignore')

# 数据预处理
data = pd.read_csv("F:\\project\\中医药大学\\2021.12.29\\data\\data_1_65.csv")  # 训练集
data_test = pd.read_csv("F:\\project\\中医药大学\\2021.12.29\\data\\data_1_33.csv")  # 测试集

# 读取按显著性降序排序的特征
df_feature = pd.read_excel("F:\\project\\中医药大学\\2021.12.29\\data\\feature_importance_1.xlsx", usecols=[0],
                           names=None)
temp = df_feature.values.tolist()
feature = []
for tmp in temp:
    feature.append(tmp[0])

# 划分特征组
list_feature = []
x = []
min_num = 1  # 最少特征
max_num = len(feature)  # 最多特征
split = 1  # 分隔
for i in range(min_num, max_num, split):
    list1 = feature[:i]
    list_feature.append(list1)
    x.append(i)

# 训练模型
svm_score = []  # 存储每个划分子集的得分
svm_analysis = []  # svm分析结果d q


def svm_test(train_X, train_Y, test_X, test_Y, svm):
    '''
    svm 训练函数
    :param train_X:训练集
    :param train_Y:
    :param test_X:测试集
    :param test_Y:
    :param svm:svm模型
    :return:[0:svm score, 1:SVM Accuracy_Score, 2:SVM Precision, 3:SVM Recall, 4:SVM F1 - Score, 5:SVM AUC value, 6,7,8:roc曲线参数]
    '''
    temp = []  # 存储模型输出结果
    # 0均值标准化
    ss = StandardScaler()
    train_X = ss.fit_transform(train_X)
    test_X = ss.fit_transform(test_X)

    # 创建svm分类器 网格搜索
    parameters = {'kernel': ('linear', 'rbf'), 'C': range(
        1, 10, 1), 'random_state': range(1, 10, 2)}
    svm = GridSearchCV(svm.SVC(probability=True), parameters)
    svm.fit(train_X, train_Y)

    # 预测测试集分类结果
    predict_Y = svm.predict(test_X)

    # 0:SVM Score
    score = svm.score(test_X, test_Y)

    # print("Svm Score : score = %f " % score)
    temp.append(score)
    svm_score.append(score)

    # 1:SVM Accuracy_Score
    accuracy = metrics.accuracy_score(test_Y, predict_Y)
    # print("Svm Accuracy：accuracy = %f" % accuracy)
    temp.append(accuracy)

    # 2:SVM Precision
    precision = metrics.precision_score(test_Y, predict_Y)
    # print("Svm Precision：precision = %f" % precision)
    temp.append(precision)

    # 3:SVM Recall
    recall = metrics.recall_score(test_Y, predict_Y)
    # print("Svm Recall：recall = %f" % recall)
    temp.append(recall)

    # 4:SVM F1 - Score
    f1_score = metrics.f1_score(test_Y, predict_Y)
    # print("Svm Recall：f1_score = %f" % f1_score)
    temp.append(f1_score)

    # 5:SVM AUC value
    auc = metrics.roc_auc_score(test_Y, predict_Y)
    # print("SVM AUC value：auc = %f" % auc)
    temp.append(auc)

    # 6,7,8 roc曲线参数
    fpr, tpr, thershold = roc_curve(test_Y, predict_Y)
    temp.append(fpr)  # 6
    temp.append(tpr)  # 7
    temp.append(thershold)  # 8

    return temp


def roc_plt(fpr, tpr):
    '''
    绘制roc曲线
    :param fpr:
    :param tpr:
    :param thershold:
    :return:
    '''
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.savefig("65例选取特征验证.jpeg")
    # plt.savefig("65例选取特征验证.tiff")
    plt.savefig("33例选取特征验证.jpeg")
    plt.savefig("33例选取特征验证.tiff")
    plt.show()

'''
# 训练集调参，挑选特征
for rs in range(0, 100):
    svm_score = []
    svm_analysis = []

    for i in range(len(list_feature)):
        global svm
        train, test = train_test_split(data, test_size=0.2, random_state=rs) 
        train_X = train[list_feature[i]]
        train_Y = train['sample']
        test_X = test[list_feature[i]]
        test_Y = test['sample']
        svm_analysis.append(svm_test(train_X, train_Y, test_X, test_Y, svm))
        best = svm_score.index(max(svm_score))
        best_score = max(svm_score)
    print(rs, best, best_score)
'''

for i in range(len(list_feature)):
    # 循环训练模型
    # 使用各自的样本集，使用split函数做划分，训练集与测试集划分比例4：1
    global svm
    train, test = train_test_split(data, test_size=0.2, random_state=0)  
    train_X = train[list_feature[i]]
    train_Y = train['sample']
    test_X = test[list_feature[i]]
    test_Y = test['sample']
    svm_analysis.append(svm_test(train_X, train_Y, test_X, test_Y, svm))
    # 获取使得模型效果最好的特征数量
    best = svm_score.index(max(svm_score))

    print(best, max(svm_score), feature[:best])

    # 绘制训练曲线
    plt.plot(x, svm_score)
    # plt.title("analysis")
    plt.xlabel("num of features", fontsize=14)
    plt.ylabel("accuracy", fontsize=14)
    plt.xticks(np.arange(0, 26, 2))
    plt.savefig("65例样本集上选取特征过程.jpeg")
    plt.savefig("65例样本集上选取特征过程.tiff")
    plt.show()

# 绘制best对应的ROC曲线

fpr = svm_analysis[best][6]
tpr = svm_analysis[best][7]
thershold = svm_analysis[best][8]
roc_auc = auc(fpr, tpr)
# roc_plt(fpr, tpr)  # 绘制ROC曲线

# 使用另一个样本集做验证，使用split函数做划分，训练集与测试集划分比例4：1

train, test = train_test_split(data_test, test_size=0.2, random_state=4)
train_X = train[list_feature[best]]
train_Y = train['sample']
test_X = test[list_feature[best]]
test_Y = test['sample']
temp_test = svm_test(train_X, train_Y, test_X, test_Y, svm)
print(temp_test[0])  # 输出测试集准确率
fpr_test = temp_test[6]
tpr_test = temp_test[7]
roc_auc_test = auc(fpr_test, tpr_test)
roc_plt(fpr_test, tpr_test)  # 绘制roc曲线

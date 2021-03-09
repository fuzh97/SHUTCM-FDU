import pandas as pd
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import re

# 读入数据
data = pd.read_csv("data_2.csv")
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                data.columns.values]

# 划分特征值
x = data.iloc[:, data.columns != "sample"]
y = data.iloc[:, data.columns == "sample"]

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x
                                                    , y.values.reshape(y.shape[0], )
                                                    , test_size=0.3
                                                    , random_state=22
                                                    )

# 拟合XGBoost模型
model = XGBClassifier()
model.fit(x_train, y_train)

# 对测试集做预测
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

# 评估预测结果
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# 训练结果可视化
plot_importance(model)
plt.show()

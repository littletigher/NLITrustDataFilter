import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 样例数据
# data_features = np.random.rand(100, 3)  # 假设 A 数据有 3 个特征
trustworthiness = np.random.rand(100, 1)  # 可信度指标（0-1）
contradiction = np.random.rand(100, 1)  # 矛盾度指标（0-1）
reliability_labels = np.random.randint(0, 2, size=(100,))  # 真实可靠性标签（0 或 1）

# 样例文字原因
trustworthiness_reasons = ["Reason A"] * 100
contradiction_reasons = ["Reason B"] * 100

# 使用TF-IDF进行tokenizer编码
vectorizer = TfidfVectorizer()
trustworthiness_reasons_encoded = vectorizer.fit_transform(trustworthiness_reasons).toarray()
contradiction_reasons_encoded = vectorizer.fit_transform(contradiction_reasons).toarray()

# 将所有特征拼接在一起
X = np.hstack((data_features, trustworthiness, contradiction, trustworthiness_reasons_encoded, contradiction_reasons_encoded))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, reliability_labels, test_size=0.2, random_state=42)

# 构建决策树模型
'''
    log_loss:是交叉熵损失函数，适用于二分类问题，他在需要高置信度的概率预测时，表现得很好
    max_depth:二分类问题中，通常3到5的深度就可以得到不错的效果。
    random_state:随机种子，保证每次运行结果一致
'''
clf = DecisionTreeClassifier(
    criterion="log_loss",
    max_depth=5,
    random_state=42
)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 输出准确率
accuracy = clf.score(X_test, y_test)
print(f"模型的准确率: {accuracy:.2f}")

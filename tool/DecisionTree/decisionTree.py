import pickle
import numpy as np
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
import pandas as pd

def decisionTree(confidence_flag, score_of_confidence, contradict_flag, score_of_contradict):
    # 1. 使用 pickle 加载模型
    with open(r'D:\PycharmProjects\TDFilter\tool\DecisionTree\radiation_dctree_qwen.pkl', 'rb') as file:
        clf_loaded = pickle.load(file)

    # 2. 构造要预测的数据，使用 DataFrame 并指定列名
    X_new = pd.DataFrame({
        'confidence_flag': [confidence_flag],
        'score_of_confidence': [score_of_confidence],
        'contradict_flag': [contradict_flag],
        'score_of_contradict': [score_of_contradict]
    })

    # 3. 使用加载的模型进行预测
    y_pred = clf_loaded.predict(X_new)

    return y_pred[0]


if __name__ == "__main__":
    # 1. 加载模型
    confidence_flag = 1
    score_of_confidence = 0.85
    contradict_flag = 0
    score_of_contradict = 0.2

    # 使用 decisionTree 函数进行预测
    prediction = decisionTree(confidence_flag, score_of_confidence, contradict_flag, score_of_contradict)

    # 输出预测结果
    # print(f"预测结果: {prediction}")
import json
from collections import Counter
import matplotlib.pyplot as plt

# 读取 JSON 文件
file_path = r'D:\PycharmProjects\TDFilter\DataSet\Biology\biologicExtend.json'
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    data = []

# 统计 topic 的数量
topic_counts = Counter(record.get("topic", "Unknown") for record in data)

# 提取数据用于柱状图
topics = list(topic_counts.keys())
counts = list(topic_counts.values())

# 打印统计的 topic 数量
print(f"Number of unique topics: {len(topics)}")

# 绘制柱状图
plt.figure(figsize=(12, 6))
plt.bar(topics, counts, color='skyblue', edgecolor='black')
plt.xlabel('Topics', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Topics', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.tight_layout()
plt.show()

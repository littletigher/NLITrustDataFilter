from pymilvus import connections, Collection
import numpy as np

from pymilvus import connections, Collection
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import os
load_dotenv()
class MilvusWrapper:
    def __init__(self, host='localhost', port='19530', collection_name='text_collection'):
        self.collection_name = collection_name
        self.collection = None

        # 连接到 Milvus
        connections.connect("default", host=host, port=port)

        # 加载集合
        self.collection = Collection(self.collection_name)
        print(f"Connected to Milvus collection: {self.collection_name}")

        # 初始化嵌入模型
        self.model_name = os.getenv("embeddings_model")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()  # 设定模型为评估模式（不启用 Dropout 等机制）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model {self.model_name} loaded and moved to {self.device}.")

    def embed_text(self, text):
        """
        将文本嵌入为向量。

        参数:
        text (str): 要编码的文本。

        返回:
        np.ndarray: 生成的嵌入向量。
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs, return_dict=True).pooler_output
        return embeddings.cpu().numpy().astype(np.float32)

    def insert_data(self, ids, texts):
        """
        插入文本和对应的嵌入向量到 Milvus 集合中。

        参数:
        ids (list[int]): 数据的唯一标识符。
        texts (list[str]): 与向量关联的知识文本。
        """
        embeddings = [self.embed_text(text) for text in texts]  # 自动编码文本

        # 插入数据
        self.collection.insert([ids, embeddings, texts])
        print(f"Inserted {len(ids)} records into the collection.")

    def query(self, text, top_k=10):
        """
        根据文本查询最相似的向量。

        参数:
        text (str): 用于查询的文本。
        top_k (int): 返回最相似的前 k 个结果。

        返回:
        list: 包含最相似向量的 ID、知识文本和距离的列表。
        """
        embedding = self.embed_text(text).tolist()

        results = self.collection.search(
            embedding,
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["knowledge"],  # 确保返回 `knowledge` 字段
            expr=None
        )

        # 解析并返回查询结果
        query_results = []
        for result in results[0]:
            result_info = {
                'id': result.id,
                'knowledge': result.entity.get('knowledge'),
                'distance': result.distance
            }
            query_results.append(result_info)

        return "["+query_results[0]['knowledge'] +"]"


if __name__ == "__main__":
    # 创建 MilvusWrapper 实例
    milvus_wrapper = MilvusWrapper()

    # 插入示例数据
    # ids = [1, 2]
    # texts = ["What is the capital of France?", "How to bake a cake?"]
    # milvus_wrapper.insert_data(ids, texts)

    # 查询示例
    query_text = "Tell me about the capital of France."
    results = milvus_wrapper.query(query_text, top_k=1)
    print(results[0]['knowledge'] if results else "No results found.")

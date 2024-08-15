from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from typing import List


class MilvusHelper:
    def __init__(self, host: str = "localhost", port: str = "19530", collection_name: str = "expert_collection"):
        self.collection_name = collection_name
        self.connect(host, port)
        self.collection = self.get_or_create_collection()

    def connect(self, host: str, port: str):
        connections.connect("default", host=host, port=port)

    def disconnect(self):
        connections.disconnect("default")

    def get_or_create_collection(self):
        if not utility.has_collection(self.collection_name):
            # Define the fields
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=16384),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # 设置维度为384
                FieldSchema(name="dim", dtype=DataType.INT64)
            ]

            # Create a schema
            schema = CollectionSchema(fields, "example collection schema")

            # Create the collection
            collection = Collection(self.collection_name, schema)

            # 创建索引
            index_params = {
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
                "metric_type": "L2"
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()

            return collection
        else:
            collection = Collection(self.collection_name)
            collection.load()
            return collection

    def add_data(self, summary: List[str], document: List[str], embedding: List[List[float]], dims: List[int]):
        data = [
            summary,  # summary
            document,  # document
            embedding,  # embedding
            dims  # dim
        ]
        self.collection.insert(data)
        self.collection.flush()

    def delete_data(self, ids: List[int]):
        expr = f"id in {ids}"
        self.collection.delete(expr)
        self.collection.flush()

    def query_data(self, expr: str):
        # Load the collection into memory before querying
        self.collection.load()
        results = self.collection.query(expr)
        return results

    def search_embeddings(self, query_vectors: List[List[float]], top_k: int = 10):
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=None
        )
        return results

    def update_data(self, id: int, summary: str = None, document: str = None, embedding: List[float] = None,
                    dim: int = None):
        expr = f"id == {id}"
        new_data = {}
        if summary is not None:
            new_data["summary"] = summary
        if document is not None:
            new_data["document"] = document
        if embedding is not None:
            new_data["embedding"] = embedding
        if dim is not None:
            new_data["dim"] = dim
        self.collection.update(expr, new_data)
        self.collection.flush()


# 示例使用
if __name__ == "__main__":
    helper = MilvusHelper()

    # 示例数据
    summaries = ["Example summary"]
    documents = ["Example document"]
    embeddings = [[0.1] * 384]  # 示例向量，维度为384
    dims = [384]

    # 添加数据
    helper.add_data(summaries, documents, embeddings, dims)

    # 查询数据：id 等于 0
    results = helper.query_data("id == 0")
    print("Query results:", results)

    # 搜索嵌入
    search_results = helper.search_embeddings([[0.1] * 384])
    print("Search results:", search_results)

    # 更新数据
    helper.update_data(0, summary="Updated summary")

    # 删除数据
    helper.delete_data([0])

    # 断开连接
    helper.disconnect()

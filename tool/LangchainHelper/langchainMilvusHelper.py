import os
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
class LangchainMilvusHelper:
    def __init__(self,uri:str = "http://localhost:19530" , collection_name: str = "expert_knowledge"):
        self.collection_name = collection_name
        self.vector_store = self.create_vector_store(uri=uri,collection_name=collection_name)

    # 创建方面
    def create_embeddings(self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        return embeddings
    def create_vector_store(self, uri:str, collection_name:str):
        embeddings = self.create_embeddings()
        vector_store = Milvus(
            embeddings,
            collection_name=collection_name,
            connection_args={"uri": uri},
            auto_id=True
        )
        return vector_store
    def get_vector_store(self):
        return self.vector_store
    # 新建
    def add_data(self,documents):
        self.vector_store.add_documents(documents)

    # 向量匹配查找
    def search_data(self, query:str, top_k:int):
        results = self.vector_store.similarity_search(
            query,
            k=top_k,
            filter={},
        )
        # for res in results:
        #     print(f"* {res.page_content} [{res.metadata}]")
        return results

# 测试代码
if __name__ == "__main__":
    helper = LangchainMilvusHelper()
    document_1 = Document(
        page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"summary": "this is a test document"},
    )
    documents = [document_1]
    helper.add_data(documents)
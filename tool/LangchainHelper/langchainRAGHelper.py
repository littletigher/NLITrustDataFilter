from dotenv import load_dotenv
from dotenv import dotenv_values
import os
from langchainMilvusHelper import LangchainMilvusHelper
load_dotenv(dotenv_path="D:\PycharmProjects\TrustDataFilter\.env")

class LangchainRAGHelper:
    def __init__(self):
        self.helper = LangchainMilvusHelper()
        self.vectorstore = self.helper.get_vector_store()

    def similarity_search(self, question):
        docs = self.vectorstore.similarity_search(question)
        return docs

    def add_data(self, documents):
        self.helper.add_data(documents)

    def search_data(self, query, top_k):
        results = self.helper.search_data(query, top_k)

if __name__ == "__main__":
    helper = LangchainRAGHelper()
    question = "I had a pen"
    docs = helper.similarity_search(question)
    print(docs[0])
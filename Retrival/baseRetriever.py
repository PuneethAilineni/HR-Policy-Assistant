from VectorStore.vectorStore import VectorStore
from langchain_core.documents import Document

class BaseRetriver():
    def __init__(self):
        vector_store = VectorStore()._get_vector_store()
        self.retriever = vector_store.as_retriever(
            search_type = 'similarity',
            search_kwargs = {'k':30}) 
        
    def invoke(self, query:str) -> list[Document]:
        return self.retriever.invoke(query)

    def _get_base_retriever(self):
        return self.retriever

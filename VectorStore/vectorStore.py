from consts import Consts
from langchain_chroma import Chroma
from langchain_core.documents import Document

class VectorStore():
    def __init__(self):
        self.persist_directory = Consts()._get_persist_directory()
        self.embedding = Consts()._get_embegging()
        self.vector_store = Chroma(
            collection_name='HR-Policy',
            embedding_function=self.embedding,
            persist_directory=self.persist_directory
        )
        
    def store(self, documents:Document):
        self.vector_store.add_documents(
            documents=documents
        )

    def _get_vector_store(self):
        return self.vector_store



from consts import Consts
from VectorStore.vectorStore import VectorStore
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever

class multiQueryRetriver():
    def __init__(self):
        llm = Consts()._get_llm()
        prompt = Consts()._get_prompt()

        vector_store = VectorStore()._get_vector_store()
        base_retriever = vector_store.as_retriever(
            search_type = 'mmr',
            search_kwargs = {'k':10,'fetch_k':100,'lambda_mult':0.8}) 
        
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            prompt=prompt,
        )

    def invoke(self, query:str) -> list[Document]:
        return self.retriever.invoke(query)

    def _get_multi_query_retriever(self):
        return self.retriever

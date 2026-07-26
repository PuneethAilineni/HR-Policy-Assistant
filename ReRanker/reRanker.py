from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from Retrival.multiQueryRetriever import multiQueryRetriver
from dotenv import load_dotenv

class ReRanker:
    def __init__(self):
        load_dotenv()
        base_retriever = multiQueryRetriver()._get_multi_query_retriever()
        compressor = NVIDIARerank(model='nv-rerank-qa-mistral-4b:1',top_n=3)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

    def invoke(self,query):
        return self.retriever.invoke(query)
        

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import PromptTemplate

class Consts:

    def __init__(self):
        load_dotenv()
        self.folder_path = './Content'
        self.persist_directory = './HR-Policy-Chunks'
        self.prompt = PromptTemplate.from_template(
            """Generate 3 alternative versions of the given user question to retrieve relevant documents from a vector database. Provide ONLY the alternative questions separated by newlines, no intro text.

Original question: {question}"""
        )
        self.embegging = NVIDIAEmbeddings(model='nvidia/nv-embed-v1')
        self.llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct", 
            temperature=0
        )

    def _get_folder_path(self):
        return self.folder_path

    def _get_persist_directory(self):
        return self.persist_directory

    def _get_prompt(self):
        return self.prompt

    def _get_embegging(self):
        return self.embegging

    def _get_llm(self):
        return self.llm
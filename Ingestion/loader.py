import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from Utils import logger

class Loader(BaseLoader):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def load(self) -> list[Document]:
        docs = []
        try:
            dir_loader = DirectoryLoader(
                path=self.folder_path,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},
                show_progress=True
            )
            docs = dir_loader.load()
        except Exception as e:
            logger.warning(f"Could not extract text from Content because of -> {e}")
        logger.info(f"Loading successfully done, total docs: {len(docs)}")
        return docs

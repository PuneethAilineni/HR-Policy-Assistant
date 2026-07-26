from Ingestion.loader  import Loader
from Chunking.textSplitter import Splitter
from VectorStore.vectorStore import VectorStore
from Utils import logger
from consts import Consts

class RAG:
    def __init__(self):
        self.Consts = Consts()
        self.folder_path = Consts()._get_folder_path()
        self.loader = Loader()
        self.splitter = Splitter()
        self.vector_store = VectorStore()

    def pipeline(self):
        try:
            docs = self.loader.load(folder_path=self.folder_path)
            chunks = self.splitter.split_documents(documents=docs)
            self.vector_store.store(chunks)
        except Exception as e:
            logger.info("pipeline failed to store vectors in database")

        logger.info("successfully stored chunks in database")


if __name__ == '__main__':
    RAG().pipeline()




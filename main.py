from Ingestion.loader  import Loader
from Chunking.textSplitter import Splitter
from VectorStore.vectorStore import VectorStore
from Retrival.multiQueryRetriever import multiQueryRetriver 
from ReRanker.reRanker import ReRanker
from Utils import logger
from consts import Consts

if __name__ == '__main__':
    # logger.info('<---- ingestion started ---->')
    # try:
    #     folder_path = Consts()._get_folder_path()
    #     loader = Loader(folder_path=folder_path)
    #     docs = loader.load()
    #     logger.info(len(docs))
    # except Exception as e:
    #     logger.warning(f"failed at ingestion due to -> {e}")
    #     exit
    # logger.info('<---- ingestion ended ---->')


    # logger.info('<---- text splitting started ---->')
    # try:
    #     splitter = Splitter(documents=docs)
    #     chunks = splitter.split_documents()
    #     logger.info(len(chunks))
    # except Exception as e:
    #     logger.warning(f"failed at text splitting due to -> {e}")
    #     exit
    # logger.info('<---- text splitting ended ---->')


    # logger.info('<---- vector store started ---->')
    # try:
    #     VectorStore().store(documents=chunks)
    # except Exception as e:
    #     logger.warning(f"failed at vector store due to -> {e}")
    #     exit
    # logger.info('<---- vector store ended ---->')


    # logger.info('<---- Retrival started ---->')
    # try:
    #     query = 'How many days in a week can i take work from home in a week?'
    #     retriever = multiQueryRetriver()
    #     results = retriever.invoke(query=query)
    #     for i,doc in enumerate(results,1):
    #         print(f"Result {i} | source = {doc.metadata.get('source', 'N/A')}")
    #         print(f"{doc.page_content}")
    #         print('\n\n')
    # except Exception as e:
    #     logger.warning(f"failed at retrieval due to -> {e}")
    #     exit
    # logger.info('<---- Retrieval ended ---->')


    logger.info('<---- Reranking started ---->')
    try:
        query = 'How many days in a week can i take work from home in a week?'
        retriever = ReRanker()
        results = retriever.invoke(query=query)
        for i,doc in enumerate(results,1):
            print(f"Result {i} | source = {doc.metadata.get('source', 'N/A')}")
            print(f"{doc.page_content}")
            print('\n\n')
    except Exception as e:
        logger.warning(f"failed at reranking due to -> {e}")
        exit
    logger.info('<---- Reranking ended ---->')




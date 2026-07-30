import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from Utils import logger

class Splitter():
    def __init__(self):
        pass
    def split_documents(self, documents: List[Document]) -> List[Document]:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        semantic_chunks = []
        for doc in documents:
            source_path = doc.metadata.get('source', 'unknown_source')
            
            md_splits = markdown_splitter.split_text(doc.page_content)
            
            for split in md_splits:
                split.metadata['source'] = source_path
                semantic_chunks.append(split)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
        final_ready_chunks = text_splitter.split_documents(semantic_chunks)

        logger.info(f"Created {len(final_ready_chunks)} chunks ready for your Vector Database.")
        return final_ready_chunks

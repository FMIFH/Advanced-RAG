from typing import List
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

from dotenv import load_dotenv
import os

load_dotenv('.env')
storage_path = os.getenv('STORAGE_PATH')

class vectorStore():
    def __init__(self) -> None:
        if storage_path is None:
            raise ValueError('STORAGE_PATH environment variable is not set')
        os.makedirs(storage_path, exist_ok=True)
        self.client = Chroma(collection_name="rag-chroma", embedding_function=OpenAIEmbeddings(), persist_directory=storage_path)
        print(len(self.client.get()['documents']))
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=0)

    def as_retriever(self) -> VectorStoreRetriever:
        return self.client.as_retriever()
    
    def add_documents(self, paths: List[str]) -> None:
        docs = []
        sources = list(set([v['source'] for v in self.client.get()["metadatas"]]))
        for file in paths:
            if file in sources:
                continue
            print('New Doc')
            docs += PyPDFLoader(file).load()
        if len(docs) > 0:
            doc_splits = self.text_splitter.split_documents(docs)
            self.client.add_documents(doc_splits)
        else:
            print('No new docs')
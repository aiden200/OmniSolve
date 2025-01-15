from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.llms import OpenAI
# from langchain.transformers import LLMGraphTransformer

class RAGSystem:
    def __init__(self, db_dir, model_name = "sentence-transformers/all-MiniLM-L12-v1", warnings=[]):
        self.all_docs = []  # Store embeddings of text documents
        self.metadata = {}
        self.failed = []
        self.db_dir = db_dir
        self.db = None
        self.huggingface_ef = SentenceTransformerEmbeddings(model_name=model_name, model_kwargs={"trust_remote_code" : True})
        self.warnings = {warning: self.huggingface_ef.embed_query(warning) for warning in warnings}


    def _create_vector_store(self, docs):
        persistent_directory = os.path.join(self.db_dir)
        self.db = Chroma.from_documents(
            docs, self.huggingface_ef, persist_directory=persistent_directory, collection_metadata={"hnsw:space": "cosine"}
        )

    def detect_warnings(self, new_objects, threshold=0.8):
        
        warning_messages = []

        for object in new_objects:
            new_object_embedding = self.huggingface_ef.embed_query(object)
            
            for warning, embedding in self.warnings.items():
                similarity = cosine_similarity(
                    [new_object_embedding], [embedding]
                )[0][0]
                # print(f"object {warning} matched object {object} with similarity of: {similarity}")
                if similarity > threshold:
                    warning_messages.append(f"WARNING: object {warning} matched object {object} with similarity of: {similarity}")
        
        return warning_messages

    def _split_text_per_vid(self, texts, vid_id):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        documents = text_splitter.create_documents([texts])
        for doc in documents:
            if not hasattr(doc, "metadata"):
                doc.metadata = {}
            doc.metadata["full_video_id"] = vid_id
        return documents

    def add_to_rag(self, next_context, vid_id):
        if next_context:
            docs = self._split_text_per_vid(next_context, vid_id)
            self.all_docs += docs
            self._create_vector_store(self.all_docs)
    
    

    def add_to_kg(self, object_information, vid_id):

        pass


    def query_vector_store(self, query):
        if not self.db:
            return "__NONE__"
        retriever = self.db.as_retriever(
            # search_type="similarity_score_threshold",
            # search_kwargs={"k": 10, "score_threshold": 0.1},
            search_type="mmr",
            search_kwargs={'k': 10, 'lambda_mult': 0.25}
        )
        relevant_docs = retriever.invoke(query)
        return relevant_docs
    
    def query_kg(self, query):
        pass




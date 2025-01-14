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
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.embeddings import SentenceTransformerEmbeddings


import csv


# Load HuggingFace multilingual CLIP model and processor
# clip_model = CLIPModel.from_pretrained(model_name)
# clip_processor = CLIPProcessor.from_pretrained(model_name)

class RAGSystem:
    def __init__(self, model_name = "sentence-transformers/all-MiniLM-L12-v1"):
        self.all_docs = []  # Store embeddings of text documents
        self.metadata = {}
        self.failed = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_dir = os.path.join(current_dir, "db")
        self.db = None
        self.huggingface_ef = SentenceTransformerEmbeddings(model_name=model_name, model_kwargs={"trust_remote_code" : True})
        # self.huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
        #         api_key=os.environ["hf_token"],
        #         model_name=model_name
        #     )


    def _create_vector_store(self, docs, store_name):
        persistent_directory = os.path.join(self.db_dir, store_name)

        if not os.path.exists(persistent_directory):
            print(f"\n--- Creating vector store {store_name} ---")
            self.db = Chroma.from_documents(
                docs, self.huggingface_ef, persist_directory=persistent_directory, collection_metadata={"hnsw:space": "cosine"}
            )
            print(f"--- Finished creating vector store {store_name} ---")
        else:
            self.db = Chroma(persist_directory=persistent_directory, embedding_function=self.huggingface_ef, collection_metadata={"hnsw:space": "cosine"})
            print(
                f"Vector store {store_name} already exists. No need to initialize.")    
    
    def _load_from_dir(self, store_name="multivent1"):
        persistent_directory = os.path.join(self.db_dir, store_name)

        if not os.path.exists(persistent_directory):
            raise ValueError("Have not initialized db")
        
        self.db = Chroma(persist_directory=persistent_directory, embedding_function=self.huggingface_ef, collection_metadata={"hnsw:space": "cosine"})

    def _split_text_per_vid(self, texts, full_video_id, short_video_id=None):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        documents = text_splitter.create_documents([texts])
        for doc in documents:
            if not hasattr(doc, "metadata"):
                doc.metadata = {}
            doc.metadata["full_video_id"] = full_video_id
            if short_video_id:
                doc.metadata["short_video_id"] = short_video_id
        return documents

    def read_video_llm_file(self, folder_path, summarization_eval, store_name="multivent1", filters=None):
        # vid_name, vid_path, vid_text, vid_id
        vid_path = os.path.join(folder_path, "split_videos/")
        persistent_directory = os.path.join(self.db_dir, store_name)
        if os.path.exists(persistent_directory):
            self._load_from_dir()
        #     print("DB already exists, skipping")
        #     return

        with open(os.path.join(folder_path, "results.csv")) as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                vid_name, failed, language, event_category, event_name = row
                
                if failed == "True":
                    self.failed.append(vid_name)
                else:
                    i = 0
                    if os.path.exists(persistent_directory):
                        continue
                    if filters and language != filters:
                        continue
                    if summarization_eval:
                        with open(os.path.join(vid_path, vid_name, f"summary.txt"), 'r') as file:
                            content = file.read()
                        docs = self._split_text_per_vid(content, vid_name)
                        self.all_docs += docs
                    else:
                        while os.path.exists(os.path.join(vid_path, vid_name, f"{i}.mp4")):
                            with open(os.path.join(vid_path, vid_name, f"{i}.txt"), 'r') as file:
                                content = file.read()
                            docs = self._split_text_per_vid(content, vid_name, i)
                            self.all_docs += docs
                            i += 1
        
        self._create_vector_store(self.all_docs, store_name)


    def query_vector_store(self, query, store_name = "multivent1"):
        if not self.db:
            self._load_from_dir(store_name=store_name)
        retriever = self.db.as_retriever(
            # search_type="similarity_score_threshold",
            # search_kwargs={"k": 10, "score_threshold": 0.1},
            search_type="mmr",
            search_kwargs={'k': 10, 'lambda_mult': 0.25}
        )
        relevant_docs = retriever.invoke(query)
        return relevant_docs
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('full_video_id', 'Unknown')}, \
                        video: {doc.metadata.get('short_video_id', 'Unknown')}\n")



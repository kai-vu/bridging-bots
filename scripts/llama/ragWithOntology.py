import warnings
warnings.filterwarnings('ignore')

import os
import re
import sys
import time
import json
import numpy as np
import chromadb

from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def get_llm(llm_model, api_key):
    llm = Groq(model=llm_model, api_key=api_key)
    return llm

def get_collection():
    chroma_client = chromadb.EphemeralClient()
    try:
        chroma_client.delete_collection("euRobin-collection")
    except:
        pass
    chroma_collection = chroma_client.create_collection("euRobin-collection")
    return chroma_collection

def get_embedding_function(embedding_model):
    embedding_function = HuggingFaceEmbedding(model_name=embedding_model)
    return embedding_function

def load_rag_files(ontology_path):
    rag_files = SimpleDirectoryReader(ontology_path).load_data()
    return rag_files

def get_vector_store(chroma_collection, rag_files, embedding_function):
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        rag_files, storage_context=storage_context, embed_model=embedding_function
    )
    return index

def get_image_description(description_path):
    content = open(description_path, 'r', encoding='utf-8').read()
    return content

def make_user_prompt(prompt_template, content):
    user_prompt = prompt_template.format(input_content=content)
    return user_prompt

def get_response(llm_model, index, llm, user_prompt):
    Settings.llm = Groq(model=llm_model)
    query_engine = index.as_query_engine(llm)
    response = query_engine.query(user_prompt)
    return response

def main(llm_model, api_key, embedding_model, ontology_path, description_path, prompt_template):
    llm = get_llm(llm_model, api_key)
    chroma_collection = get_collection()
    embedding_function = get_embedding_function(embedding_model)
    rag_files = load_rag_files(ontology_path)
    index = get_vector_store(chroma_collection, rag_files, embedding_function)
    content = get_image_description(description_path)
    user_prompt = make_user_prompt(prompt_template, content)
    response = get_response(llm_model, index, llm, user_prompt)
    print(response)

if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(current_dir, ".env")
    load_dotenv(dotenv_path)

    llm_model = os.getenv("LLM_MODEL")
    api_key = os.getenv("API_KEY")
    ontology_path = os.getenv("ONTOLOGY_PATH")
    description_path = os.getenv("OUTPUT_PATH")
    output_path = os.path.join(os.path.split(description_path)[0], 'llama_graph_schemaExtractor.html')
    
    embedding_model = "BAAI/bge-small-en"

    prompt_template = """
    Your task is to extract triples from the image description below, based on the ontology provided to you in the vector. 
    Return results in JSON format.

    ## IMAGE DESCRIPTION ##
    {input_content}
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main(llm_model, api_key, embedding_model, ontology_path, description_path, prompt_template)
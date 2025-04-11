import warnings
warnings.filterwarnings('ignore')

import os
import re
import sys
import time
import json
import base64
import numpy as np
import chromadb

from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import ImageDocument

current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path)

llm_model = os.getenv("MODEL")
api_key = os.getenv("API_KEY")
image_path = os.getenv("IMAGE_PATH")
ontology_path = os.getenv("ONTOLOGY_PATH")

user_query = "What's in this image?"

llm = Groq(model=llm_model, api_key=api_key)

with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

test_image = [ImageDocument(image_path=image_path)]

documents = SimpleDirectoryReader(ontology_path).load_data()

chroma_client = chromadb.EphemeralClient()
try:
    chroma_client.delete_collection("test-collection")
except:
    pass
chroma_collection = chroma_client.create_collection("test-collection")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model)

query_engine = index.as_query_engine(llm)

response = query_engine.query(user_query, image_documents=test_image)

#print(response)

"""
from llama_index.multi_modal_llms.groq import GROQLLaVa
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.schema import ImageDocument
# Initialize the GROQ LLaVa model
llava_model = GROQLLaVa(model="llava-latest", api_key="your_groq_api_key")
# Load your documents and images
documents = SimpleDirectoryReader("path_to_your_text_documents").load_data()
images = [ImageDocument(image_path="path_to_your_image.jpg")]
# Create a Multi-Modal Index
storage_context = StorageContext.from_defaults()
index = MultiModalVectorStoreIndex.from_documents(documents + images, storage_context=storage_context)
# Query the index using LLaVa
prompt = "Your combined text and image prompt here"
response = llava_model.query(prompt, image_documents=images)
print(response)
"""
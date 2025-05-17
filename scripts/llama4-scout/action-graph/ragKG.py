import warnings
warnings.filterwarnings('ignore')

import os
import re
import json
import chromadb
import tiktoken

from pyld import jsonld
from rdflib import Graph
from pathlib import Path
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document


def extract_description_from_json(description_path):
    with open(description_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    content = data['choices'][0]['message']['content']
    return content

def make_chroma_collection(collection_name):
    chroma_client = chromadb.EphemeralClient()
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    chroma_collection = chroma_client.create_collection(collection_name)
    return chroma_collection

def load_ontology(ontology_path):
    g = Graph()
    g.parse(ontology_path, format='turtle')
    ontology_text = g.serialize(format='turtle')
    ontology = [Document(text=ontology_text)]
    return ontology

def make_vector_store_index(ontology_path, embed_model, llm, callback_manager):
    collection_name = "euRobin-collection"
    chroma_collection = make_chroma_collection(collection_name)
    ontology = load_ontology(ontology_path)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        ontology, storage_context=storage_context, embed_model=embed_model, llm=llm, callback_manager=callback_manager)
    return index, chroma_collection

def get_response(llm, embedding_model, description_path, ontology_path, callback_manager, user_query):
    content = extract_description_from_json(description_path)
    embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    index, chroma_collection = make_vector_store_index(ontology_path, embed_model, llm, callback_manager)
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(user_query)
    return response, chroma_collection

def save_response(response, output_path):
    response_usage_path = os.path.join(output_path, "response_usage.json")
    response_usage = {
        "embedding_tokens": token_counter.total_embedding_token_count,
        "llm_prompt_tokens": token_counter.prompt_llm_token_count,
        "llm_completion_tokens": token_counter.completion_llm_token_count,
        "total_tokens": token_counter.total_llm_token_count,
    }
    with open(response_usage_path, 'w', encoding='utf-8') as file:
        json.dump(response_usage, file, indent=4, ensure_ascii=False)
    ttl_response_path = os.path.join(output_path, "kg.ttl")
    ttl_response = str(response)
    ttl_response = re.sub(r"^```[^\n]*\n", "", ttl_response.strip())
    ttl_response = re.sub(r"```$", "", ttl_response.strip())
    ttl_response = ttl_response.strip()
    with open(ttl_response_path, 'w') as f:
        f.write(ttl_response)
    return 

def clean_up_chroma(chroma_collection):
    doc_ids = chroma_collection.get()["ids"]
    chroma_collection.delete(ids=doc_ids)
    return

def main(llm_model, groq_key, embedding_model, description_path, output_path, ontology_path, callback_manager, user_query):
    llm = Groq(model=llm_model, api_key=groq_key, callback_manager=callback_manager)
    response, chroma_collection = get_response(llm, embedding_model, description_path, ontology_path, callback_manager, user_query)
    save_response(response, output_path)
    clean_up_chroma(chroma_collection)

if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    groq_key = os.getenv("GROQ_KEY")
    robot_task = os.getenv("ROBOT_TASK")
    embedding_model = "BAAI/bge-small-en"

    description_path = "../../../output/llama4-scout/observation-graph/image-description.json"
    with open(description_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    description_txt = data["choices"][0]["message"]["content"]
    
    output_path = "../../../output/llama4-scout/action-graph/ragKG"
    ontology_path = "../../../ontology/ontoActionGraph.ttl"

    token_counter = TokenCountingHandler(
        tokenizer = tiktoken.get_encoding("cl100k_base").encode
    )
    callback_manager = CallbackManager([token_counter])

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    user_query = f"""
    Given the ontology information, your task is to generates a Knowledge Graph of the actions a robot must complete to fulfil a task in an environment.
    You are provided with text description of the environment and the task, found below. 

    You must use the ontology **as a strict schema** to construct the Knowledge Graph.
    This means:
    - Use **only** the classes and properties defined in the ontology.
    - Do **not invent or infer** terms not explicitly defined in the ontology.
    - All entities and relations must conform to the structure and semantics of the ontology.

    Output format:
    - Output only text, with no extra explanations.
    - Output must consist of triples in turtle format.
    - Output must contain the prefixes and namespaces.

    ---------------------
    TASK: {robot_task}

    ENVIRONMENT DESCRIPTION: {description_txt}
    ---------------------
    """

    main(llm_model, groq_key, embedding_model, description_path, output_path, ontology_path, callback_manager, user_query)
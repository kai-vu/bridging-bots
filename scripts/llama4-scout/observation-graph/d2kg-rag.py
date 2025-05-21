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

def get_response(llm, embedding_model, description_path, ontology_path, callback_manager):
    content = extract_description_from_json(description_path)
    embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    index, chroma_collection = make_vector_store_index(ontology_path, embed_model, llm, callback_manager)
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(f"""
You are an intelligent assistant tasked to generate a **Knowledge Graph of the environment described as follows**:
---------------------
ENVIRONMENT DESCRIPTION: {content}
---------------------

Instructions:
- Analyze the description carefully to understand the complete layout of the environment.
- Based on the ontology stored in the vector, **generate a Knowledge Graph describing the environment**.
- All entities and relations must conform to the structure and semantics of the ontology.
- **Use only classes and properties from the ontology.**
- Do **NOT invent or infer any terms or actions outside of the ontology schema.**

Output format:
- Return only the generated Knowledge Graph.
- Output only text, no extra explanations.
- Use Turtle format for the output, such as <subject> <predicate> <object> .
- Include all prefixes and namespaces at the beginning. 
- Use the ex: prefix with namespace <http://example.org/data/> only for newly instantiated entities instantiated, such as specific actions, objects, or locations.
- Do not use the ex: prefix for ontology classes, properties, or schema definitions, those must strictly come from the provided ontology with their original prefixes and namespaces.
""")
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

def main(llm_model, groq_key, embedding_model, description_path, output_path, ontology_path, callback_manager):
    llm = Groq(model=llm_model, api_key=groq_key, callback_manager=callback_manager)
    response, chroma_collection = get_response(llm, embedding_model, description_path, ontology_path, callback_manager)
    save_response(response, output_path)
    clean_up_chroma(chroma_collection)

if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    groq_key = os.getenv("GROQ_KEY")
    embedding_model = "BAAI/bge-small-en"

    description_path = "../../../output/llama4-scout/observation-graph/image-description.json"
    output_path = "../../../output/llama4-scout/observation-graph/d2kg-rag"
    ontology_path = "../../../ontology/ontoObservationGraph.ttl"

    token_counter = TokenCountingHandler(
        tokenizer = tiktoken.get_encoding("cl100k_base").encode
    )
    callback_manager = CallbackManager([token_counter])

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main(llm_model, groq_key, embedding_model, description_path, output_path, ontology_path, callback_manager)
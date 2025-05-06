import warnings
warnings.filterwarnings('ignore')

import os
import re
import json
import shutil
import chromadb

from pyld import jsonld
from pathlib import Path
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document


def extract_sections_from_json(description_path):
    with open(description_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    content = data['choices'][0]['message']['content']
    sections = content.strip().split('\n\n')
    section1 = sections[0].strip()
    section2 = sections[1].strip()
    return section1, section2


def make_output_file_path(output_path, dir_name, output_file):
    full_output_dir = os.path.join(output_path, dir_name)
    if os.path.exists(full_output_dir):
        shutil.rmtree(full_output_dir)
    os.makedirs(full_output_dir)
    full_output_file = os.path.join(full_output_dir, output_file)
    return full_output_file

def make_chroma_collection(collection_name):
    chroma_client = chromadb.EphemeralClient()
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    chroma_collection = chroma_client.create_collection(collection_name)
    return chroma_collection

def load_ontology(ontology_path):
    with open(ontology_path, "r") as f:
        ontology_json = json.load(f)
    flattened = jsonld.flatten(ontology_json)
    ontology_text = json.dumps(flattened, indent=2)
    ontology = [Document(text=ontology_text)]
    return ontology

def make_vector_store_index(ontology_path, embed_model, llm):
    collection_name = "euRobin-collection"
    chroma_collection = make_chroma_collection(collection_name)
    ontology = load_ontology(ontology_path)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        ontology, storage_context=storage_context, embed_model=embed_model, llm=llm,
    )
    return index

def get_response(llm, embedding_model, description_path, ontology_path):
    section1, section2 = extract_sections_from_json(description_path)
    embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    index = make_vector_store_index(ontology_path, embed_model, llm)
    query_engine = index.as_query_engine(llm)
    response = query_engine.query(f"""
## INSTRUCTIONS ##
You are an intelligent assistant that generates Knowledge Graphs for robotics planning tasks.

You are provided with:
- A description of a physical environment.
- A workflow of ordered actions for a robot to perform a certain task.
- An ontology (retrieved from context) that defines the allowed vocabulary: classes, properties, and relations.

## TASK ##
You must use the ontology **as a strict schema** to construct a Knowledge Graph.
This means:
- Use **only** the classes and properties defined in the ontology.
- Do **not invent or infer** terms not explicitly defined in the ontology.
- All entities and relations must conform to the structure and semantics of the ontology.

## OUTPUT FORMAT ##
- Output only text, with no extra explanations.
- Output must consist of triples in turtle format.
- Organize your output into two clearly labeled sections:

### Triples from Environment Description ###
(triples based on Section 1)

### Triples from Robot Actions ###
(triples based on Section 2)

## INPUT ##
### Section 1: Environment Description ###
{section1}

### Section 2: Ordered Robot Actions ###
{section2}
""")
    return response

def save_llm_kg_response(response, output_path):
    output_path_og = make_output_file_path(output_path, "observationGraph", "kg.ttl")
    output_path_ag = make_output_file_path(output_path, "actionGraph", "kg.ttl")

    sections = re.split(r'##\s*.*?\s*##', str(response))
    sections = [s.strip() for s in sections if s.strip()]
    og_section, ag_section = sections

    with open(output_path_og, 'w', encoding='utf-8') as f:
        f.write(og_section)

    with open(output_path_ag, 'w', encoding='utf-8') as f:
        f.write(ag_section)
    return 

def main(llm_model, api_key, embedding_model, description_path, output_path, ontology_path):
    llm = Groq(model= llm_model, api_key=api_key)
    response = get_response(llm, embedding_model, description_path, ontology_path)
    save_llm_kg_response(response, output_path)

if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    api_key = os.getenv("GROQ_KEY")
    embedding_model = "BAAI/bge-small-en"

    description_path = "../../output/vlm_vlm/llama-image-description.json"
    output_path = "../../output/vlm_vlm/ragKG"
    ontology_path = "../../ontology/onto.jsonld"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main(llm_model, api_key, embedding_model, description_path, output_path, ontology_path)
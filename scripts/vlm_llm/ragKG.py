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
from llama_index.core import StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.chat_engine import SimpleChatEngine


def extract_description_from_json(description_path):
    with open(description_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    description = data['choices'][0]['message']['content']
    return description

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
    return chroma_collection, index

def get_response(llm, embedding_model, description, ontology_path, robot_task):
    embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    chroma_collection, index = make_vector_store_index(ontology_path, embed_model, llm)
    query_engine = index.as_query_engine(llm)
    response_obs = query_engine.query(f"""
    ## INSTRUCTIONS ##
    You are an intelligent assistant that generates Knowledge Graphs for robotics planning tasks.

    You are provided with:
    - A description of a physical environment.
    - An ontology (retrieved from context) that defines the allowed vocabulary: classes, properties, and relations.

    ## TASK ##
    You must use the ontology **as a strict schema** to construct a Knowledge Graph based solely on the environment description.
    This means:
    - Use **only** the classes and properties defined in the ontology.
    - Do **not invent or infer** terms not explicitly defined in the ontology.
    - All entities and relations must conform to the structure and semantics of the ontology.

    ## OUTPUT FORMAT ##
    - Output only text, with no extra explanations.
    - Output must consist of triples in turtle format.
    - Organize your output into a single clearly labeled section:

    ### Triples from Environment Description ###
    (triples generated from the input description)

    ## INPUT ##
    ### Environment Description ###
    {description}
    """)

    response_acs = query_engine.query(f"""
    ## INSTRUCTIONS ##
    You are an intelligent assistant that generates Knowledge Graphs for robotics planning tasks.

    You are provided with:
    - A description of a physical environment.
    - A task that a robot must complete in that environment.
    - An ontology (retrieved from context) that defines the allowed vocabulary: classes, properties, and relations.

    ## TASK ##
    Using the environment description and the task, you must generate an **ordered sequence of robot actions** that would complete the task in that environment.

    You must express these actions as RDF triples in **Turtle format**, strictly conforming to the ontology.

    ### Constraints:
    - Use **only** the classes, properties, and relations defined in the ontology.
    - Do **not invent** new terms.
    - Ensure that each action is:
    - A single, atomic, physically valid step
    - Referencing objects and spatial relationships explicitly present in the environment
    - Represented using correct ontology semantics (e.g., action types, agents, targets, tools, locations)

    ## OUTPUT FORMAT ##
    - Output only text, with no extra explanations.
    - Output must consist of triples in Turtle format.
    - Organize your output into a single clearly labeled section:

    ### Triples from Robot Actions ###
    (triples expressing each ordered robot action)

    ## INPUT ##
    ### Environment Description ###
    {description}

    ### Robot Task ###
    {robot_task}
    """)
    return response_obs, response_acs, chroma_collection

def trim_before_prefixes(text):
    match = re.search(r'(^|\n)(@prefix|PREFIX)\s', text)
    if match:
        return text[match.start():]
    else:
        return text

def save_llm_kg_response(response_obs, response_acs, output_path):
    output_path_og = make_output_file_path(output_path, "observationGraph", "kg.ttl")
    output_path_ag = make_output_file_path(output_path, "actionGraph", "kg.ttl")

    response_obs_text = trim_before_prefixes(str(response_obs))
    response_acs_text = trim_before_prefixes(str(response_acs))

    with open(output_path_og, 'w', encoding='utf-8') as f:
        f.write(response_obs_text)

    with open(output_path_ag, 'w', encoding='utf-8') as f:
        f.write(response_acs_text)
    return 

def clean_up_chroma(chroma_collection):
    doc_ids = chroma_collection.get()["ids"]
    chroma_collection.delete(ids=doc_ids)
    return

def main(llm_model, api_key, robot_task, embedding_model, description_path, output_path, ontology_path):
    llm = Groq(model=llm_model, api_key=api_key)
    description = extract_description_from_json(description_path)
    response_obs, response_acs, chroma_collection = get_response(llm, embedding_model, description, ontology_path, robot_task)
    save_llm_kg_response(response_obs, response_acs, output_path)
    clean_up_chroma(chroma_collection)

if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    api_key = os.getenv("GROQ_KEY")
    robot_task = os.getenv("ROBOT_TASK")
    embedding_model = "BAAI/bge-small-en"

    description_path = "../../output/vlm_vlm/llama-image-description.json"
    output_path = "../../output/vlm_vlm/ragKG"
    ontology_path = "../../ontology/onto.jsonld"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main(llm_model, api_key, robot_task, embedding_model, description_path, output_path, ontology_path)
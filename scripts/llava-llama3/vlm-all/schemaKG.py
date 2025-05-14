import os
import re
import json
import shutil

from pathlib import Path
from dotenv import load_dotenv
from pyvis.network import Network
from rdflib import Graph, URIRef, Literal, Namespace

from typing import Literal
from llama_index.llms.groq import Groq
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings


def get_llm(api_key, llm_model):
    llm = Groq(model= llm_model, api_key=api_key)
    return llm

def extract_sections_from_json(description_path):
    with open(description_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    content = data['choices'][0]['message']['content']
    sections = content.strip().split('\n\n')
    section1 = sections[0].strip()
    section2 = sections[1].strip()
    return section1, section2

def make_kg_extractor(llm):
    kg_extractor = SchemaLLMPathExtractor(
        llm=llm,
        max_triplets_per_chunk=50,
        num_workers=4,
        strict=False,
        possible_entities=None,
        possible_relations=None,
        possible_relation_props=None,
        possible_entity_props=None,
    )
    return kg_extractor

def make_schema_index(description_path, llm, kg_extractor):
    section1, section2 = extract_sections_from_json(description_path)
    document1 = Document(text=section1)
    document2 = Document(text=section2)
    # schema index section 1
    schema_index_1 = PropertyGraphIndex.from_documents(
        [document1],
        llm=llm,
        embed_kg_nodes=False,
        kg_extractors=[kg_extractor],
        show_progress=True,
    )

    # schema index section 2
    schema_index_2 = PropertyGraphIndex.from_documents(
        [document2],
        llm=llm,
        embed_kg_nodes=False,
        kg_extractors=[kg_extractor],
        show_progress=True,
    )

    return schema_index_1, schema_index_2

def make_output_dir(output_path, dir_name):
    full_output_path = os.path.join(output_path, dir_name)
    if os.path.exists(full_output_path):
        shutil.rmtree(full_output_path)
    os.makedirs(full_output_path)
    return full_output_path

def save_index(schema_index, full_output_path):
    schema_index.storage_context.persist(persist_dir=full_output_path)
    return

def save_turtle(full_output_path):
    json_path = os.path.join(full_output_path, "property_graph_store.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    triplets = data.get("triplets", [])
    g = Graph()
    ns = Namespace("http://example.org/kitchen#")
    g.bind("kitchen", ns)
    for subj, pred, obj in triplets:
        subj_uri = ns[subj.replace(" ", "_")]
        pred_uri = ns[pred.replace(" ", "_")]
        obj_uri  = ns[obj.replace(" ", "_")]
        g.add((subj_uri, pred_uri, obj_uri))
    output_path_turtle = os.path.join(full_output_path, "kg.ttl")
    g.serialize(destination=output_path_turtle, format="turtle")
    return

def save_network_graph(schema_index, full_output_path):
    output_path_network_graph = os.path.join(full_output_path, "kg.html")
    schema_index.property_graph_store.save_networkx_graph(
        name=output_path_network_graph
    )
    return 

def main(llm_model, api_key, description_path, output_path):
    llm = get_llm(api_key, llm_model)
    output_path_og = make_output_dir(output_path, "observationGraph")
    output_path_ag = make_output_dir(output_path, "actionGraph")

    kg_extractor = make_kg_extractor(llm)
    schema_index_1, schema_index_2 = make_schema_index(description_path, llm, kg_extractor)

    save_index(schema_index_1, output_path_og)
    save_index(schema_index_2, output_path_ag)

    save_turtle(output_path_og)
    save_turtle(output_path_ag)

    save_network_graph(schema_index_1, output_path_og)
    save_network_graph(schema_index_2, output_path_ag)

if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    api_key = os.getenv("GROQ_KEY")
    description_path = "../../../output/llava-llama3/vlm-all/llama-image-description.json"
    output_path = "../../../output/llava-llama3/vlm-all/schemaKG"

    main(llm_model, api_key, description_path, output_path)
    
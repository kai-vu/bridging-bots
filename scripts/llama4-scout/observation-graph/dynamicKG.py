import os
import re
import json
import shutil

from pathlib import Path
from dotenv import load_dotenv
from rdflib import Graph, URIRef, Literal, Namespace

from llama_index.llms.groq import Groq
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings


def get_llm(api_key, llm_model):
    llm = Groq(model=llm_model, api_key=api_key)
    return llm

def define_schema():
    entities = ["Environment", "Component", "Appliance", "Furniture", "Object", 
                "Location", "Affordance", "Closing", "Opening", "Delivering",
                "Holding", "PickingUp", "PuttingDown", "Pulling", "Pushing",
                "Grasping"]
    relations = ["isA", "label", "hasComponent", "hasAffordance", "hasLocation", 
                 "onTopOf", "sfContains", "sfWithin", "sfOverlaps"]
    return entities, relations

def extract_description_from_json(description_path):
    with open(description_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    content = data['choices'][0]['message']['content']
    return content

def make_kg_extractor(llm):
    entities, relations = define_schema()
    kg_extractor = DynamicLLMPathExtractor(
        llm=llm,
        max_triplets_per_chunk=50,
        num_workers=4,
        allowed_entity_types=entities,
        allowed_entity_props=["isA", "label", "comment"],
        allowed_relation_types=relations,
        allowed_relation_props=["isA", "label", "domain", "range"],
        #kg_validation_schema=schema,
    )
    return kg_extractor

def make_dynamic_index(description_path, llm, kg_extractor):
    content = extract_description_from_json(description_path)
    document = Document(text=content)
    dynamic_index = PropertyGraphIndex.from_documents(
        [document],
        llm=llm,
        embed_kg_nodes=False,
        kg_extractors=[kg_extractor],
        show_progress=True,
    )
    return dynamic_index

def save_index(dynamic_index, full_output_path):
    dynamic_index.storage_context.persist(persist_dir=full_output_path)
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

def save_network_graph(dynamic_index, full_output_path):
    output_path_network_graph = os.path.join(full_output_path, "kg.html")
    dynamic_index.property_graph_store.save_networkx_graph(
        name=output_path_network_graph
    )
    return 

def main(llm_model, api_key, description_path, output_path):
    llm = get_llm(api_key, llm_model)

    kg_extractor = make_kg_extractor(llm)
    dynamic_index = make_dynamic_index(description_path, llm, kg_extractor)

    save_index(dynamic_index, output_path)
    save_turtle(output_path)
    save_network_graph(dynamic_index, output_path)

if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    api_key = os.getenv("GROQ_KEY")
    description_path = "../../../output/llama4-scout/observation-graph/image-description.json"
    output_path = "../../../output/llama4-scout/observation-graph/dynamicKG"

    main(llm_model, api_key, description_path, output_path)
    
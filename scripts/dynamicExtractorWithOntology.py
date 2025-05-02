import os
import json
from pathlib import Path

from dotenv import load_dotenv
from rdflib import Graph, URIRef, Literal, Namespace

from llama_index.llms.groq import Groq
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings


def get_llm(api_key, llm_model):
    llm = Groq(model= llm_model, api_key=api_key)
    return llm

def define_schema():
    entities = ["Environment", "Component", "Location", "Appliance", 
                "Furniture", "Object", "Affordance", "Closing", "Opening",
                "Pulling", "Pushing", "Dropping", "PickingUp", "PuttingDown", 
                "Grasping", "Action", "Task", "Workflow", "Instruction"]
    relations = ["isInstanceOf", "hasComponent", "hasLocation", "sfContains", 
                 "sfWithin", "sfOverlaps", "hasAffordance", "isAffordedBy", 
                 "actsOn", "isExecutedIn", "precedes", "follows", "hasTask", 
                 "hasWorkflow", "hasNaturalLanguage"]
    schema = {
        "Environment": ["hasComponent"],
        "Component": ["hasLocation", "hasAffordance"],
        "Location": ["sfContains", "sfWithin", "sfOverlaps"],
        "Appliance": ["isInstanceOf"],
        "Furniture": ["isInstanceOf"],
        "Object": ["isInstanceOf"],
        "Affordance": ["label", "comment"],
        "Closing": ["isInstanceOf"],
        "Opening": ["isInstanceOf"],
        "Pulling": ["isInstanceOf"],
        "Pushing": ["isInstanceOf"],
        "Dropping": ["isInstanceOf"],
        "PickingUp": ["isInstanceOf"],
        "PuttingDown": ["isInstanceOf"],
        "Grasping": ["isInstanceOf"],
        "Action": ["isAffordedBy", "actsOn"],
        "Task": ["isExecutedIn", "precedes", "follows"],
        "Workflow": ["hasTask"],
        "Instruction": ["hasWorkflow", "hasNaturalLanguage"]
    }
    return entities, relations, schema

def make_kg_extractor(llm):
    entities, relations, schema = define_schema()
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
    with open(description_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    content = data['choices'][0]['message']['content']
    document = Document(text=content)
    dynamic_index = PropertyGraphIndex.from_documents(
        [document],
        llm=llm,
        embed_kg_nodes=False,
        kg_extractors=[kg_extractor],
        show_progress=True,
    )
    return dynamic_index

def save_index(dynamic_index, output_path_index):
    dynamic_index.storage_context.persist(persist_dir=output_path_index)
    return

def save_turtle(output_path_index, output_path_turtle):
    json_path = Path(output_path_index+"/property_graph_store.json")
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
    g.serialize(destination=output_path_turtle, format="turtle")
    print(f"Turtle file saved to: {output_path_turtle}")
    return

def save_network_graph(dynamic_index, output_path_network_graph):
    dynamic_index.property_graph_store.save_networkx_graph(
        name=output_path_network_graph
    )
    return 

def main(llm_model, api_key, description_path, output_path_index, output_path_turtle, output_path_network_graph):
    llm = get_llm(api_key, llm_model)
    kg_extractor = make_kg_extractor(llm)
    dynamic_index = make_dynamic_index(description_path, llm, kg_extractor)
    save_index(dynamic_index, output_path_index)
    save_turtle(output_path_index, output_path_turtle)
    save_network_graph(dynamic_index, output_path_network_graph)

if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(current_dir, ".env")
    load_dotenv(dotenv_path)

    llm_model = os.getenv("LLM_MODEL")
    api_key = os.getenv("GROQ_KEY")
    description_path = "../output/llama-image-description.json"
    output_path_index = "../output/dynamicWithOntology/dynamic_index"
    output_path_turtle = "../output/dynamicWithOntology/dynamicWithOntology.ttl"
    output_path_network_graph = "../output/dynamicWithOntology/dynamicWithOntology.html"

    main(llm_model, api_key, description_path, output_path_index, output_path_turtle, output_path_network_graph)
    
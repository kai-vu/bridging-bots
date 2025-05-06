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
from llama_index.core.chat_engine import SimpleChatEngine


def get_llm(api_key, llm_model):
    llm = Groq(model= llm_model, api_key=api_key)
    return llm

def extract_description_from_json(description_path):
    with open(description_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    description = data['choices'][0]['message']['content']
    return description

def make_output_dir(output_path, dir_name):
    full_output_path = os.path.join(output_path, dir_name)
    if os.path.exists(full_output_path):
        shutil.rmtree(full_output_path)
    os.makedirs(full_output_path)
    return full_output_path

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

def make_dynamic_index(text, llm, kg_extractor):
    document = Document(text=text)
    dynamic_index = PropertyGraphIndex.from_documents(
        [document],
        llm=llm,
        embed_kg_nodes=False,
        kg_extractors=[kg_extractor],
        show_progress=True,
    )
    return dynamic_index

def get_response(llm, robot_task, description):
    chat_engine = SimpleChatEngine.from_defaults(llm=llm)
    prompt = f"""
    ## INSTRUCTIONS ##
    You are given a detailed description of an environment, as observed from multiple images.  
    This environment is the context in which a robot must perform a specific task.

    Use only the information provided in the environment description to generate an ordered list of physical actions the robot must take to complete the task.  
    Each action must reference specific objects and spatial relationships mentioned in the environment description.  
    Do not infer or invent objects, locations, or relationships that are not explicitly described.

    The robot's task is: [{robot_task}]

    ## INPUT ##
    {description}

    ## OUTPUT FORMAT ##
    You must return only text output, with no introductory text or explanations.  
    Return the robot's action in sequential order. Each action must be a single, atomic and physically valid action, grounded in the environment.
    """
    response = chat_engine.chat(prompt)
    return response

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

def main(llm_model, api_key, robot_task, description_path, output_path):
    llm = get_llm(api_key, llm_model)
    output_path_og = make_output_dir(output_path, "observationGraph")
    output_path_ag = make_output_dir(output_path, "actionGraph")

    kg_extractor = make_kg_extractor(llm)
    description = extract_description_from_json(description_path)
    response = get_response(llm, robot_task, description)

    dynamic_index_obs = make_dynamic_index(description, llm, kg_extractor)
    dynamic_index_acs = make_dynamic_index(str(response), llm, kg_extractor)

    save_index(dynamic_index_obs, output_path_og)
    save_index(dynamic_index_acs, output_path_ag)

    save_turtle(output_path_og)
    save_turtle(output_path_ag)

    save_network_graph(dynamic_index_obs, output_path_og)
    save_network_graph(dynamic_index_acs, output_path_ag)

if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    api_key = os.getenv("GROQ_KEY")
    robot_task = os.getenv("ROBOT_TASK")

    description_path = "../../output/vlm_llm/llama-image-description.json"
    output_path = "../../output/vlm_llm/dynamicKG"

    main(llm_model, api_key, robot_task, description_path, output_path)
    
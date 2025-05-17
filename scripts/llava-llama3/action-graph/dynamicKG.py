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
from llama_index.core.llms import ChatMessage


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

def get_response(llm, user_query):
    messages = [
        ChatMessage(role="user", content=user_query),
    ]
    response = llm.chat(messages)
    return response

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
    )
    return kg_extractor

def make_dynamic_index(groq_key, user_query, llm, kg_extractor):
    content = get_response(llm, user_query)
    document = Document(text=str(content))
    dynamic_index = PropertyGraphIndex.from_documents(
        [document],
        llm=llm,
        embed_kg_nodes=False,
        kg_extractors=[kg_extractor],
        show_progress=True,
    )
    return dynamic_index

def save_index(dynamic_index, output_path):
    dynamic_index.storage_context.persist(persist_dir=output_path)
    return

def save_turtle(output_path):
    json_path = os.path.join(output_path, "property_graph_store.json")
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
    output_path_turtle = os.path.join(output_path, "kg.ttl")
    g.serialize(destination=output_path_turtle, format="turtle")
    return

def save_network_graph(dynamic_index, output_path):
    output_path_network_graph = os.path.join(output_path, "kg.html")
    dynamic_index.property_graph_store.save_networkx_graph(
        name=output_path_network_graph
    )
    return 

def main(llm_model, groq_key, user_query, output_path):
    llm = get_llm(groq_key, llm_model)

    kg_extractor = make_kg_extractor(llm)
    dynamic_index = make_dynamic_index(groq_key, user_query, llm, kg_extractor)

    save_index(dynamic_index, output_path)
    save_turtle(output_path)
    save_network_graph(dynamic_index, output_path)

if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    groq_key = os.getenv("GROQ_KEY")
    robot_task = os.getenv("ROBOT_TASK")

    description_path = "../../../output/llava-llama3/observation-graph/image-description.json"
    with open(description_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    description_txt = data["choices"][0]["message"]["content"]

    output_path = "../../../output/llava-llama3/action-graph/dynamicKG"

    user_query = f"""
You are an intelligent assistant tasked to generate a the sequence of actions a robot must perform to accomplish the following task in the environment described below**:
---------------------
TASK: {robot_task}

ENVIRONMENT DESCRIPTION: {description_txt}
---------------------

You are given a text description of the environment where the robot must perform the task.

Instructions:
- Analyze the the environment description carefully to understand the complete layout of the environment, objects, and relevant affordances.
- Generate the sequence of actions required for the robot to complete the task**.
- Each action is a **single, atomic, clear action**.

Output format:
- Return only the generated actions.
- Output only text, no extra explanations.
"""

    main(llm_model, groq_key, user_query, output_path)
    
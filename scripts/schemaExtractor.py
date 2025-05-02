import os
import json
from pathlib import Path

from dotenv import load_dotenv

from typing import Literal
from llama_index.llms.groq import Groq
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

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
    kg_extractor = SchemaLLMPathExtractor(
        llm=llm,
        max_triplets_per_chunk=50,
        num_workers=4,
        strict=True,
        possible_entities=entities,
        possible_relations=relations,
        kg_validation_schema=schema,
    )
    return kg_extractor

def make_schema_index(description_path, llm, kg_extractor):
    with open(description_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    content = data['choices'][0]['message']['content']
    document = Document(text=content)
    schema_index = PropertyGraphIndex.from_documents(
        [document],
        llm=llm,
        embed_kg_nodes=False,
        kg_extractors=[kg_extractor],
        show_progress=True,
    )

    return schema_index

def save_graph(schema_index, output_path):
    schema_index.property_graph_store.save_networkx_graph(
        name=output_path
    )
    return 

def main(llm_model, api_key, description_path, output_path):
    llm = get_llm(api_key, llm_model)
    kg_extractor = make_kg_extractor(llm)
    schema_index = make_schema_index(description_path, llm, kg_extractor)
    save_graph(schema_index, output_path)


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(current_dir, ".env")
    load_dotenv(dotenv_path)

    llm_model = os.getenv("LLM_MODEL")
    api_key = os.getenv("GROQ_KEY")
    description_path = "../output/llama-image-description.json"
    output_path = os.path.join(os.path.split(description_path)[0], 'llama_graph_schemaExtractor.html')

    main(llm_model, api_key, description_path, output_path)
    
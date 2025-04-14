import os
import base64

from dotenv import load_dotenv

from typing import Literal
from llama_index.llms.groq import Groq
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    SchemaLLMPathExtractor,
    DynamicLLMPathExtractor,
)
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings


def get_llm(api_key, llm_model):
    llm = Groq(model= llm_model, api_key=api_key)
    return llm

def define_schema():
    entities = Literal["KitchenEnvironment", "Object", "Component", "ObjectsStack"]
    relations = Literal["isA", "label", "comment", "domain", "range", "hasObject", "hasComponent", "hasCurrentLocation", "hasTargetLocation", "hasStackObject", "hasOrderInStack"]
    schema = {
        "KitchenEnvironment": ["isA", "label", "comment"],
        "Object": ["isA", "label", "comment"],
        "Component": ["isA", "label", "comment"],
        "ObjectsStack": ["isA", "label", "comment"],
        "hasObject": ["isA", "label", "comment", "domain", "range"],
        "hasComponent": ["isA", "label", "comment", "domain", "range"],
        "hasCurrentLocation": ["isA", "label", "comment", "domain", "range"],
        "hasTargetLocation": ["isA", "label", "comment", "domain", "range"],
        "hasStackObject": ["isA", "label", "comment", "domain", "range"],
        "hasOrderInStack": ["isA", "label", "comment", "domain", "range"],
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
    content = open(description_path, 'r', encoding='utf-8').read()
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
    api_key = os.getenv("API_KEY")
    description_path = os.getenv("OUTPUT_PATH")
    output_path = os.path.join(os.path.split(description_path)[0], 'llama_graph_schemaExtractor.html')

    main(llm_model, api_key, description_path, output_path)
    
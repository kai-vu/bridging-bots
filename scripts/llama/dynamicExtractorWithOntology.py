import os
import base64

from dotenv import load_dotenv

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

def make_kg_extractor(llm):
    kg_extractor = DynamicLLMPathExtractor(
        llm=llm,
        max_triplets_per_chunk=50,
        num_workers=4,
        allowed_entity_types=["KitchenEnvironment", "Object", "Component", "ObjectsStack"],
        allowed_entity_props=["label", "comment"],
        allowed_relation_types=["hasObject", "hasComponent", "hasCurrentLocation", "hasTargetLocation", "hasStackObject", "hasOrderInStack"],
        allowed_relation_props=["label", "domain", "range"],
    )
    return kg_extractor

def make_dynamic_index(description_path, llm, kg_extractor):
    content = open(description_path, 'r', encoding='utf-8').read()
    document = Document(text=content)
    dynamic_index = PropertyGraphIndex.from_documents(
        [document],
        llm=llm,
        embed_kg_nodes=False,
        kg_extractors=[kg_extractor],
        show_progress=True,
    )
    return dynamic_index

def save_graph(dynamic_index, output_path):
    dynamic_index.property_graph_store.save_networkx_graph(
        name=output_path
    )
    return 

def main(llm_model, api_key, description_path, output_path):
    llm = get_llm(api_key, llm_model)
    kg_extractor = make_kg_extractor(llm)
    dynamic_index = make_dynamic_index(description_path, llm, kg_extractor)
    save_graph(dynamic_index, output_path)


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(current_dir, ".env")
    load_dotenv(dotenv_path)

    llm_model = os.getenv("LLM_MODEL")
    api_key = os.getenv("API_KEY")
    description_path = os.getenv("OUTPUT_PATH")
    output_path = os.path.join(os.path.split(description_path)[0], 'llama_graph_dynamicExtractorWithOntology.html')

    main(llm_model, api_key, description_path, output_path)
    
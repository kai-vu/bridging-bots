import os
import re
import json
import shutil

from openai import OpenAI
from pyld import jsonld
from pathlib import Path
from dotenv import load_dotenv


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

def parse_ontology(ontology_json):
    class_labels = []
    predicate_labels = []
    for entry in ontology_json:
        types = entry.get("@type", [])
        labels = [l["@value"] for l in entry.get("http://www.w3.org/2000/01/rdf-schema#label", [])]
        if not labels:
            continue
        if any("Class" in t for t in types):
            class_labels.extend(labels)
        elif any("Property" in t or "ObjectProperty" in t or "DatatypeProperty" in t for t in types):
            predicate_labels.extend(labels)
    return class_labels, predicate_labels

def build_system_prompt(ontology_path):
    with open(ontology_path) as f:
        ontology = json.load(f)
    classes, predicates = parse_ontology(ontology)
    system_prompt = f"""
    You are a semantic annotator that extracts RDF triples from natural language.
    Only use the following RDF classes: {', '.join(classes)}.
    Only use the following predicates: {', '.join(predicates)}.
    Return triples in the format:
    [
    {{ "subject": "...", "predicate": "...", "object": "..." }}
    ]
    """
    return system_prompt

def chat_with_model(gpt_key, llm_model, ontology_path, description_path):
    client = OpenAI(api_key=gpt_key)
    system_prompt = build_system_prompt(ontology_path)
    section1, section2 = extract_sections_from_json(description_path)
    user_query = f"""
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
    """
    response = client.chat.completions.create(
            model=llm_model,
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_query }
            ]
        )
    response = response.choices[0].message.content
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

def main(gpt_key, llm_model, output_path, ontology_path, description_path):
    response = chat_with_model(gpt_key, llm_model, ontology_path, description_path)
    save_llm_kg_response(response, output_path)


if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    gpt_key = os.getenv("GPT_KEY")

    description_path = "../../../output/gpt4-nano/llm-all/image-description.json"
    output_path = "../../../output/gpt4-nano/llm-all/ragKG"
    ontology_path = "../../../ontology/onto.jsonld"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main(gpt_key, llm_model, output_path, ontology_path, description_path)
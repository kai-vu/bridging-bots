import warnings
warnings.filterwarnings('ignore')

import os
import re
import json

from pathlib import Path
from dotenv import load_dotenv
from groq import Groq


def extract_description_from_json(description_path):
    with open(description_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    description_txt = data['choices'][0]['message']['content']
    return description_txt

def extract_ontology_text(ontology_path):
    with open(ontology_path, 'r', encoding='utf-8') as file:
        ontology_txt = file.read()
    return ontology_txt

def make_user_query(ontology_path, description_path, prompt_tmpl):
    ontology_txt = extract_ontology_text(ontology_path)
    description_txt = extract_description_from_json(description_path)
    user_query = prompt_tmpl.format(
        ontology_txt=ontology_txt ,
        description_txt=description_txt ,
    )
    return user_query

def get_response(ontology_path, description_path, prompt_tmpl, llm_model, client):
    user_query = make_user_query(ontology_path, description_path, prompt_tmpl)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_query
            }
        ],
        model=llm_model,
    )
    response = chat_completion
    return response

def save_response(response, output_path):
    response_usage_path = os.path.join(output_path, "response_usage.json")
    response_usage = response.usage.dict()
    with open(response_usage_path, 'w', encoding='utf-8') as file:
        json.dump(response_usage, file, indent=4, ensure_ascii=False)
    ttl_response_path = os.path.join(output_path, "kg.ttl")
    ttl_response = response.choices[0].message.content
    ttl_response = str(ttl_response)
    ttl_response = re.sub(r"^```[^\n]*\n", "", ttl_response.strip())
    ttl_response = re.sub(r"```$", "", ttl_response.strip())
    ttl_response = ttl_response.strip()
    with open(ttl_response_path, 'w') as f:
        f.write(ttl_response)
    return 

def main(llm_model, groq_key, ontology_path, description_path, prompt_tmpl, output_path):
    client = Groq(api_key=groq_key)
    response = get_response(ontology_path, description_path, prompt_tmpl, llm_model, client)
    save_response(response, output_path)

if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    groq_key = os.getenv("GROQ_KEY")
    embedding_model = "BAAI/bge-small-en"

    description_path = "../../../output/llava-llama3/observation-graph/image-description.json"
    output_path = "../../../output/llava-llama3/observation-graph/d2kg"
    ontology_path = "../../../ontology/ontoObservationGraph.ttl"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    prompt_tmpl = """
Ontology as context information is below.
---------------------
{ontology_txt}
---------------------

You are an intelligent assistant tasked to generate a **Knowledge Graph of the environment described as follows**:
---------------------
ENVIRONMENT DESCRIPTION: {description_txt}
---------------------

Instructions:
- Analyze the description carefully to understand the complete layout of the environment.
- Based on the ontology, **generate a Knowledge Graph describing the environment**.
- All entities and relations must conform to the structure and semantics of the ontology.
- **Use only classes and properties from the ontology.**
- Do **NOT invent or infer any terms or actions outside of the ontology schema.**

Output format:
- Return only the generated Knowledge Graph.
- Output only text, no extra explanations.
- Use Turtle format for the output, such as <subject> <predicate> <object> .
- Include all prefixes and namespaces at the beginning. 
- Use the ex: prefix with namespace <http://example.org/data/> only for newly instantiated entities instantiated, such as specific actions, objects, or locations.
- Do not use the ex: prefix for ontology classes, properties, or schema definitions, those must strictly come from the provided ontology with their original prefixes and namespaces.
"""

    main(llm_model, groq_key, ontology_path, description_path, prompt_tmpl, output_path)
import warnings
warnings.filterwarnings('ignore')

import os
import re
import json

from pathlib import Path
from dotenv import load_dotenv
from groq import Groq


def get_response(llm_model, client, user_query):
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

def main(llm_model, groq_key, user_query, output_path):
    client = Groq(api_key=groq_key)
    response = get_response(llm_model, client, user_query)
    save_response(response, output_path)


if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    groq_key = os.getenv("GROQ_KEY")
    robot_task = os.getenv("ROBOT_TASK")

    description_path = "../../../output/llama4-maverick/observation-graph/image-description.json"
    with open(description_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    description_txt = data["choices"][0]["message"]["content"]

    ontology_path = "../../../ontology/ontoActionGraph.ttl"
    with open(ontology_path, "r", encoding="utf-8") as file:
        ontology_txt = file.read()

    output_path = "../../../output/llama4-maverick/action-graph/d2kg"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    user_query = f"""
Ontology as context information is below.
---------------------
{ontology_txt}
---------------------

You are an intelligent assistant tasked to generate a **Knowledge Graph of the sequence of actions a robot must perform to accomplish the following task**:
---------------------
ROBOT TASK: {robot_task}
---------------------

You are provided with text description of an environment, below:
---------------------
ENVIRONMENT DESCRIPTION: {description_txt}
---------------------

Instructions:
- Analyze the description carefully to understand the complete layout of the environment.
- Based on the ontology, **generate the sequence of actions required for the robot to complete the task**.
- Each action is a **single, atomic, clear action**.
- **All actions, entities, and relationships must strictly follow the provided ontology.**
- **Use only classes and properties from the ontology.**
- Do **NOT invent or infer any terms or actions outside of the ontology schema.**
- The graph should represent actions, objects involved, and their relations according to the ontology's structure and semantics.

Output format:
- Return only the generated Knowledge Graph of actions.
- Output only text, no extra explanations.
- Use Turtle format for the output, such as <subject> <predicate> <object> .
- Include all prefixes and namespaces at the beginning. 
- Use the ex: prefix with namespace <http://example.org/data/> only for newly instantiated entities instantiated, such as specific actions, objects, or locations.
- Do not use the ex: prefix for ontology classes, properties, or schema definitions, those must strictly come from the provided ontology with their original prefixes and namespaces.
"""

    main(llm_model, groq_key, user_query, output_path)
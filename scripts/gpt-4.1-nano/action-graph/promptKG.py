import os
import re
import json
import base64

from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv


def get_response(user_query, llm_model, client):
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

def main(gpt_key, user_query, llm_model):
    client = OpenAI(api_key=gpt_key)
    response = get_response(user_query, llm_model, client)
    save_response(response, output_path)


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=Path('../.env'))

    gpt_key = os.getenv("GPT_KEY")
    llm_model = os.getenv("LLM_MODEL")
    robot_task = os.getenv("ROBOT_TASK")

    description_path = "../../../output/gpt-4.1-nano/observation-graph/image-description.json"
    with open(description_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    description_txt = data["choices"][0]["message"]["content"]
    output_path = "../../../output/gpt-4.1-nano/action-graph/promptKG"
    ontology_path = "../../../ontology/ontoActionGraph.ttl"
    with open(ontology_path, 'r', encoding='utf-8') as file:
        ontology_txt = file.read()

    user_query = f"""
Ontology as context information is below.
---------------------
{ontology_txt}
---------------------


Given the ontology information, your task is to generates a Knowledge Graph of the actions a robot must complete to fulfil a task in an environment.
You are provided with text description of the environment and the task, found below. 

You must use the ontology **as a strict schema** to construct the Knowledge Graph.
This means:
- Use **only** the classes and properties defined in the ontology.
- Do **not invent or infer** terms not explicitly defined in the ontology.
- All entities and relations must conform to the structure and semantics of the ontology.

Output format:
- Output only text, with no extra explanations.
- Output must consist of triples in turtle format.
- Output must contain the prefixes and namespaces.

---------------------
TASK: {robot_task}

ENVIRONMENT DESCRIPTION: {description_txt}
---------------------
"""
        
    main(gpt_key, user_query, llm_model)
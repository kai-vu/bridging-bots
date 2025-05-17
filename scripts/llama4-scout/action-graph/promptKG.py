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

    description_path = "../../../output/llama4-scout/observation-graph/image-description.json"
    with open(description_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    description_txt = data["choices"][0]["message"]["content"]

    ontology_path = "../../../ontology/ontoActionGraph.ttl"
    with open(ontology_path, "r", encoding="utf-8") as file:
        ontology_txt = file.read()

    output_path = "../../../output/llama4-scout/action-graph/promptKG"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    main(llm_model, groq_key, user_query, output_path)
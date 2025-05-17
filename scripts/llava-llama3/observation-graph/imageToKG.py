import os
import re
import json
import base64
import requests

from dotenv import load_dotenv
from pathlib import Path


def convert_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_all_images_base64(images_folder_path):
    base64_images = []
    for filename in os.listdir(images_folder_path):
        file_path = os.path.join(images_folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            base64_images.append(convert_image_to_base64(file_path))
    return base64_images

def chat_with_model(nebula_key, nebula_url, vlm_model, user_query, images_folder_path):
    url = nebula_url
    headers = {
        'Authorization': f'Bearer {nebula_key}',
        'Content-Type': 'application/json'
    }

    base64_images = get_all_images_base64(images_folder_path)
    content = [{"type": "text", "text": user_query}]
    for base64_image in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    data = {
        "model": vlm_model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    return response_json

def save_response_to_file(output_path, response_json):
    json_response_path = os.path.join(output_path, "response.json")
    with open(json_response_path, 'w', encoding='utf-8') as f:
        json.dump(response_json, f, ensure_ascii=False, indent=4)
    ttl_response_path = os.path.join(output_path, "kg.ttl")
    ttl_response = str(response_json['choices'][0]['message']['content'])
    ttl_response = re.sub(r"^```[^\n]*\n", "", ttl_response.strip())
    ttl_response = re.sub(r"```$", "", ttl_response.strip())
    ttl_response = ttl_response.strip()
    with open(ttl_response_path, 'w') as f:
        f.write(ttl_response)
    return 

def main(images_folder_path, nebula_key, nebula_url, vlm_model, user_query, output_path):
    response_json = chat_with_model(nebula_key, nebula_url, vlm_model, user_query, images_folder_path)
    save_response_to_file(output_path, response_json)


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=Path('../.env'))

    nebula_key = os.getenv("NEBULA_KEY")
    nebula_url = os.getenv("NEBULA_URL")
    vlm_model = os.getenv("VLM_MODEL")

    images_folder_path = "../../../images"
    output_path = "../../../output/llava-llama3/observation-graph/imageToKG"
    ontology_path = "../../../ontology/ontoObservationGraph.ttl"
    with open(ontology_path, 'r', encoding='utf-8') as file:
        ontology_txt = file.read()

    user_query = f"""
Ontology as context information is below.
---------------------
{ontology_txt}
---------------------

Given the ontology information, your task is to generates a Knowledge Graph from a set of images.
The images are taken from the same environment at different angles. 
These images together represent the complete layout and state of the environment.

Instructions:
- Analyze the images carefully to understand the complete layout of the environment, objects, and relevant affordances.
- Based on the ontology, **generate a Knowledge Graph describing the environment**.
- Use **only** the classes and properties defined in the ontology.
- Do **not invent or infer** terms not explicitly defined in the ontology.
- All entities and relations must conform to the structure and semantics of the ontology.

Output format:
- Output only text, no extra explanations.
- Use Turtle format for the output.
- Include the prefixes and namespaces at the beginning.
"""

    main(images_folder_path, nebula_key, nebula_url, vlm_model, user_query, output_path)
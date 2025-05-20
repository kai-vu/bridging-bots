import os
import re
import json
import base64
import requests

from groq import Groq
from pathlib import Path
from dotenv import load_dotenv


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

def chat_with_model(groq_key, images_folder_path, user_query, llm_model):
    client = Groq(api_key=groq_key)
    base64_images = get_all_images_base64(images_folder_path)
    content = [{"type": "text", "text": user_query}]
    for base64_image in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": content,
        }
    ],
    model=llm_model,
    )
    response = chat_completion
    return response

def save_response_to_file(output_path, response):
    json_response_path = os.path.join(output_path, "response.json")
    with open(json_response_path, 'w', encoding='utf-8') as f:
        json.dump(response.to_dict(), f, ensure_ascii=False, indent=4)
    ttl_response_path = os.path.join(output_path, "kg.ttl")
    ttl_response = response.choices[0].message.content
    ttl_response = re.sub(r"^```[^\n]*\n", "", ttl_response.strip())
    ttl_response = re.sub(r"```$", "", ttl_response.strip())
    ttl_response = ttl_response.strip()
    with open(ttl_response_path, 'w') as f:
        f.write(ttl_response)
    return 

def main(images_folder_path, groq_key, llm_model, user_query, output_path):
    response = chat_with_model(groq_key, images_folder_path, user_query, llm_model)
    save_response_to_file(output_path, response)


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=Path('../.env'))

    groq_key = os.getenv("GROQ_KEY")
    llm_model = os.getenv("LLM_MODEL")
    robot_task = os.getenv("ROBOT_TASK")

    images_folder_path = "../../../images"
    output_path = "../../../output/llama4-scout/action-graph/i2kg"
    ontology_path = "../../../ontology/ontoActionGraph.ttl"
    with open(ontology_path, 'r', encoding='utf-8') as file:
        ontology_txt = file.read()

    user_query = f"""
Ontology as context information is below.
---------------------
{ontology_txt}
---------------------

You are an intelligent assistant tasked to generate a **Knowledge Graph of the sequence of actions a robot must perform to accomplish the following task**:
---------------------
TASK: {robot_task}
---------------------

You are given a set of images taken from the same environment at different angles. 
These images together represent the complete layout and state of the environment in which a robot must perform a task.

Instructions:
- Analyze the images carefully to understand the complete layout of the environment, objects, and relevant affordances.
- Based on the ontology, **generate the sequence of actions required for the robot to complete the task**.
- Each action is a **single, atomic, clear action**.
- **All actions, entities, and relationships must strictly follow the provided ontology.**
- **Use only classes and properties from the ontology.**
- Do **NOT invent or infer any terms or actions outside of the ontology schema.**
- The graph should represent actions, objects involved, and their relations according to the ontology's structure and semantics.

Output format:
- Return only the generated Knowledge Graph of actions.
- Output only text, no extra explanations.
- Use Turtle format for the output.
- Include the prefixes and namespaces at the beginning.
"""

    main(images_folder_path, groq_key, llm_model, user_query, output_path)
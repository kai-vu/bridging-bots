import os
import json
import base64

from openai import OpenAI
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

def chat_with_model(gpt_key, images_folder_path, task_description, llm_model, system_prompt):
    client = OpenAI(api_key=gpt_key)
    base64_images = get_all_images_base64(images_folder_path)
    content = [{"type": "text", "text": task_description}]
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
            "role": "system",
            "content": system_prompt,
        },
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
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(response.to_dict(), f, ensure_ascii=False, indent=4)
    return 

def main(gpt_key, images_folder_path, task_description, llm_model, system_prompt, output_path):
    response = chat_with_model(gpt_key, images_folder_path, task_description, llm_model, system_prompt)
    save_response_to_file(output_path, response)


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=Path('../.env'))

    gpt_key = os.getenv("GPT_KEY")
    llm_model = os.getenv("LLM_MODEL")
    robot_task = os.getenv("ROBOT_TASK")

    images_folder_path = "../../../images"
    output_path = "../../../output/gpt4-nano/test.json"

    system_prompt = """
    You are a robotics planning assistant. You are given:
    - A task description
    - An image of the environment

    Your goal is to reason about the environment and generate a JSON-LD action graph based on the task.

    The graph must follow this ontology:

    Each node represents an action, which is:
    - An instance of a class (e.g., PickUp, Place, MoveTo)
    - Linked to other actions using temporal relationships

    Each action must include:
    - `@id`: unique identifier (e.g., "action1", "action2")
    - `@type`: the action class name (e.g., "PickUp")
    - `hasObject`: the object being manipulated (if applicable)
    - `hasLocation`: the location associated with the action (if applicable)
    - `follows`: the ID of a previous action (if applicable)
    - `precedes`: the ID of a next action (optional; use either `follows` or `precedes` to indicate order)

    Use this JSON-LD context:

    {
    "@context": {
        "Action": "http://example.org/ontology#Action",
        "precedes": { "@id": "http://example.org/ontology#precedes", "@type": "@id" },
        "follows": { "@id": "http://example.org/ontology#follows", "@type": "@id" },
        "hasObject": { "@id": "http://example.org/ontology#hasObject", "@type": "@id" },
        "hasLocation": { "@id": "http://example.org/ontology#hasLocation", "@type": "@id" },
        "PickUp": "http://example.org/ontology#PickUp",
        "Place": "http://example.org/ontology#Place",
        "MoveTo": "http://example.org/ontology#MoveTo"
    }
    }

    You should return only the JSON-LD structure inside the @graph array. Base your reasoning only on the image and the task description.
    """

    task_description = "The robot needs to reorganise the kitchen"

    main(gpt_key, images_folder_path, task_description, llm_model, system_prompt, output_path)



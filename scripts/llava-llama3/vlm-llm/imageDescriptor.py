import os
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
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(response_json, f, ensure_ascii=False, indent=4)
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
    output_path = "../../../output/llava-llama3/vlm-llm/llama-image-description.json"

    user_query = """
    ## INSTRUCTIONS ##
    You are given a set of images taken from the same environment at different angles.  
    These images together represent the complete layout and state of the environment.  
    Carefully analyse all visual details from the images.  

    Your task is to produce a comprehensive description of the environment as seen in the images.

    ## OUTPUT FORMAT ##
    You must return only text output, with no introductory text or explanations.  
    Format your output into exactly one section, clearly marked by the section title below.

    ## SECTION TITLE AND EXPECTED CONTENT ##
    1. Environment Description: describe the environment as seen from all images combined. Include: all visible objects, their positions relative to each other and to the environment; stacked or nested relationships (e.g., “a bowl inside a plate on top of a placemat”); spatial orientation (e.g., “to the left of the sink”, “at the far end of the table”)
    """

    # 3. Final Environment Description: describe how the environment should look after the task is completed. Include: the new positions of any moved objects

    main(images_folder_path, nebula_key, nebula_url, vlm_model, user_query, output_path)
import os
import json
import base64
import requests

from dotenv import load_dotenv
from pathlib import Path
from groq import Groq


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

def chat_with_model(groq_key, llm_model, user_query, images_folder_path):

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
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(response.to_dict(), f, ensure_ascii=False, indent=4)
    return 

def main(images_folder_path, groq_key, llm_model, user_query, output_path):
    response_json = chat_with_model(groq_key, llm_model, user_query, images_folder_path)
    save_response_to_file(output_path, response_json)


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=Path('../.env'))

    groq_key = os.getenv("GROQ_KEY")
    llm_model = os.getenv("LLM_MODEL")
    robot_task = os.getenv("ROBOT_TASK")

    images_folder_path = "../../../images"
    output_path = "../../../output/llama4/llm-chat/llm_response.json"

    user_query = """
    ## INSTRUCTIONS ##
    You are given a set of images taken from the same environment at different angles. 
    These images together represent the complete layout and state of the environment in which a robot must perform a task.
    Carefully analyse all visual details from the images, and return the ordered robot actions that the robot must perform to complete the task, such that: each step is a single, atomic, clear action; the plan is physically and logically valid; actions reference specific objects and locations based on the environment description.
    
    The robot must perform the following task in the environment: [{robot_task}]

    ## OUTPUT FORMAT ##
    You must return only text output, with no introductory text or explanations.
    """

    # 3. Final Environment Description: describe how the environment should look after the task is completed. Include: the new positions of any moved objects

    main(images_folder_path, groq_key, llm_model, user_query, output_path)
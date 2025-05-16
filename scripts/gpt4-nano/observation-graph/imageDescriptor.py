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

def chat_with_model(gpt_key, images_folder_path, user_query, llm_model):
    client = OpenAI(api_key=gpt_key)
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
    response_json = response.to_dict()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(response_json, f, ensure_ascii=False, indent=4)
    return 

def main(images_folder_path, gpt_key, llm_model, user_query, output_path):
    response = chat_with_model(gpt_key, images_folder_path, user_query, llm_model)
    save_response_to_file(output_path, response)


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=Path('../.env'))

    gpt_key = os.getenv("GPT_KEY")
    llm_model = os.getenv("LLM_MODEL")

    images_folder_path = "../../../images"
    output_path = "../../../output/gpt4-nano/observation-graph/image-description.json"

    user_query = """
    ## INSTRUCTIONS ##
    You are given a set of images taken from the same environment at different angles. 
    These images together represent the complete layout and state of the environment in which a robot must perform a task.
    Your task is to carefully analyse all visual details from the images and provide a description of the environment as seen from all images combines. 
    Specifically, include: all visible objects, their positions relative to each other and to environment; stacked or nested relationships (e.g., “a bowl inside a plate on top of a placemat”); spatial orientation (e.g., “to the left of the sink”, “at the far end of the table”)

    ## OUTPUT FORMAT ##
    You must return only text output, with no introductory text or explanations.
    """

    main(images_folder_path, gpt_key, llm_model, user_query, output_path)



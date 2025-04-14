import os
import base64

from groq import Groq
from dotenv import load_dotenv

def get_client(api_key):
   client = Groq(api_key=api_key)
   return client

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image
  
def get_response(client, user_query, base64_image, vlm_model):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model=vlm_model,
    )
    response = chat_completion.choices[0].message.content
    return response

def save_response_to_file(image_path, output_path, response):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(response)
    return 

def main(api_key, image_path, output_path, user_query, vlm_model):
    client = get_client(api_key)
    base64_image = encode_image(image_path)
    response = get_response(client, user_query, base64_image, vlm_model)
    save_response_to_file(image_path, output_path, response)

if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(current_dir, ".env")
    load_dotenv(dotenv_path)

    vlm_model = os.getenv("VLM_MODEL")
    api_key = os.getenv("API_KEY")
    image_path = os.getenv("IMAGE_PATH")
    output_path = os.getenv("OUTPUT_PATH")

    user_query = "Give a detailed description of the image."

    main(api_key, image_path, output_path, user_query, vlm_model)
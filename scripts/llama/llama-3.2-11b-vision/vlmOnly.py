import os
import json
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

def get_response(client, user_query, base64_image, llm_model):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {   "type": "text", 
                        "text": user_query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            },
        ],
        model=llm_model,
    )
    response = chat_completion.choices[0].message.content
    return response
  
def main(llm_model, api_key, image_path, user_query):
   client = get_client(api_key)
   base64_image = encode_image(image_path)
   response = get_response(client, user_query, base64_image, llm_model)
   print(response)


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(current_dir, ".env")
    load_dotenv(dotenv_path)

    llm_model = os.getenv("MODEL")
    api_key = os.getenv("API_KEY")
    image_path = os.getenv("IMAGE_PATH")

    user_query = "Given a detailed description of the image."

    main(llm_model, api_key, image_path, user_query)
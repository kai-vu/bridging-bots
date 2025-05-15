import os
import sys
import json
import numpy as np
import pandas as pd

from openai import OpenAI
from dotenv import load_dotenv
from IPython.display import Image, display
from sklearn.metrics.pairwise import cosine_similarity


def gpt_client(gpt_key):
    client = OpenAI(api_key=gpt_key)
    return client

def create_assistant(client, llm_model):
    assistant = client.beta.assistants.create(
        name = "PSR-tak-assistant",
        instructions="", #### HERE TO ADD INSTRUCTIONS
        tools=[{"type": "file_search"}],
        model=llm_model
    )
    return assistant

def add_file_to_vector_store(ontology_path, client, vector_store):
    with open(ontology_path, 'rb') as file:
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=[file]
        )
    return vector_store

def create_vector_store(client, ontology_path):
    vector_store = client.beta.vector_stores.create(name="PSR-task-vector-store")
    add_file_to_vector_store(ontology_path, client, vector_store)
    return vector_store

def update_assistant(assistant, client, vector_store):
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    return assistant

def create_thread(client, user_query):
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": user_query,
            }
        ]
    )
    return thread

def get_response(client, assistant, thread):
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant.id
    )
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
    return messages, run

def get_response_value(client, llm_model, ontology_path, user_query):
    assistant = create_assistant(client, llm_model)
    vector_store = create_vector_store(client, ontology_path)
    assistant = update_assistant(assistant, client, vector_store)
    thread = create_thread(client, user_query)
    messages, run = get_response(client, assistant, thread)
    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
    return message_content.value, run

def save_response_to_file(output_path, response):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(response.to_dict(), f, ensure_ascii=False, indent=4)
    return 

def main(gpt_key, images_folder_path, user_query, llm_model, output_path):
    client = gpt_client(gpt_key)
    response_value = get_response_value(client, llm_model, images_folder_path, user_query)
    save_response_to_file(output_path, response_value)


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=Path('../.env'))

    gpt_key = os.getenv("GPT_KEY")
    llm_model = os.getenv("LLM_MODEL")
    robot_task = os.getenv("ROBOT_TASK")

    images_folder_path = "../../../images"
    output_path = "../../../output/gpt4-nano/llm-rag/llama-image-description.json"

    user_query = """
    ## INSTRUCTIONS ##
    You are given a set of images taken from the same environment at different angles. 
    These images together represent the complete layout and state of the environment in which a robot must perform a task.
    Carefully analyse all visual details from the images. 
    
    The robot must perform the following task in the environment: [{robot_task}]

    ## OUTPUT FORMAT ##
    You must return only text output, with no introductory text or explanations.
    Structure your output as a **knowledge graph** using RDF triples in the format: (subject, predicate, object)

    Use URIs or local names that are consistent with the ontology definitions.

    ## SECTION TITLES AND EXPECTED CONTENT ## 
    1. Environment Description: describe the environment as seen from all images combined. Include: all visible objects, their positions relative to each other and to environment; stacked or nested relationships (e.g., “a bowl inside a plate on top of a placemat”); spatial orientation (e.g., “to the left of the sink”, “at the far end of the table”)

    2. Ordered Robot Actions:list the robot's actions in order to complete the task, such that: each step is a single, atomic, clear action; the plan is physically and logically valid; actions reference specific objects and locations based on the environment description
    
    ## ONTOLOGY ##
    The following ontology defines the classes and properties to use in the knowledge graph.
    It is written in Turtle (TTL) syntax:

    ```ttl
    {ontology_ttl}
    """

    main(gpt_key, images_folder_path, user_query, llm_model)
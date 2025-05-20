import os
import io
import re
import json

from pathlib import Path
from dotenv import load_dotenv
from rdflib import Graph
from openai import OpenAI

def create_assistant(client, llm_model):
    assistant = client.beta.assistants.create(
        name="euRobin assistant",
        instructions="""
        You are an intelligent assistant that generates Knowledge Graphs from text, following an ontology.

        You are provided with:
        - A text description of a physical environment, given to you by the user.
        - An ontology (retrieved from context) that defines the allowed vocabulary: classes, properties, and relations.

        ## TASK ##
        You must use the ontology **as a strict schema** to construct a Knowledge Graph.
        This means:
        - Use **only** the classes and properties defined in the ontology.
        - Do **not invent or infer** terms not explicitly defined in the ontology.
        - All entities and relations must conform to the structure and semantics of the ontology.

        ## OUTPUT FORMAT ##
        - Output only text, with no extra explanations.
        - Output must consist of triples in turtle format.
        - Output must contain the prefixes and namespaces. 
        """,
        model=llm_model,
        tools=[{"type": "file_search"}],
    )
    assistant_id = assistant.id
    return assistant, assistant_id

def create_vector_store(client, ttl_path):
    vector_store = client.vector_stores.create(name="euRobin vector store")
    with open(ttl_path, "r", encoding="utf-8") as f:
        ttl_text = f.read()
    ttl_stream = io.BytesIO(ttl_text.encode("utf-8"))
    ttl_stream.name = "ontology.txt"
    file_batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=[ttl_stream]
    )
    return vector_store

def update_assistant(client, assistant_id, vector_store):
    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
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

def get_response(client, llm_model, ontology_path, user_query):
    assistant, assistant_id = create_assistant(client, llm_model)
    vector_store = create_vector_store(client, ontology_path)
    assistant = update_assistant(client, assistant_id, vector_store)
    thread = create_thread(client, user_query)
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant.id
    )
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")
    #print(message_content.value)
    response = message_content
    return response, run

def save_response(response, run, output_path):
    response_usage_path = os.path.join(output_path, "response_usage.json")
    usage_data = {
        "prompt_tokens": run.usage.prompt_tokens,
        "completion_tokens": run.usage.completion_tokens,
        "total_tokens": run.usage.total_tokens
    }
    with open(response_usage_path, "w") as f:
        json.dump(usage_data, f, indent=4)
    ttl_response_path = os.path.join(output_path, "kg.ttl")
    ttl_response = response.value
    ttl_response = str(ttl_response)
    ttl_response = re.sub(r"^```[^\n]*\n", "", ttl_response.strip())
    ttl_response = re.sub(r"```$", "", ttl_response.strip())
    ttl_response = ttl_response.strip()
    with open(ttl_response_path, 'w') as f:
        f.write(ttl_response)
    return 

def main(gpt_key, llm_model, ontology_path, user_query, output_path):
    client = OpenAI(api_key=gpt_key)
    response, run = get_response(client, llm_model, ontology_path, user_query)
    save_response(response, run, output_path)

if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    gpt_key = os.getenv("GPT_KEY")
    embedding_model = "BAAI/bge-small-en"

    description_path = "../../../output/gpt-o1/observation-graph/image-description.json"
    output_path = "../../../output/gpt-o1/observation-graph/d2kg"
    ontology_path = "../../../ontology/ontoObservationGraph.ttl"

    with open(description_path, 'r', encoding='utf-8') as file:
        description_txt = file.read()

    user_query = f"""
    Create a Knowledge Graph following the ontology, for the environment description below. 

    ## OUTPUT FORMAT ##
    - Output only text, with no extra explanations.
    - Output must consist of triples in turtle format.
    - Output must contain the prefixes and namespaces.

    ### Environment Description ###
    {description_txt}
    """

    main(gpt_key, llm_model, ontology_path, user_query, output_path)

    
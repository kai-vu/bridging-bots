import os
import io

from pathlib import Path
from dotenv import load_dotenv
from rdflib import Graph
from openai import OpenAI

def create_assistant(client, llm_model):
    assistant = client.beta.assistants.create(
        name="assistant",
        instructions="You are an expert in knowledge graph creation",
        model=llm_model,
        tools=[{"type": "file_search"}],
    )
    assistant_id = assistant.id
    return assistant, assistant_id

def create_vector_store(client, ttl_path):
    vector_store = client.vector_stores.create(name="vector store")

    with open(ttl_path, "r", encoding="utf-8") as f:
        ttl_text = f.read()

    ttl_stream = io.BytesIO(ttl_text.encode("utf-8"))
    ttl_stream.name = "ontology.txt"  # pretend the file is text

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
    print(message_content.value)

def main(gpt_key, llm_model, ontology_path, user_query, output_path):
    client = OpenAI(api_key=gpt_key)
    get_response(client, llm_model, ontology_path, user_query)

if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    gpt_key = os.getenv("GPT_KEY")
    embedding_model = "BAAI/bge-small-en"

    description_path = "../../../output/gpt4-nano/observation-graph/image-description.json"
    output_path = "../../../output/gpt4-nano/observation-graph/ragKG"
    ontology_path = "../../../ontology/ontoObservationGraph.ttl"

    with open(description_path, 'r', encoding='utf-8') as file:
        description_txt = file.read()

    user_query = f"""
    ## INSTRUCTIONS ##
    You are an intelligent assistant that generates Knowledge Graphs from text.

    You are provided with:
    - A text description of a physical environment.
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

    ## INPUT ##
    ### Environment Description ###
    {description_txt}
    """

    main(gpt_key, llm_model, ontology_path, user_query, output_path)

    
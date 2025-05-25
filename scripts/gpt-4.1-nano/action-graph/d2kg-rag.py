import os
import io
import re
import json

from pathlib import Path
from dotenv import load_dotenv
from rdflib import Graph
from openai import OpenAI

def get_response(client, llm_model, ontology_path, user_query, assistant_instructions):
    assistant = client.beta.assistants.create(
        name="euRobin assistant",
        instructions=assistant_instructions,
        model=llm_model,
        tools=[{"type": "file_search"}],
    )
    vector_store = client.vector_stores.create(name="euRobin vector store")
    with open(ontology_path, "r", encoding="utf-8") as original_file:
        content = original_file.read()
        fake_file = io.BytesIO(content.encode("utf-8"))
        fake_file.name = "ontology.txt"

        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=[fake_file])
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    assistant = client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    thread = client.beta.threads.create(
        messages=[
            {
            "role": "user",
            "content": user_query,
            }
        ]
    )
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

def main(gpt_key, llm_model, ontology_path, user_query, output_path, assistant_instructions):
    client = OpenAI(api_key=gpt_key)
    response, run = get_response(client, llm_model, ontology_path, user_query, assistant_instructions)
    save_response(response, run, output_path)

if __name__ == "__main__":

    load_dotenv(dotenv_path=Path('../.env'))

    llm_model = os.getenv("LLM_MODEL")
    gpt_key = os.getenv("GPT_KEY")
    robot_task = os.getenv("ROBOT_TASK")

    description_path = "../../../output/gpt-4.1-nano/observation-graph/image-description.json"
    output_path = "../../../output/gpt-4.1-nano/action-graph/d2kg-rag"
    ontology_path = "../../../ontology/ontoActionGraph.ttl"

    with open(description_path, 'r', encoding='utf-8') as file:
        description_txt = file.read()

    user_query = f"""
You are an intelligent assistant tasked to generate a **Knowledge Graph of the sequence of actions a robot must perform to accomplish the following task**:
---------------------
ROBOT TASK: {robot_task}
---------------------

You are provided with text description of an environment, below:
---------------------
ENVIRONMENT DESCRIPTION: {description_txt}
---------------------

Instructions:
- Analyze the description carefully to understand the complete layout of the environment.
- Based on the ontology stored in the vector, **generate the sequence of actions required for the robot to complete the task**.
- Each action is a **single, atomic, clear action**.
- **All actions, entities, and relationships must strictly follow the provided ontology.**
- **Use only classes and properties from the ontology.**
- Do **NOT invent or infer any terms or actions outside of the ontology schema.**
- The graph should represent actions, objects involved, and their relations according to the ontology's structure and semantics.

Output format:
- Return only the generated Knowledge Graph of actions.
- Output only text, no extra explanations.
- Use Turtle format for the output, such as <subject> <predicate> <object> .
- Include all prefixes and namespaces at the beginning. 
- Use the ex: prefix with namespace <http://example.org/data/> only for newly instantiated entities instantiated, such as specific actions, objects, or locations.
- Do not use the ex: prefix for ontology classes, properties, or schema definitions, those must strictly come from the provided ontology with their original prefixes and namespaces.
    """

    assistant_instructions = """
You are an intelligent assistant tasked to generate a **Knowledge Graph of the sequence of actions a robot must perform to accomplish a task**.

The user will provide the description of the environment and the task the robot must performand. The ontology is stored in the vector. 

Instructions:
- Analyze the description carefully to understand the complete layout of the environment.
- Based on the ontology stored in the vector, **generate the sequence of actions required for the robot to complete the task**.
- Each action is a **single, atomic, clear action**.
- **All actions, entities, and relationships must strictly follow the provided ontology.**
- **Use only classes and properties from the ontology.**
- Do **NOT invent or infer any terms or actions outside of the ontology schema.**
- The graph should represent actions, objects involved, and their relations according to the ontology's structure and semantics.

Output format:
- Return only the generated Knowledge Graph of actions.
- Output only text, no extra explanations.
- Use Turtle format for the output, such as <subject> <predicate> <object> .
- Include all prefixes and namespaces at the beginning. 
- Use the ex: prefix with namespace <http://example.org/data/> only for newly instantiated entities instantiated, such as specific actions, objects, or locations.
- Do not use the ex: prefix for ontology classes, properties, or schema definitions, those must strictly come from the provided ontology with their original prefixes and namespaces.
    """

    main(gpt_key, llm_model, ontology_path, user_query, output_path, assistant_instructions)

    
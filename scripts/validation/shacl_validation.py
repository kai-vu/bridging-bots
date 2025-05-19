import subprocess
import json
import csv

output_file = "../../output/validation/shacl_validation_report.txt"
output_csv = "../../output/validation/shacl_validation_report.csv"

files_observation_graph = [
    # llava-llama3
    "../../output/llava-llama3/observation-graph/dynamicKG/kg.ttl",
    "../../output/llava-llama3/observation-graph/imageToKG/kg.ttl",
    "../../output/llava-llama3/observation-graph/promptKG/kg.ttl",
    "../../output/llava-llama3/observation-graph/ragKG/kg.ttl",
    # llama4-scout
    "../../output/llama4-scout/observation-graph/dynamicKG/kg.ttl",
    "../../output/llama4-scout/observation-graph/imageToKG/kg.ttl",
    "../../output/llama4-scout/observation-graph/promptKG/kg.ttl",
    "../../output/llama4-scout/observation-graph/ragKG/kg.ttl",
    # llama4-maverick
    "../../output/llama4-maverick/observation-graph/dynamicKG/kg.ttl",
    "../../output/llama4-maverick/observation-graph/imageToKG/kg.ttl",
    "../../output/llama4-maverick/observation-graph/promptKG/kg.ttl",
    "../../output/llama4-maverick/observation-graph/ragKG/kg.ttl",
    # gpt-o1
    "../../output/gpt-o1/observation-graph/imageToKG/kg.ttl",
    "../../output/gpt-o1/observation-graph/promptKG/kg.ttl",
    "../../output/gpt-o1/observation-graph/ragKG/kg.ttl",
    # gpt-4.1-nano
    "../../output/gpt-4.1-nano/observation-graph/imageToKG/kg.ttl",
    "../../output/gpt-4.1-nano/observation-graph/promptKG/kg.ttl",
    "../../output/gpt-4.1-nano/observation-graph/ragKG/kg.ttl",
]
shapes_observation_graph = "../../ontology/shacl-shapes/shapesObservationGraph.ttl"
ontology_observation_graph = "../../ontology/ontoObservationGraph.ttl"


files_action_graph = [
    # llava-llama3
    "../../output/llava-llama3/action-graph/dynamicKG/kg.ttl",
    "../../output/llava-llama3/action-graph/imageToKG/kg.ttl",
    "../../output/llava-llama3/action-graph/promptKG/kg.ttl",
    "../../output/llava-llama3/action-graph/ragKG/kg.ttl",
    # llama4-scout
    "../../output/llama4-scout/action-graph/dynamicKG/kg.ttl",
    "../../output/llama4-scout/action-graph/imageToKG/kg.ttl",
    "../../output/llama4-scout/action-graph/promptKG/kg.ttl",
    "../../output/llama4-scout/action-graph/ragKG/kg.ttl",
    # llama4-maverick
    "../../output/llama4-maverick/action-graph/dynamicKG/kg.ttl",
    "../../output/llama4-maverick/action-graph/imageToKG/kg.ttl",
    "../../output/llama4-maverick/action-graph/promptKG/kg.ttl",
    "../../output/llama4-maverick/action-graph/ragKG/kg.ttl",
    # gpt-o1
    "../../output/gpt-o1/action-graph/imageToKG/kg.ttl",
    "../../output/gpt-o1/action-graph/promptKG/kg.ttl",
    "../../output/gpt-o1/action-graph/ragKG/kg.ttl",
    # gpt-4.1-nano
    "../../output/gpt-4.1-nano/action-graph/imageToKG/kg.ttl",
    "../../output/gpt-4.1-nano/action-graph/promptKG/kg.ttl",
    "../../output/gpt-4.1-nano/action-graph/ragKG/kg.ttl",
]
shapes_action_graph = "../../ontology/shacl-shapes/shapesActionGraph.ttl"
ontology_action_graph = "../../ontology/ontoActionGraph.ttl"

def run_pyshacl(data_file, shapes_file, ontology_file):
    result = subprocess.run([
        "pyshacl",
        "-d", data_file,
        "-s", shapes_file,
        "-e", ontology_file,
        "-i", "rdfs",
        "-f", "json-ld"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Validation failed for {data_file}")
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON for {data_file}")
        return None

def extract_summary(jsonld_data):
    # jsonld_data is a list of dicts
    for node in jsonld_data:
        if "@type" in node and "http://www.w3.org/ns/shacl#ValidationReport" in node["@type"]:
            conforms = node.get("http://www.w3.org/ns/shacl#conforms", [{"@value": False}])[0]["@value"]
            results = node.get("http://www.w3.org/ns/shacl#result", [])
            if not isinstance(results, list):
                results = [results]
            return conforms, len(results)
    # If no report found
    return "NoReportFound", 0

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Data File", "Conforms", "Number of Validation Results"])

    for data_file in files_observation_graph:
        print(f"Validating {data_file} ...")
        report = run_pyshacl(data_file, shapes_observation_graph, ontology_observation_graph)
        if report:
            conforms, num_results = extract_summary(report)
            writer.writerow([data_file, conforms, num_results])
        else:
            writer.writerow([data_file, "ERROR", "ERROR"])
    for data_file in files_action_graph:
        print(f"Validating {data_file} ...")
        report = run_pyshacl(data_file, shapes_action_graph, ontology_action_graph)
        if report:
            conforms, num_results = extract_summary(report)
            writer.writerow([data_file, conforms, num_results])
        else:
            writer.writerow([data_file, "ERROR", "ERROR"])


import subprocess
import json
import csv
import os

from pyshacl import validate
from rdflib import Graph


output_txt = "../../output/validation/shacl_validation_report.txt"
output_csv = "../../output/validation/shacl_validation_report.csv"

files_observation_graph = [
    # llava-llama3
    "../../output/llava-llama3/observation-graph/dpe/kg.ttl",
    "../../output/llava-llama3/observation-graph/i2kg/kg.ttl",
    "../../output/llava-llama3/observation-graph/d2kg/kg.ttl",
    "../../output/llava-llama3/observation-graph/d2kg-rag/kg.ttl",
    # llama4-scout
    "../../output/llama4-scout/observation-graph/dpe/kg.ttl",
    "../../output/llama4-scout/observation-graph/i2kg/kg.ttl",
    "../../output/llama4-scout/observation-graph/d2kg/kg.ttl",
    "../../output/llama4-scout/observation-graph/d2kg-rag/kg.ttl",
    # llama4-maverick
    "../../output/llama4-maverick/observation-graph/dpe/kg.ttl",
    "../../output/llama4-maverick/observation-graph/i2kg/kg.ttl",
    "../../output/llama4-maverick/observation-graph/d2kg/kg.ttl",
    "../../output/llama4-maverick/observation-graph/d2kg-rag/kg.ttl",
    # gpt-o1
    "../../output/gpt-o1/observation-graph/i2kg/kg.ttl",
    "../../output/gpt-o1/observation-graph/d2kg/kg.ttl",
    "../../output/gpt-o1/observation-graph/d2kg-rag/kg.ttl",
    # gpt-4.1-nano
    "../../output/gpt-4.1-nano/observation-graph/i2kg/kg.ttl",
    "../../output/gpt-4.1-nano/observation-graph/d2kg/kg.ttl",
    "../../output/gpt-4.1-nano/observation-graph/d2kg-rag/kg.ttl",
]
shapes_observation_graph = "../../ontology/shacl-shapes/shapesObservationGraph.ttl"
ontology_observation_graph = "../../ontology/ontoObservationGraph.ttl"

files_action_graph = [
    # llava-llama3
    "../../output/llava-llama3/action-graph/dpe/kg.ttl",
    "../../output/llava-llama3/action-graph/i2kg/kg.ttl",
    "../../output/llava-llama3/action-graph/d2kg/kg.ttl",
    "../../output/llava-llama3/action-graph/d2kg-rag/kg.ttl",
    # llama4-scout
    "../../output/llama4-scout/action-graph/dpe/kg.ttl",
    "../../output/llama4-scout/action-graph/i2kg/kg.ttl",
    "../../output/llama4-scout/action-graph/d2kg/kg.ttl",
    "../../output/llama4-scout/action-graph/d2kg-rag/kg.ttl",
    # llama4-maverick
    "../../output/llama4-maverick/action-graph/dpe/kg.ttl",
    "../../output/llama4-maverick/action-graph/i2kg/kg.ttl",
    "../../output/llama4-maverick/action-graph/d2kg/kg.ttl",
    "../../output/llama4-maverick/action-graph/d2kg-rag/kg.ttl",
    # gpt-o1
    "../../output/gpt-o1/action-graph/i2kg/kg.ttl",
    "../../output/gpt-o1/action-graph/d2kg/kg.ttl",
    "../../output/gpt-o1/action-graph/d2kg-rag/kg.ttl",
    # gpt-4.1-nano
    "../../output/gpt-4.1-nano/action-graph/i2kg/kg.ttl",
    "../../output/gpt-4.1-nano/action-graph/d2kg/kg.ttl",
    "../../output/gpt-4.1-nano/action-graph/d2kg-rag/kg.ttl",
]
shapes_action_graph = "../../ontology/shacl-shapes/shapesActionGraph.ttl"
ontology_action_graph = "../../ontology/ontoActionGraph.ttl"

def validate_graph(g_file, sg_file, og_file, log_file_path):
    try:
        g = Graph()
        g.parse(g_file, format="turtle")
        sg = Graph()
        sg.parse(sg_file, format="turtle")
        og = Graph()
        og.parse(og_file, format="turtle")

        result = validate(
            g,
            shacl_graph=sg,
            ont_graph=og,
            inference='none',
            abort_on_first=False,
            allow_infos=False,
            allow_warnings=False,
            meta_shacl=False,
            advanced=False,
            js=False,
            debug=False
        )
        return result 

    except Exception as e:
        with open(log_file_path, "a") as f:
            f.write(f"=== ERROR validating: {g_file} ===\n")
            f.write(f"{type(e).__name__}: {e}\n\n")
        print(f"[ERROR] Could not validate {g_file}: {e}")
        return None

def run_all_validations(file_list, shapes_file, ontology_file, result_txt_path, csv_rows):
    for file in file_list:
        print(f"Validating: {file}")
        result = validate_graph(file, shapes_file, ontology_file, result_txt_path)
        
        if result is None:
            csv_rows.append({
                "file": file,
                "conforms": "ERROR"
            })
            continue

        conforms, results_graph, results_text = result

        with open(result_txt_path, "a") as out_txt:
            out_txt.write(f"=== Validation Report for: {file} ===\n")
            out_txt.write(results_text)
            out_txt.write("\n\n")

        csv_rows.append({
            "file": file,
            "conforms": conforms
        })


os.makedirs(os.path.dirname(output_txt), exist_ok=True)
with open(output_txt, "w") as f:
    f.write("SHACL Validation Report\n\n")

csv_data = []
run_all_validations(files_observation_graph, shapes_observation_graph, ontology_observation_graph, output_txt, csv_data)
run_all_validations(files_action_graph, shapes_action_graph, ontology_action_graph, output_txt, csv_data)

with open(output_csv, "w", newline="") as csvfile:
    fieldnames = ["file", "conforms"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_data)

print("Validation complete. Results saved to:")
print(f" - Text: {output_txt}")
print(f" - CSV:  {output_csv}")
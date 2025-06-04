import subprocess
import json
import csv
import os

from pyshacl import validate
from rdflib import Graph
from rdflib import Namespace, Literal, URIRef


runs = ["run1", "run2", "run3", "run4", "run5", "run6", "run7", "run8", "run9", "run10"]

models = ["llava-llama3", "llama4-scout", "llama4-maverick", "gpt-o1", "gpt-4.1-nano"]
graph_types = ["observation-graph", "action-graph"]
methods = ["dpe", "i2kg", "d2kg", "d2kg-rag"]

shapes_files = {
    "observation-graph": "../../ontology/shacl-shapes/shapesObservationGraph.ttl",
    "action-graph": "../../ontology/shacl-shapes/shapesActionGraph.ttl"
}
ontology_files = {
    "observation-graph": "../../ontology/ontoObservationGraph.ttl",
    "action-graph": "../../ontology/ontoActionGraph.ttl"
}


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

def run_all_validations(file_list, shapes_file, ontology_file, result_txt_path):
    result_rows = []
    results_graph = Graph()
    EX = Namespace("http://example.org/validation/")

    for file in file_list:
        print(f"Validating: {file}")
        result = validate_graph(file, shapes_file, ontology_file, result_txt_path)

        if result is None:
            result_rows.append({"file": file, "conforms": "ERROR"})
            continue

        conforms, r_graph, results_text = result

        for s in r_graph.subjects(predicate=URIRef("http://www.w3.org/ns/shacl#conforms")):
            r_graph.add((s, EX.sourceFile, Literal(file)))
            break

        results_graph += r_graph

        with open(result_txt_path, "a") as out_txt:
            out_txt.write(f"=== Validation Report for: {file} ===\n")
            out_txt.write(results_text)
            out_txt.write("\n\n")

        result_rows.append({"file": file, "conforms": conforms})

    return results_graph, result_rows


for run in runs:
    print(f"\n=== Running validation for {run} ===")

    output_txt = f"../../output/{run}/validation/shacl/shacl_validation_report.txt"
    output_csv = f"../../output/{run}/validation/shacl/shacl_validation_report.csv"
    output_ttl = f"../../output/{run}/validation/shacl/shacl_validation_report.ttl"

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    with open(output_txt, "w") as f:
        f.write(f"SHACL Validation Report for {run}\n\n")

    final_results_graph = Graph()
    csv_data = []

    for graph_type in graph_types:
        files_to_validate = []

        for model in models:
            for method in methods:
                file_path = f"../../output/{run}/{model}/{graph_type}/{method}/kg.ttl"
                if os.path.exists(file_path):
                    files_to_validate.append(file_path)
                else:
                    print(f"[WARNING] File not found: {file_path}")

        results_graph, result_rows = run_all_validations(
            files_to_validate,
            shapes_files[graph_type],
            ontology_files[graph_type],
            output_txt
        )

        final_results_graph += results_graph
        csv_data.extend(result_rows)

    final_results_graph.serialize(destination=output_ttl, format="turtle")

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["file", "conforms"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"Validation complete for {run}. Results saved to:")
    print(f" - Text: {output_txt}")
    print(f" - CSV:  {output_csv}")
    print(f" - TTL:  {output_ttl}")
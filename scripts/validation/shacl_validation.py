import subprocess
import json
import csv

from pyshacl import validate
from rdflib import Graph


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

def validate(g_file, sg_file, og_file):
    g = Graph()
    g.parse(g_file, format="turtle")
    sg = Graph()
    sg.parse(sg_file, format="turtle")
    og = Graph()
    og.parse(og_file, format="turtle")

    r = validate(g,
        shacl_graph=sg,
        ont_graph=og,
        inference='both',
        abort_on_first=False,
        allow_infos=False,
        allow_warnings=False,
        meta_shacl=False,
        advanced=False,
        js=False,
        debug=False)
    conforms, results_graph, results_text = r
    results_graph.serialize(destination="output_file.ttl", format="turtle")
    print(conforms)

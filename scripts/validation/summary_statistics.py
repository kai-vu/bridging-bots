import rdflib
from rdflib import Graph, RDF, OWL
from rdflib.plugins.parsers.notation3 import BadSyntax
import pandas as pd
import os

def try_parse_graph(file_path):
    g = Graph()
    try:
        g.parse(file_path, format='turtle')
        return g, True, 0
    except Exception as e:
        print(f"[WARNING] Full parse failed for {file_path}: {e}")
        # Try line-by-line parsing fallback
        g_partial = Graph()
        valid_triples = 0
        invalid_triples = 0
        with open(file_path, "r") as f:
            lines = f.readlines()

        buffer = ""
        for line in lines:
            buffer += line.strip() + " "
            if line.strip().endswith("."):
                try:
                    g_partial.parse(data=buffer, format="turtle")
                    valid_triples += 1
                except Exception:
                    invalid_triples += 1
                buffer = ""

        return g_partial, False, invalid_triples

def get_ontology_terms(ont_graph):
    classes = set(ont_graph.subjects(RDF.type, OWL.Class))
    properties = set(ont_graph.subjects(RDF.type, rdflib.RDF.Property))
    properties |= set(ont_graph.subjects(RDF.type, OWL.ObjectProperty))
    properties |= set(ont_graph.subjects(RDF.type, OWL.DatatypeProperty))
    return classes, properties

def get_used_terms(kg_graph):
    used_classes = set(kg_graph.objects(None, RDF.type))
    used_properties = set(kg_graph.predicates(None, None))
    return used_classes, used_properties

def summarize_kg(kg_graph, ont_classes, ont_properties, full_parse, invalid_triples):
    used_classes, used_properties = get_used_terms(kg_graph)
    total_triples = len(kg_graph)

    unexpected_classes = used_classes - ont_classes
    unexpected_properties = used_properties - ont_properties

    typed_instances_known = sum(1 for s in set(kg_graph.subjects(RDF.type, None))
                                if any((s, RDF.type, c) in kg_graph for c in ont_classes))
    
    classes_coverage = (len(used_classes & ont_classes) / len(used_classes) * 100
                        if used_classes else 0)
    ontology_class_utilization = (len(used_classes & ont_classes) / len(ont_classes) * 100
                                  if ont_classes else 0)

    properties_coverage = (len(used_properties & ont_properties) / len(used_properties) * 100
                           if used_properties else 0)
    ontology_property_utilization = (len(used_properties & ont_properties) / len(ont_properties) * 100
                                     if ont_properties else 0)

    return {
        'Full Parse OK': full_parse,
        'Total triples in KG': total_triples + invalid_triples,
        'Valid triples': total_triples,
        'Invalid triples': invalid_triples,
        'Distinct classes used': len(used_classes),
        'Unexpected classes count': len(unexpected_classes),
        'Unexpected classes (examples)': ', '.join([str(c) for c in list(unexpected_classes)[:5]]) or 'None',
        'Distinct properties used': len(used_properties),
        'Unexpected properties count': len(unexpected_properties),
        'Unexpected properties (examples)': ', '.join([str(p) for p in list(unexpected_properties)[:5]]) or 'None',
        'Typed instances with known classes': typed_instances_known,
        'Classes coverage (KG-to-Ontology)': f"{classes_coverage:.1f}%",
        'Ontology-to-KG Classes Coverage': f"{ontology_class_utilization:.1f}%",
        'Properties coverage (KG-to-Ontology)': f"{properties_coverage:.1f}%",
        'Ontology-to-KG Properties Coverage': f"{ontology_property_utilization:.1f}%",
    }

def main(ontology_path, kg_paths, output_file):
    ont_graph = Graph()
    ont_graph.parse(ontology_path, format='turtle')
    ont_classes, ont_properties = get_ontology_terms(ont_graph)

    all_data = []

    for kg_path in kg_paths:
        print(f"Validating {kg_path}")

        kg_graph, full_parse_ok, invalid_triples = try_parse_graph(kg_path)
        summary = summarize_kg(kg_graph, ont_classes, ont_properties, full_parse_ok, invalid_triples)

        all_data.append(summary)

    df = pd.DataFrame(all_data, index=kg_paths)
    print(df)
    df.to_csv(output_file)

if __name__ == '__main__':

    output_observation = "../../output/validation/summary-statistics/summary_observation.csv"
    output_action = "../../output/validation/summary-statistics/summary_action.csv"

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
    ontology_action_graph = "../../ontology/ontoActionGraph.ttl"

    main(ontology_observation_graph, files_observation_graph, output_observation)
    main(ontology_action_graph, files_action_graph, output_action)
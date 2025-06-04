import rdflib
from rdflib import Graph, RDF, URIRef
import pandas as pd
import os


def try_parse_graph(file_path):
    g = Graph()
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        if "@prefix :" not in content and "@base" not in content:
            default_base = "@prefix : <http://example.org/data/> .\n"
            content = default_base + content

        g.parse(data=content, format='turtle')
        return g, True, 0
    except Exception as e:
        print(f"[WARNING] Full parse failed for {file_path}: {e}")
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
                    if "@prefix :" not in buffer and "@base" not in buffer:
                        buffer = "@prefix : <http://example.org/data/> .\n" + buffer
                    g_partial.parse(data=buffer, format="turtle")
                    valid_triples += 1
                except Exception:
                    invalid_triples += 1
                buffer = ""
        return g_partial, False, invalid_triples

def load_ontology_graph(ontology_path):
    g = Graph()
    g.parse(ontology_path, format='turtle')
    return g

def clean_kg_graph(kg_graph, ontology_graph):
    kg_graph_cleaned = Graph()
    for triple in kg_graph:
        if triple not in ontology_graph:
            kg_graph_cleaned.add(triple)
    return kg_graph_cleaned

def get_used_terms(kg_graph):
    return set(kg_graph.objects(None, RDF.type)), set(kg_graph.predicates(None, None))

def compute_compliance(required_groups, used_terms):
    allowed_terms = set().union(*required_groups)
    valid_terms = used_terms & allowed_terms
    total_used = len(used_terms)
    compliance_pct = (len(valid_terms) / total_used * 100) if total_used else 0
    return compliance_pct, len(valid_terms), total_used

def compute_grouped_coverage(required_groups, used_terms):
    matched_groups = sum(1 for group in required_groups if group & used_terms)
    return (matched_groups / len(required_groups) * 100 if required_groups else 0), matched_groups, len(required_groups)

def summarize_kg(kg_graph, required_class_groups, required_property_groups, full_parse, invalid_triples):
    used_classes, used_properties = get_used_terms(kg_graph)
    total_triples = len(kg_graph)

    class_coverage_pct, matched_class_groups, total_class_groups = compute_grouped_coverage(required_class_groups, used_classes)
    prop_coverage_pct, matched_prop_groups, total_prop_groups = compute_grouped_coverage(required_property_groups, used_properties)

    class_compliance_pct, valid_classes, total_used_classes = compute_compliance(required_class_groups, used_classes)
    prop_compliance_pct, valid_properties, total_used_properties = compute_compliance(required_property_groups, used_properties)

    return {
        'Full_Parse_OK': full_parse,
        'Total_triples_in_KG': total_triples + invalid_triples,
        'Valid_triples': total_triples,
        'Invalid_triples': invalid_triples,
        'Distinct_classes_used': total_used_classes,
        'Class_Compliance': f"{valid_classes}/{total_used_classes}",
        'Class_Coverage': f"{matched_class_groups}/{total_class_groups}",
        'Distinct_properties_used': total_used_properties,
        'Property_Compliance': f"{valid_properties}/{total_used_properties}",
        'Property_Coverage': f"{matched_prop_groups}/{total_prop_groups}",
    }

def main(runs, models, graph_types, methods, output_root, ontology_files, class_requirements, property_requirements):
    for run in runs:
        for graph_type in graph_types:
            file_paths = []
            for model in models:
                for method in methods:
                    path = f"../../output/{run}/{model}/{graph_type}/{method}/kg.ttl"
                    if os.path.exists(path):
                        file_paths.append(path)
                    else:
                        print(f"[WARNING] Missing: {path}")

            if not file_paths:
                print(f"[INFO] No files found for {run}/{graph_type}, skipping.")
                continue

            ontology_path = ontology_files[graph_type]
            required_classes = class_requirements[graph_type]
            required_properties = property_requirements[graph_type]
            ontology_graph = load_ontology_graph(ontology_path)

            all_data = []
            for kg_path in file_paths:
                kg_graph, full_parse_ok, invalid_triples = try_parse_graph(kg_path)
                kg_graph_cleaned = clean_kg_graph(kg_graph, ontology_graph)
                summary = summarize_kg(kg_graph_cleaned, required_classes, required_properties, full_parse_ok, invalid_triples)
                summary['file'] = kg_path
                all_data.append(summary)

            output_csv = os.path.join(output_root, run, "validation", "summary-statistics", f"summary_{graph_type.replace('-', '_')}.csv")
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df = pd.DataFrame(all_data)
            df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    runs = ["run1", "run2", "run3", "run4", "run5", "run6", "run7", "run8", "run9", "run10"]
    models = ["llava-llama3", "llama4-scout", "llama4-maverick", "gpt-o1", "gpt-4.1-nano"]
    graph_types = ["observation-graph", "action-graph"]
    methods = ["dpe", "i2kg", "d2kg", "d2kg-rag"]

    output_root = "../../output"

    ontology_files = {
        "observation-graph": "../../ontology/ontoObservationGraph.ttl",
        "action-graph": "../../ontology/ontoActionGraph.ttl"
    }

    required_class_observation = [
        {URIRef("https://w3id.org/onto-bot#Environment")},
        {URIRef("https://w3id.org/onto-bot#Component"), URIRef("https://w3id.org/onto-bot#Appliance"),
         URIRef("https://w3id.org/onto-bot#Furniture"), URIRef("https://w3id.org/onto-bot#Object")},
        {URIRef("http://www.ease-crc.org/ont/SOMA.owl#Location"),
         URIRef("https://w3id.org/onto-bot#CurrentLocation"),
         URIRef("https://w3id.org/onto-bot#StandardLocation")},
        {URIRef("http://www.ease-crc.org/ont/SOMA.owl#Affordance"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Closing"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Opening"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Delivering"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Holding"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#PickingUp"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#PuttingDown"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Pulling"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Pushing"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Grasping")}
    ]
    required_property_observation = [
        {URIRef("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasComponent")},
        {URIRef("https://w3id.org/onto-bot#hasAffordance")},
        {URIRef("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasLocation"),
         URIRef("https://w3id.org/onto-bot#onTopOf"),
         URIRef("http://www.opengis.net/ont/geosparql#sfContains"),
         URIRef("http://www.opengis.net/ont/geosparql#sfWithin"),
         URIRef("http://www.opengis.net/ont/geosparql#sfOverlaps")}
    ]

    required_class_action = [
        {URIRef("https://w3id.org/onto-bot#Instruction")},
        {URIRef("https://w3id.org/onto-bot#Workflow")},
        {URIRef("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Action")},
        {URIRef("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Agent")},
        {URIRef("https://w3id.org/onto-bot#Component"), URIRef("https://w3id.org/onto-bot#Appliance"),
         URIRef("https://w3id.org/onto-bot#Furniture"), URIRef("https://w3id.org/onto-bot#Object")},
        {URIRef("http://www.ease-crc.org/ont/SOMA.owl#Affordance"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Closing"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Opening"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Delivering"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Holding"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#PickingUp"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#PuttingDown"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Pulling"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Pushing"),
         URIRef("http://www.ease-crc.org/ont/SOMA.owl#Grasping")}
    ]
    required_property_action = [
        {URIRef("https://w3id.org/onto-bot#hasWorkflow")},
        {URIRef("https://w3id.org/onto-bot#hasAction")},
        {URIRef("https://w3id.org/onto-bot#precedes")},
        {URIRef("https://w3id.org/onto-bot#follows")},
        {URIRef("http://www.ease-crc.org/ont/SOMA.owl#isPerformedBy")},
        {URIRef("https://w3id.org/onto-bot#actsOn")},
        {URIRef("https://w3id.org/onto-bot#isAffordedBy")},
        {URIRef("https://w3id.org/onto-bot#hasNaturalLanguage")}
    ]

    class_requirements = {
        "observation-graph": required_class_observation,
        "action-graph": required_class_action
    }

    property_requirements = {
        "observation-graph": required_property_observation,
        "action-graph": required_property_action
    }

    main(runs, models, graph_types, methods, output_root, ontology_files, class_requirements, property_requirements)
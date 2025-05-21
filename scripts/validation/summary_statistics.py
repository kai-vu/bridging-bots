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
        # Try line-by-line fallback
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

def get_used_terms(kg_graph):
    used_classes = set(kg_graph.objects(None, RDF.type))
    used_properties = set(kg_graph.predicates(None, None))
    return used_classes, used_properties

def compute_compliance(required_groups, used_terms):
    # flatten all allowed terms from your groups
    allowed_terms = set().union(*required_groups)
    valid_terms = used_terms & allowed_terms
    total_used = len(used_terms)
    compliance_pct = (len(valid_terms) / total_used * 100) if total_used else 0
    return compliance_pct, len(valid_terms), total_used

def compute_grouped_coverage(required_groups, used_terms):
    matched_groups = 0
    for group in required_groups:
        if group & used_terms:
            matched_groups += 1
    coverage = (matched_groups / len(required_groups)) * 100 if required_groups else 0
    return coverage, matched_groups, len(required_groups)

def summarize_kg(kg_graph, required_class_groups, required_property_groups, full_parse, invalid_triples):
    used_classes, used_properties = get_used_terms(kg_graph)
    total_triples = len(kg_graph)

    class_coverage_pct, matched_class_groups, total_class_groups = compute_grouped_coverage(required_class_groups, used_classes)
    prop_coverage_pct, matched_prop_groups, total_prop_groups = compute_grouped_coverage(required_property_groups, used_properties)

    class_compliance_pct, valid_classes, total_used_classes = compute_compliance(required_class_groups, used_classes)
    prop_compliance_pct, valid_properties, total_used_properties = compute_compliance(required_property_groups, used_properties)

    return {
        'Full Parse OK': full_parse,
        'Total triples in KG': total_triples + invalid_triples,
        'Valid triples': total_triples,
        'Invalid triples': invalid_triples,

        'Distinct classes used': total_used_classes,
        'Class Compliance (valid/used)': f"{class_compliance_pct:.1f}% ({valid_classes}/{total_used_classes})",
        'Class Coverage (groups matched/defined)': f"{class_coverage_pct:.1f}% ({matched_class_groups}/{total_class_groups})",

        'Distinct properties used': total_used_properties,
        'Property Compliance (valid/used)': f"{prop_compliance_pct:.1f}% ({valid_properties}/{total_used_properties})",
        'Property Coverage (groups matched/defined)': f"{prop_coverage_pct:.1f}% ({matched_prop_groups}/{total_prop_groups})",
    }

def main(kg_paths, output_file, required_class_groups, required_property_groups):
    all_data = []

    for kg_path in kg_paths:
        print(f"Validating {kg_path}")

        kg_graph, full_parse_ok, invalid_triples = try_parse_graph(kg_path)
        summary = summarize_kg(kg_graph, required_class_groups, required_property_groups, full_parse_ok, invalid_triples)
        all_data.append(summary)

    df = pd.DataFrame(all_data, index=kg_paths)
    print(df)
    df.to_csv(output_file)

if __name__ == '__main__':

    output_observation = "../../output/validation/summary-statistics/summary_observation.csv"
    output_action = "../../output/validation/summary-statistics/summary_action.csv"

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
    ontology_action_graph = "../../ontology/ontoActionGraph.ttl"

    required_class_observation = [
        {URIRef("https://w3id.org/psr-action#Environment")},
        {
            URIRef("https://w3id.org/psr-action#Component"),
            URIRef("https://w3id.org/psr-action#Appliance"),
            URIRef("https://w3id.org/psr-action#Furniture"),
            URIRef("https://w3id.org/psr-action#Object"),
        },
        {
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Location"),
            URIRef("https://w3id.org/psr-action#CurrentLocation"),
            URIRef("https://w3id.org/psr-action#StandardLocation"),
        },
        {
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Affordance"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Closing"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Opening"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Delivering"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Holding"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#PickingUp"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#PuttingDown"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Pulling"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Pushing"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Grasping"),
        },
    ]
    required_property_observation = [
        {URIRef("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasComponent")},
        {URIRef("https://w3id.org/psr-action#hasAffordance")},
        {
            URIRef("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasLocation"),
            URIRef("https://w3id.org/psr-action#onTopOf"),
            URIRef("http://www.opengis.net/ont/geosparql#sfContains"),
            URIRef("http://www.opengis.net/ont/geosparql#sfWithin"),
            URIRef("http://www.opengis.net/ont/geosparql#sfOverlaps"),
        },
    ]

    required_class_action = [
        {URIRef("https://w3id.org/psr-action#Instruction")},
        {URIRef("https://w3id.org/psr-action#Workflow")},
        {URIRef("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Action")},
        {URIRef("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Agent")},
        {
            URIRef("https://w3id.org/psr-action#Component"),
            URIRef("https://w3id.org/psr-action#Appliance"),
            URIRef("https://w3id.org/psr-action#Furniture"),
            URIRef("https://w3id.org/psr-action#Object"),
        },
        {
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Affordance"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Closing"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Opening"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Delivering"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Holding"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#PickingUp"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#PuttingDown"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Pulling"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Pushing"),
            URIRef("http://www.ease-crc.org/ont/SOMA.owl#Grasping"),
        },
    ]
    required_property_action = [
        {URIRef("https://w3id.org/psr-action#hasWorkflow")},
        {URIRef("https://w3id.org/psr-action#hasAction")},
        {URIRef("https://w3id.org/psr-action#precedes")},
        {URIRef("https://w3id.org/psr-action#follows")},
        {URIRef("http://www.ease-crc.org/ont/SOMA.owl#isPerformedBy")},
        {URIRef("https://w3id.org/psr-action#actsOn")},
        {URIRef("https://w3id.org/psr-action#isAffordedBy")},
        {URIRef("https://w3id.org/psr-action#hasNaturalLanguage")},
    ]

    main(files_observation_graph, output_observation, 
         required_class_observation, required_property_observation)
    main(files_action_graph, output_action, 
         required_class_action, required_property_action)
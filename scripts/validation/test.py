import rdflib
from rdflib import RDF, OWL
import pandas as pd

def load_graph(file_path):
    g = rdflib.Graph()
    g.parse(file_path, format='turtle')
    return g

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

def summarize_kg(kg_graph, ont_classes, ont_properties):
    used_classes, used_properties = get_used_terms(kg_graph)
    total_triples = len(kg_graph)

    unexpected_classes = used_classes - ont_classes
    unexpected_properties = used_properties - ont_properties

    # Count typed instances with known classes
    typed_instances_known = sum(1 for s in set(kg_graph.subjects(RDF.type, None))
                                if any((s, RDF.type, c) in kg_graph for c in ont_classes))

    # Percent coverage
    classes_coverage = (len(used_classes & ont_classes) / len(used_classes) * 100
                        if used_classes else 0)
    properties_coverage = (len(used_properties & ont_properties) / len(used_properties) * 100
                           if used_properties else 0)

    return {
        'Total distinct classes used': len(used_classes),
        'Classes in ontology': len(ont_classes),
        'Unexpected classes count': len(unexpected_classes),
        'Unexpected classes (examples)': ', '.join([str(c) for c in list(unexpected_classes)[:5]]) or 'None',
        'Total distinct properties used': len(used_properties),
        'Properties in ontology': len(ont_properties),
        'Unexpected properties count': len(unexpected_properties),
        'Unexpected properties (examples)': ', '.join([str(p) for p in list(unexpected_properties)[:5]]) or 'None',
        'Total triples in KG': total_triples,
        'Typed instances with known classes': typed_instances_known,
        'Percentage classes coverage': f"{classes_coverage:.1f}%",
        'Percentage properties coverage': f"{properties_coverage:.1f}%"
    }

def main(ontology_path, kg_paths):
    # Load ontology
    ont_graph = load_graph(ontology_path)
    ont_classes, ont_properties = get_ontology_terms(ont_graph)

    all_data = []
    kg_names = []

    for kg_path in kg_paths:
        kg_graph = load_graph(kg_path)
        summary = summarize_kg(kg_graph, ont_classes, ont_properties)
        all_data.append(summary)
        kg_names.append(kg_path)

    df = pd.DataFrame(all_data, index=kg_names)
    print(df)

    # Optional: save to CSV or Excel
    df.to_csv('kg_alignment_summary.csv')
    # df.to_excel('kg_alignment_summary.xlsx')

if __name__ == '__main__':
    # Set your ontology and KG file paths here:
    ontology_file = 'your_ontology.ttl'
    kg_files = [
        'kg1.ttl',
        'kg2.ttl',
        'kg3.ttl',
        # Add more KG files here
    ]

    main(ontology_file, kg_files)
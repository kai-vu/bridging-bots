#!/usr/bin/env python
import math
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, OWL, XSD

class BasicOntologyPopulator:
    def __init__(self, ontology_path, namespace_uri, label_to_class):
        # Load the ontology from file and initialize the namespace and label-class mapping
        self.ontology_path = ontology_path
        self.graph = Graph()
        self.graph.parse(ontology_path, format="turtle")
        self.EX = Namespace(namespace_uri)
        self.label_to_class = label_to_class

    def sanitize_label(self, raw_label):
        # Normalize label: lowercase and replace spaces with underscores
        return raw_label.lower().strip().replace(" ", "_")

    def first_word(self, label_str):
        # Extract the first word of a label (used for fallback matching)
        return label_str.lower().split()[0]

    def find_individual_by_label(self, label_str):
        # Check if an individual with the given label already exists in the ontology
        query = f"""
        PREFIX ex: <{self.EX}>
        SELECT ?obj
        WHERE {{
          ?obj ex:hasName \"{label_str}\" .
        }}
        """
        results = self.graph.query(query)
        for row in results:
            return row.obj
        return None

    def get_or_create_individual(self, label_str):
        # Return existing individual or create a new one with the appropriate class
        existing = self.find_individual_by_label(label_str)
        if existing:
            return existing

        raw_key = label_str.lower().strip()
        if not raw_key:
            print(f"Label is empty; skipping creation.")
            return None

        norm_label = self.sanitize_label(label_str)

        # Try full label match; fallback to first word
        if raw_key in self.label_to_class:
            mapped_class = self.label_to_class[raw_key]
        else:
            parts = raw_key.split()
            if not parts:
                print(f"Label '{label_str}' is invalid; skipping creation.")
                return None
            fw = parts[0]
            if fw not in self.label_to_class:
                print(f"Label '{label_str}' not in dictionary; skipping creation.")
                return None
            mapped_class = self.label_to_class[fw]

        # Check that the mapped class exists in the ontology
        if (self.EX[mapped_class], RDF.type, OWL.Class) not in self.graph:
            print(f"Mapped class '{mapped_class}' not found in ontology; skipping '{label_str}'.")
            return None

        # Create the new individual and add it to the graph
        individual_uri = self.EX[norm_label]
        self.graph.add((individual_uri, RDF.type, self.EX[mapped_class]))
        self.graph.add((individual_uri, self.EX.hasName, Literal(label_str)))
        print(f"Created new individual '{norm_label}' as {mapped_class} with label '{label_str}'.")
        return individual_uri

    def update_individual_location(self, obj_uri, pos):
        # Set or update the individual's location using coordinates from sensor data
        loc_uri = self.EX[f"{obj_uri.split('#')[-1]}_Location"]
        self.graph.remove((loc_uri, None, None))
        self.graph.remove((obj_uri, self.EX.hasLocation, None))
        self.graph.add((loc_uri, RDF.type, self.EX.Location))
        self.graph.add((loc_uri, self.EX.hasCoordinate_X, Literal(pos[0], datatype=XSD.float)))
        self.graph.add((loc_uri, self.EX.hasCoordinate_Y, Literal(pos[2], datatype=XSD.float)))
        self.graph.add((obj_uri, self.EX.hasLocation, loc_uri))
        print(f"Updated location for {obj_uri} to ({pos[0]:.2f}, {pos[2]:.2f}).")

    def update_individual_orientation(self, obj_uri, orientation_value):
        # Set or update the individual's orientation
        orient_uri = self.EX[f"{obj_uri.split('#')[-1]}_Orientation"]
        self.graph.remove((orient_uri, None, None))
        self.graph.add((orient_uri, RDF.type, self.EX.Orientation))
        self.graph.add((orient_uri, self.EX.hasCoordinate_Z, Literal(orientation_value, datatype=XSD.float)))
        self.graph.remove((obj_uri, self.EX.hasOrientation, None))
        self.graph.add((obj_uri, self.EX.hasOrientation, orient_uri))
        print(f"Updated orientation for {obj_uri} to {orientation_value:.2f}.")

    def process_recognition(self, obj):
        # Main handler for a recognized object
        # Extract its label, position and orientation, and create/update the corresponding individual
        raw_label = obj.getModel()
        pos = obj.getPosition()
        orientation_value = pos[1]  # Typically the Y axis is used for orientation
        ind_uri = self.get_or_create_individual(raw_label)
        if ind_uri:
            self.update_individual_location(ind_uri, pos)
            self.update_individual_orientation(ind_uri, orientation_value)

    def save_ontology(self, destination_path):
        # Serialize and save the ontology back to a TTL file
        self.graph.serialize(destination=destination_path, format="turtle")
        print(f"Ontology saved to '{destination_path}'.")
        
        
class SpatialReasonerPopulator:
    def __init__(self, input_ttl, output_ttl, namespace_uri):
        # Load the ontology graph and set up the namespace
        self.input_ttl = input_ttl
        self.output_ttl = output_ttl
        self.graph = Graph()
        self.graph.parse(input_ttl, format="turtle")
        self.EX = Namespace(namespace_uri)

        # Spatial thresholds
        self.HORIZONTAL_THRESHOLD_NORMAL = 1.0
        self.HORIZONTAL_THRESHOLD_TABLE = 3.0
        self.VERTICAL_DELTA_MIN = 0.1

    def parse_locations(self):
        # Retrieve all individuals that have a location in the ontology
        query = f"""
        PREFIX ex: <{self.EX}>
        SELECT ?obj ?name ?x ?y
        WHERE {{
          ?obj ex:hasName ?name ;
               ex:hasLocation ?loc .
          ?loc ex:hasCoordinate_X ?x ;
               ex:hasCoordinate_Y ?y .
        }}
        """
        results = self.graph.query(query)
        objects = {}
        for row in results:
            obj_name = str(row.name)
            objects[obj_name] = {
                "uri": row.obj,
                "x": float(row.x.toPython()),
                "y": float(row.y.toPython())
            }
        return objects

    def get_object_type(self, obj_uri):
        # Get the rdf:type of an individual, but only if it matches allowed categories
        allowed_types = {self.EX.Appliance, self.EX.Furniture, self.EX.Utensil,
                         self.EX.KitchenAccessory, self.EX.Container, self.EX.Component}
        for s, p, o in self.graph.triples((obj_uri, RDF.type, None)):
            if o in allowed_types:
                return o
        return None

    def add_relation(self, subj, pred, obj):
        # Safely add a triple to the ontology if it doesn't already exist
        triple = (subj, pred, obj)
        if triple not in self.graph:
            self.graph.add(triple)
            print(f"Added relation: {subj.split('#')[-1]} {pred.split('#')[-1]} {obj.split('#')[-1]}")

    def compute_on_top_relations(self, objects):
        # Infer and assert spatial relationships like onTopOf, isPartOf, closeTo
        onTopOf = self.EX.onTopOf
        isDownOf = self.EX.isDownOf
        closeTo  = self.EX.closeTo
        isPartOf = self.EX.isPartOf

        best_support = {}  # upper -> (lower, dx)
        support_map = {}   # lower -> [upper...]

        names = list(objects.keys())
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    continue
                nameA = names[i]
                nameB = names[j]
                A, B = objects[nameA], objects[nameB]
                dx = abs(A["x"] - B["x"])
                dy = A["y"] - B["y"]  # vertical difference

                typeA = self.get_object_type(A["uri"])
                typeB = self.get_object_type(B["uri"])
                if typeA is None or typeB is None:
                    continue

                # Special rule for Appliance ↔ Component
                if typeA == self.EX.Appliance and typeB == self.EX.Component:
                    self.add_relation(B["uri"], isPartOf, A["uri"])
                    continue
                if typeA == self.EX.Component and typeB == self.EX.Appliance:
                    self.add_relation(A["uri"], isPartOf, B["uri"])
                    continue

                allowed = False
                if typeA == self.EX.Utensil:
                    if typeB in {self.EX.Appliance, self.EX.Utensil}:
                        allowed = True
                    elif typeB == self.EX.Furniture and nameB.lower().strip() == "table":
                        allowed = True
                elif typeA == self.EX.Appliance and typeB == self.EX.Appliance:
                    allowed = True

                horiz_thresh = self.HORIZONTAL_THRESHOLD_TABLE if nameB.lower().strip() == "table" else self.HORIZONTAL_THRESHOLD_NORMAL

                print(f"Comparing '{nameA}' vs '{nameB}': dx={dx:.3f}, dy={dy:.3f}, allowed={allowed}")

                if allowed and dx <= horiz_thresh and dy >= self.VERTICAL_DELTA_MIN:
                    current = best_support.get(A["uri"])
                    if current is None or dx < current[1]:
                        best_support[A["uri"]] = (B["uri"], dx)

        for upper_uri, (lower_uri, _) in best_support.items():
            self.add_relation(upper_uri, onTopOf, lower_uri)
            self.add_relation(lower_uri, isDownOf, upper_uri)
            support_map.setdefault(lower_uri, []).append(upper_uri)

        for lower_uri, uppers in support_map.items():
            if len(uppers) >= 2:
                for i in range(len(uppers)):
                    for j in range(i+1, len(uppers)):
                        self.add_relation(uppers[i], closeTo, uppers[j])
                        self.add_relation(uppers[j], closeTo, uppers[i])

    def run(self):
        # Main execution logic
        print(f"Loaded ontology from {self.input_ttl}.")
        objects = self.parse_locations()
        print(f"Found {len(objects)} objects with location data.")
        self.compute_on_top_relations(objects)
        self.graph.serialize(destination=self.output_ttl, format="turtle")
        print(f"Spatial reasoning complete. Updated ontology saved to {self.output_ttl}.")

       

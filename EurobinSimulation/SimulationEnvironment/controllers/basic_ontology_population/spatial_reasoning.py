#!/usr/bin/env python
import math
import rdflib
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, OWL, XSD

######################################################
# CONFIGURATION
######################################################

# Paths to ontology files
INPUT_TTL = "kitchen_ontology_updated.ttl"
OUTPUT_TTL = "kitchen_ontology_spatial.ttl"

# Namespaces
EX = Namespace("http://example.org/kitchen#")

# Thresholds (tweak as you see fit)
ON_TOP_DISTANCE_XZ = 0.2   # max horizontal distance to consider "directly above"
ON_TOP_MIN_DELTA_Y  = 0.2  # minimum vertical gap to be "above"
NEXT_TO_MAX_DISTANCE = 0.5 # how close in XZ to be considered "next to"
INSIDE_MAX_DISTANCE  = 0.15 # how close in XZ to consider "inside"

######################################################
# HELPER FUNCTIONS
######################################################
def parse_locations(g):
    """
    Returns a dict:
    {
      "plate":  {"uri": ex:plate, "x": 1.2, "y": 0.8, "z": -0.2},
      "fridge": {"uri": ex:fridge, "x": 0.0, "y": 0.0, "z": 0.0},
      ...
    }
    for every individual that has a :Location with data properties
    ex:hasCoordinate_X, ex:hasCoordinate_Y (and optionally orientation info).
    """
    results = g.query(f"""
    PREFIX ex: <{EX}>
    SELECT ?obj ?objName ?loc ?xVal ?yVal
    WHERE {{
      ?obj ex:hasName ?objName ;
           ex:hasLocation ?loc .
      ?loc ex:hasCoordinate_X ?xVal ;
           ex:hasCoordinate_Y ?yVal .
    }}
    """)

    objects = {}
    for row in results:
        obj_uri  = row.obj
        obj_name = str(row.objName.toPython())
        x_val    = float(row.xVal.toPython())
        y_val    = float(row.yVal.toPython())
        # We only have X and Y from the code snippet. 
        # If you also store "hasCoordinate_Z" or orientation, you can query that as well.

        objects[obj_name] = {
            "uri": obj_uri,
            "x": x_val,
            "y": 0.0,  # no direct "vertical" coordinate if you're using 2D in your code
            "z": y_val
        }
        # Explanation: your code is using hasCoordinate_X for the 'x' axis 
        # and hasCoordinate_Y for the 'z' axis in Webots. 
        # If your "vertical" is pos[1], you'd have to store that differently or query from hasCoordinate_Z.
    return objects

def add_relation(g, subject_uri, rel_uri, object_uri):
    """
    Utility to add a triple (subject_uri, rel_uri, object_uri)
    only if it doesn't already exist, to avoid duplicates.
    """
    triple = (subject_uri, rel_uri, object_uri)
    if triple not in g:
        g.add(triple)

def compute_spatial_relations(g, objects):
    """
    Using simple thresholds, compute:
      - onTopOf
      - nextTo
      - insideOf
    and add those relationships as object properties in the graph.
    """
    # Let's define URIs for these relations if not existing in your ontology
    onTopOf = EX.onTopOf
    nextTo  = EX.nextTo
    inside  = EX.insideOf
    
    # pairwise compare every object
    obj_names = list(objects.keys())
    for i in range(len(obj_names)):
        for j in range(i+1, len(obj_names)):
            nameA = obj_names[i]
            nameB = obj_names[j]
            dataA = objects[nameA]
            dataB = objects[nameB]
            # positions
            ax, ay, az = dataA["x"], dataA["y"], dataA["z"]
            bx, by, bz = dataB["x"], dataB["y"], dataB["z"]

            # 1) Check "on top of"
            # Suppose we define "A on top of B" if horizontal dist < ON_TOP_DISTANCE_XZ
            # and A is at a higher "y" coordinate than B by at least ON_TOP_MIN_DELTA_Y
            # But we have no real vertical dimension in your code. 
            # If you do store a "vertical" coordinate, you'd do:
            #   if abs(ax - bx) < ON_TOP_DISTANCE_XZ and abs(az - bz) < ON_TOP_DISTANCE_XZ and (ay - by) > ON_TOP_MIN_DELTA_Y:
            #       g.add((dataA["uri"], onTopOf, dataB["uri"]))
            # For example:

            # 2D distance in the "XZ" plane
            distXZ = math.sqrt((ax - bx)**2 + (az - bz)**2)

            # "next to" if distXZ < NEXT_TO_MAX_DISTANCE
            # Example approach:
            if distXZ < NEXT_TO_MAX_DISTANCE:
                add_relation(g, dataA["uri"], nextTo, dataB["uri"])
                add_relation(g, dataB["uri"], nextTo, dataA["uri"])

            # "inside" if distXZ < INSIDE_MAX_DISTANCE
            # This is naive. Real "inside" logic would require bounding volumes.
            if distXZ < INSIDE_MAX_DISTANCE:
                # We'll say A is inside B
                add_relation(g, dataA["uri"], inside, dataB["uri"])
                # or B is inside A, depends on the real scenario 
                # but we'll do just one direction as an example
                # add_relation(g, dataB["uri"], inside, dataA["uri"])

            # "on top of" if we had a vertical coordinate
            # For demonstration, let's pretend y is always 0. 
            # If you do store vertical in "ay" & "by", we can do:
            # if abs(ax - bx) < ON_TOP_DISTANCE_XZ and abs(az - bz) < ON_TOP_DISTANCE_XZ and (ay - by) > ON_TOP_MIN_DELTA_Y:
            #     add_relation(g, dataA["uri"], onTopOf, dataB["uri"])
            # elif abs(ax - bx) < ON_TOP_DISTANCE_XZ and abs(az - bz) < ON_TOP_DISTANCE_XZ and (by - ay) > ON_TOP_MIN_DELTA_Y:
            #     add_relation(g, dataB["uri"], onTopOf, dataA["uri"])
            pass

def main():
    # 1) Load ontology
    g = rdflib.Graph()
    g.parse(INPUT_TTL, format="turtle")
    
    # 2) Gather objects & their coords
    objects = parse_locations(g)
    
    # 3) Compute relations
    compute_spatial_relations(g, objects)
    
    # 4) Save to new file
    g.serialize(destination=OUTPUT_TTL, format="turtle")
    print(f"Spatial reasoning done. Results saved to {OUTPUT_TTL}.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
from WebotsOntologyPopulation import SpatialReasonerPopulator

# ------------------------------
# CONFIGURATION
# ------------------------------
INPUT_TTL = "C:/Users/francesca/Documents/eurobin_ontology_updated.ttl"
OUTPUT_TTL = "C:/Users/francesca/Documents/eurobin_ontology_spatial.ttl"
NAMESPACE_URI = "http://example.org/kitchen#"

# ------------------------------
# RUN SPATIAL REASONING
# ------------------------------
if __name__ == "__main__":
    reasoner = SpatialReasonerPopulator(
        input_ttl=INPUT_TTL,
        output_ttl=OUTPUT_TTL,
        namespace_uri=NAMESPACE_URI
    )
    reasoner.run()

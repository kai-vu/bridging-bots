import re
from rdflib import Graph, Namespace
from rdflib.namespace import RDF
from collections import defaultdict, Counter
import pandas as pd

SH = Namespace("http://www.w3.org/ns/shacl#")
NS1 = Namespace("http://example.org/validation/")

report_files = [
    "../../output/run1/validation/shacl/shacl_validation_report.ttl",
    "../../output/run2/validation/shacl/shacl_validation_report.ttl",
    "../../output/run3/validation/shacl/shacl_validation_report.ttl",
    "../../output/run4/validation/shacl/shacl_validation_report.ttl",
    "../../output/run5/validation/shacl/shacl_validation_report.ttl",
    "../../output/run6/validation/shacl/shacl_validation_report.ttl",
    "../../output/run7/validation/shacl/shacl_validation_report.ttl",
    "../../output/run8/validation/shacl/shacl_validation_report.ttl",
    "../../output/run9/validation/shacl/shacl_validation_report.ttl",
    "../../output/run10/validation/shacl/shacl_validation_report.ttl",
]

aggregated = defaultdict(lambda: {
    "total_runs": 0,
    "passed_runs": 0,
    "failed_runs": 0,
    "total_violations": 0,
    "focus_nodes": Counter(),
    "constraint_components": Counter(),
    "result_paths": Counter(),
})

def extract_config_info(source_file):
    match = re.search(r'run\d+/([^/]+)/([^/]+)/([^/]+)/kg\.ttl', source_file)
    if match:
        model, graph_type, setting = match.groups()
        return model, graph_type, setting
    return None, None, None

for report_path in report_files:
    g = Graph()
    g.parse(report_path, format="turtle")

    for report in g.subjects(RDF.type, SH.ValidationReport):
        source_file_node = g.value(report, NS1.sourceFile)
        if not source_file_node:
            continue

        source_file = str(source_file_node)
        model, graph_type, setting = extract_config_info(source_file)
        if not all([model, graph_type, setting]):
            continue

        key = (model, graph_type, setting)
        aggregated[key]["total_runs"] += 1

        conforms = g.value(report, SH.conforms)
        if conforms.toPython():
            aggregated[key]["passed_runs"] += 1
        else:
            aggregated[key]["failed_runs"] += 1
            for violation in g.objects(report, SH.result):
                aggregated[key]["total_violations"] += 1
                aggregated[key]["focus_nodes"][str(g.value(violation, SH.focusNode))] += 1
                aggregated[key]["constraint_components"][str(g.value(violation, SH.sourceConstraintComponent))] += 1
                rp = g.value(violation, SH.resultPath)
                if rp:
                    aggregated[key]["result_paths"][str(rp)] += 1

summary = []
for (model, graph_type, setting), data in aggregated.items():
    summary.append({
        "model": model,
        "graph_type": graph_type,
        "setting": setting,
        "total_runs": data["total_runs"],
        "passed_runs": data["passed_runs"],
        "failed_runs": data["failed_runs"],
        "total_violations": data["total_violations"],
        "top_focus_nodes": ", ".join(f"{k} ({v})" for k, v in data["focus_nodes"].most_common(3)),
        "top_constraint_components": ", ".join(f"{k.split('#')[-1]} ({v})" for k, v in data["constraint_components"].most_common(3)),
        "top_result_paths": ", ".join(f"{k.split('#')[-1]} ({v})" for k, v in data["result_paths"].most_common(3)),
    })

df = pd.DataFrame(summary)
df.to_csv("../../output/summary-statistics/shacl_validation_summary.csv", index=False)
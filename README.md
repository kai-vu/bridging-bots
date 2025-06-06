# 🤖 Bridging Bots: From Perception to Action via Multimodal-LMs and Knowledge Graphs

This repository contains the code, ontology, prompts, data, and generated outputs for the paper:

**“Bridging Bots: from Perception to Action via Multimodal-LMs and Knowledge Graphs”**  
*19th Conference on Neurosymbolic Learning and Reasoning, 2025*  
*Margherita Martorana, Francesca Urgese, Mark Adamik, Ilaria Tiddi*  
Vrije Universiteit Amsterdam

---

## Overview

Service robots must interpret complex environments and plan actions accordingly. This project presents a **neurosymbolic framework** that integrates:

- **Raw visual input** (Webots simulation)
- **Natural language task descriptions**
- **Multimodal large language models (MLLMs)**
- **Ontology-based symbolic reasoning**

The goal: to explore how neural models and symbolic representations can be effectively combined to generate structured, context-aware representations of environments and action sequences for service robots. 

---

## ⚙️ Pipeline

![Workflow Overview](https://github.com/user-attachments/assets/271d14ff-d5d3-464a-a594-e359bcab354f)


> The figure summarizes the symbolic integration paths for generating structured KGs from different input modalities.

The pipeline builds two graphs:
- **Observation Graph**: what the state of the environment is 
- **Action Graph**: the sequence of actions needed to complete a given task

These are generated using different integration strategies combining **vision**, **language**, and **ontology schemas**.

---

## Experimental Setup

We tested 5 state-of-the-art MLLMs:

| Model               | Type             | Notes |
|--------------------|------------------|-------|
| LLaVA + LLaMA 3    | Modular           | Visual + Text (manually linked) |
| LLaMA 4 Scout      | Unified multimodal | Long-context optimized |
| LLaMA 4 Maverick   | Unified multimodal | High performance |
| GPT-4.1-nano       | Unified multimodal | Lightweight, fast |
| GPT-o1             | Unified multimodal | High accuracy, slower |

Each model was tested using 4 integration methods:
- `dpe`: Dynamic Path Extractor
- `d2kg`: Description to Knowledge Graph
- `d2kg-rag`: with Retrieval-Augmented Generation
- `i2kg`: Image to Knowledge Graph 

---

## Example: Input to KG

![Example](https://github.com/user-attachments/assets/6648c08f-d48d-4f61-8623-2d39b7a4b499)

From multiple viewpoints of a Webots kitchen simulation, the system generates:

- A **symbolic description** of the environment  
- A **sequence of robot actions** (e.g., Pick up jar → Open fridge → Put jar inside)

Each element in the resulting graph follows the formal **OntoBOT ontology**.

---

## 📊 Results Summary

- 🥇 **LLaMA 4 Maverick** and **GPT-o1** consistently outperformed other models in:
  - Ontology **compliance** (valid classes/properties used)
  - **Coverage** (how much of the ontology was represented)
  - **SHACL conformance** (structural validity)

- 📉 **GPT-4.1-nano** and some integration methods (e.g., `dpe` for LLaMA, `i2kg` for LLaVA) often failed to produce valid graphs.

- 📈 Variability across runs was non-negligible, even for top models, highlighting the challenge of **consistent ontology-compliant generation**.

---

## Repository Structure
```
bridging-bots/
│
├── ontology/ # OntoBOT ontology files (TTL, SHACL) 
├── images/ # SImulation environment screenshots 
├── output/ # Generated KGs (observation & action graphs)
├── scripts/ # Model interaction and KG construction scripts
├── webotsFiles/ # Webots simulation scripts
├── LICENSE
└── README.md # This file
```
---

If you use this work, please cite our paper:

```bibtex
@inproceedings{martorana2025bridging,
  title     = {Bridging Bots: from Perception to Action via Multimodal-LMs and Knowledge Graphs},
  author    = {Martorana, Margherita and Urgese, Francesca and Adamik, Mark and Tiddi, Ilaria},
  booktitle = {Proceedings of the 19th Conference on Neurosymbolic Learning and Reasoning},
  year      = {2025},
  publisher = {PMLR}
}



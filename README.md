# Patient Timeline Agent

**Unified patient history and an agentic visit-briefing system, built on a knowledge graph.**

---

## The Problem

A patient's medical history doesn't live in one place. The same person shows up at several hospitals over the years, and each one only sees its own slice — its own spelling of their name, its own formatting of their date of birth, its own notes. Nobody has the full picture, and a doctor meeting that patient for a new visit has no fast way to see what actually matters from everything that came before.

---

## The Solution

Patient Timeline Agent is two systems working together. **Part A** re-links a patient's fragmented records across hospitals, pulls clinical facts out of free-text notes, and merges everything into one chronological, source-attributed timeline — flagging disagreements between hospitals instead of silently trusting one. **Part B** is an agent that takes a new complaint, queries that unified history, and returns a relevance-filtered, confidence-aware briefing instead of dumping the entire record on the doctor.

> A doctor opens a new patient's file, types in today's complaint, and gets back exactly what's relevant — a HIGH-severity allergy flag, a note on an unconfirmed single-source diagnosis, a clarifying question if the complaint's too vague to act on — instead of scrolling through years of visits.

All data is synthetic (Synthea-generated). No real patient data is involved.

---

## Architecture

```
Fragmented hospital records (same patient, inconsistent identity per hospital)

          ↓

Entity Resolution (rapidfuzz)
  └── Re-links corrupted name/DOB/address/SSN fields to the real patient

          ↓

Biomedical NER + Normalization
  └── Extracts diagnoses, symptoms, medications, allergies from free-text notes
  └── Collapses surface text to one canonical concept per idea

          ↓

Conflict-Aware Timeline Merge
  └── One chronological record per patient, sourced, corroboration-flagged

          ↓

Neo4j Knowledge Graph
  └── Patient → Visit → Hospital / Concept, with pre-computed HAS_FACT trust state

          ↓

Visit Agent (LLM, graph-grounded)
  └── New complaint in → relevance-filtered, grounded briefing out
```

---

## Knowledge Graph

Part A's output isn't a pile of JSON — it's loaded into Neo4j so the agent can traverse it instead of scanning files:

```
(:PTA_Patient)-[:HAD_VISIT]->(:PTA_Visit)-[:AT_HOSPITAL]->(:PTA_Hospital)
(:PTA_Visit)-[:MENTIONS {text, match_method}]->(:PTA_Concept)
(:PTA_Patient)-[:HAS_FACT {status, reported_by, num_hospitals}]->(:PTA_Concept)
```

`PTA_Concept` is one node per distinct normalized idea (e.g. one "Penicillin allergy" node, however many visits or hospitals mention it), so a patient's clinical picture is a few graph hops away. The `HAS_FACT` edge is where corroboration lives as queryable state — its `status` (`confirmed` / `single_source` / `single_hospital_patient`) is computed once at load time, so the agent asks the graph "what does this patient have, and how sure are we" instead of recomputing trust per query.

---

## Agentic Visit Workflow

Given a patient and today's complaint, the agent:

1. Pulls that patient's full fact/medication graph in one query — a pre-corroborated summary, not raw visit history.
2. Hands that summary plus the complaint to an LLM behind a structured tool schema, which returns only relevance-filtered, confidence-aware findings — or a clarifying question if the complaint's too vague to act on.
3. Grounds every red flag in the schema itself: the model must name the exact fact it's citing, copied verbatim from the patient's actual data, so nothing ungrounded gets surfaced.

Verified behavior: correctly surfaces a single-source penicillin allergy as a HIGH-severity flag when amoxicillin is proposed (and downgrades it to LOW relevance for an unrelated complaint); asks a clarifying question for a vague complaint instead of guessing; infers a likely treatment and catches a real drug interaction against it without being told what the treatment would be.

---

## Sample Output

```json
{
  "status": "ready",
  "relevant_history": [
    {
      "fact": "Penicillin allergy",
      "related_canonical": "Penicillin allergy",
      "confirmation_status": "single_source",
      "why_relevant": "Proposed treatment (amoxicillin) is a penicillin-class antibiotic."
    }
  ],
  "red_flags": [
    {
      "flag_type": "relevant_allergy",
      "related_canonical": "Penicillin allergy",
      "description": "Patient has a recorded penicillin allergy; amoxicillin is penicillin-class.",
      "severity": "high"
    }
  ],
  "uncertainty_notes": [
    "Penicillin allergy is reported by only one hospital and has not been independently corroborated."
  ]
}
```

---

## Pipeline

| Stage | Script                         | What it does                                                                                                        |
| ----- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| 1     | `01_shatter_and_corrupt.py`  | Splits one patient population across simulated hospitals, deliberately corrupting each hospital's view of identity. |
| 2     | `03_entity_resolution.py`    | Fuzzy-matches corrupted records back to the same real patient.                                                      |
| 3     | `04_extract_entities.py`     | Pretrained biomedical NER over free-text notes — diagnoses, symptoms, medications, allergies.                      |
| 4     | `06_merge_timeline.py`       | Builds the per-patient timeline from predicted clusters, flagging single-source facts.                              |
| 5     | `07_load_neo4j.py`           | Loads the timeline into Neo4j as a graph.                                                                           |
| 6     | `08_visit_agent.py`          | The agent — complaint in, grounded briefing out.                                                                   |
| 7     | `09_patient_history_view.py` | Combined report: briefing on top, full annotated history below.                                                     |

Supporting scripts (note generation, concept normalization, name/DOB patient lookup, and a shared synonym vocabulary) round out the pipeline without needing their own writeup.

---

## Project Structure

```
patient-timeline-agent/
├── setup.sh                           # venv + data download + pipeline runner
├── streamlit_app.py                   # browser front end — check-in to briefing
│
├── scripts/
│   ├── 01_shatter_and_corrupt.py      # splits + corrupts identity across hospitals
│   ├── 02_generate_notes.py           # synthesizes free-text clinical notes
│   ├── 03_entity_resolution.py        # re-links patients across hospitals
│   ├── 04_extract_entities.py         # biomedical NER
│   ├── 05_normalize_entities.py       # collapses text to canonical concepts
│   ├── 06_merge_timeline.py           # conflict-aware timeline merge
│   ├── 07_load_neo4j.py               # loads the knowledge graph
│   ├── 08_visit_agent.py              # the visit-briefing agent
│   ├── 09_patient_history_view.py     # combined report (CLI)
│   ├── 10_lookup_patient.py           # name + DOB → patient id
│   └── vocab.py                       # shared synonym / exclusion dictionary
│
└── data/
    ├── hospitals/<hospital>/          # per-hospital patient + note records
    ├── entity_resolution/             # predicted clusters
    ├── unified_timelines/             # merged per-patient JSON
    └── ground_truth/                  # answer key for evaluation
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # fill in NEO4J_* and OPENAI_API_KEY
```

Steps 1–6 need no external services. The graph load needs a Neo4j instance (Aura Free works fine). The visit agent needs an OpenAI API key with available credit.

---

## Known Limitations

- Drug-interaction checking relies on the model's general medical knowledge, not a real pharmacological interaction database.
- STEMI / NSTEMI / history-of-MI collapse to one canonical concept, losing a real clinical distinction.
- Procedures/treatments and a general severity rating for diagnoses/symptoms aren't tracked yet.
- Note generation and NER are shaped around this project's own synthetic data and wouldn't generalize to arbitrary clinical text without further work.

---

## Roadmap

- [ ] Procedure/treatment tracking (Synthea's `procedures.csv` isn't pulled in yet)
- [ ] Severity ratings for diagnoses/symptoms, not just allergies
- [ ] A real drug-interaction database in place of model knowledge
- [ ] UMLS-backed normalization instead of a hand-built synonym dictionary

---

## Tech Stack

`Python` · `Neo4j` · `rapidfuzz` · `Hugging Face Transformers (biomedical NER)` · `OpenAI` · `Streamlit`

---

*Built as a portfolio project demonstrating applied graph + agentic system design for a real healthcare workflow.*

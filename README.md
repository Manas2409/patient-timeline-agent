# Unified Patient History & Smart Visit Agent

Simulates the real-world problem of a patient's medical history being scattered
across multiple hospitals that don't share records, then builds a system that
(A) re-links and reconciles that history, and (B) briefs a doctor on what's
relevant to a *new* visit instead of dumping the full record.

All data is synthetic (Synthea-generated). No real patient data is involved.

## Pipeline

Run from the project root with the venv's Python, in order (01 → 10). The scripts
that carry the actual ideas worth reading:

| # | Script                         | What it does                                                                                                                                                                                                                                   |
| - | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `01_shatter_and_corrupt.py`  | Splits one unified patient population across simulated hospital networks and deliberately corrupts each hospital's view of identity (name, DOB, address, SSN) — this is the entity-resolution problem the rest of Part A solves.              |
| 3 | `03_entity_resolution.py`    | Fuzzy-matches (rapidfuzz) every hospital-local record against every other on the corrupted identity fields to re-link records belonging to the same real patient, feeding the graph in Part A without ever seeing the ground-truth answer key. |
| 4 | `04_extract_entities.py`     | Runs a pretrained biomedical NER model (`d4data/biomedical-ner-all`) over each note to pull out diagnoses, symptoms, medications, and allergies, with negation detection and generic-term filtering.                                         |
| 6 | `06_merge_timeline.py`       | Builds one chronological timeline per real patient from the *predicted* clusters (not the ground truth), and flags any fact reported by only one hospital instead of silently treating one source as ground truth.                           |
| 7 | `07_load_neo4j.py`           | Loads every timeline into Neo4j as a graph so later steps query it instead of scanning JSON.                                                                                                                                                   |
| 8 | `08_visit_agent.py`          | **Part B.** Given a patient + today's complaint, queries the graph and asks an LLM (OpenAI `gpt-5`) for a relevance-filtered, confidence-aware doctor briefing — or a clarifying question if the complaint's too vague to act on.     |
| 9 | `09_patient_history_view.py` | **The combined report.** Runs script 8's briefing on top, then the full annotated history below it as reference, with items the briefing flagged called out inline.                                                                      |

Supporting scripts fill in the rest of the pipeline without needing their own
write-up: `02_generate_notes.py` (synthesizes free-text clinical notes from
Synthea's structured tables, since the raw data has none), `05_normalize_entities.py`
(collapses extracted text to one canonical concept per idea, e.g. "heart attack" /
"myocardial infarction"), `10_lookup_patient.py` (resolves a name + DOB to a patient
id, since real intake starts there, not with a database key), and `vocab.py` (the
shared synonym/exclusion dictionary scripts 2 and 5 both depend on, kept in one
place so they can't drift out of sync).

## Knowledge graph

Part A's output isn't just a pile of JSON — it's loaded into Neo4j so the agent in
Part B can traverse it instead of scanning files:

```
(:PTA_Patient)-[:HAD_VISIT]->(:PTA_Visit)-[:AT_HOSPITAL]->(:PTA_Hospital)
(:PTA_Visit)-[:MENTIONS {text, match_method}]->(:PTA_Concept)
(:PTA_Patient)-[:HAS_FACT {status, reported_by, num_hospitals}]->(:PTA_Concept)
```

`PTA_Concept` is one node per distinct normalized idea (e.g. one "Penicillin
allergy" node, regardless of how many visits or hospitals mention it), so a
patient's full clinical picture is a handful of graph hops away rather than
something to re-derive by scanning every visit every time. The `HAS_FACT` edge
is where the corroboration logic from `06_merge_timeline.py` actually lives as
queryable state: its `status` (`confirmed` / `single_source` /
`single_hospital_patient`) and `reported_by` list are computed once at load time,
so the agent can ask the graph directly "what does this patient have, and how
sure are we" instead of re-computing trust per query.

## Agentic visit workflow (Part B)

`08_visit_agent.py` is the part of this project worth spending the most time on.
Given a patient and today's complaint, it:

1. Pulls that patient's full `HAS_FACT`/medication graph in one query — not the
   raw visit history, a pre-corroborated summary.
2. Hands that summary plus the complaint (and optional proposed treatment) to an
   LLM (OpenAI `gpt-5`) behind a structured tool schema, instructed to return
   only relevance-filtered, confidence-aware findings — or a clarifying question
   if the complaint is too vague to act on safely.
3. Grounds every red flag in the schema itself: the model has to name the exact
   `related_canonical` fact it's citing, so nothing gets surfaced that isn't
   actually on the patient's chart (see "known simplifications" below for how
   this was hardened after catching the model citing ungrounded interactions).

Verified behavior across several real scenarios:

- Surfaced a single-source penicillin allergy as a HIGH-severity red flag when
  amoxicillin was the proposed treatment, and correctly downgraded that same
  allergy to LOW relevance for an unrelated chest-pain complaint.
- Asked a clarifying question for a vague complaint ("not feeling well") instead
  of guessing, then correctly incorporated the answer once given (chest pain ->
  flagged a PDE5-inhibitor/nitrate interaction caveat on its own).
- Inferred a likely treatment (TMP-SMX for a UTI complaint) and caught a real
  ACE-inhibitor/ARB interaction against it without being told what the
  treatment would be.

`09_patient_history_view.py` wraps this agent into the combined report (briefing
on top, full annotated history below), and `streamlit_app.py` wraps that into an
interactive check-in flow.

## Streamlit app

`streamlit_app.py` is a browser front end for the realistic check-in -> report
workflow, built entirely on top of scripts 9 and 10 (dynamic import, same pattern
those scripts already use on each other) -- it adds no new matching, extraction, or
agent logic of its own, only presentation.

```
streamlit run streamlit_app.py
```

Flow: a check-in form (name + DOB, matching script 10's CLI) resolves to a patient,
then a complaint form generates the same combined briefing + full history as script
9, rendered as a page -- including handling the agent's clarifying-question path
interactively (answer in a box, resubmit) instead of needing a second CLI call.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # then fill in NEO4J_* and OPENAI_API_KEY
```

Steps 1-6 need no external services. Step 7 needs a Neo4j instance (Aura Free
works fine). Step 8 needs an OpenAI API key with available credit.

## Current status

Both parts run end-to-end; Part A's entity resolution and timeline merge are
tested against a known-answer crosswalk, and Part B's agentic behavior is covered
above. Part B uses OpenAI (`gpt-5`, function calling) rather than Claude, a
provider switch made for this project's own billing reasons, not a technical
requirement — swapping back just means rewriting `call_agent()`.

## Known simplifications

- STEMI / NSTEMI / history-of-MI all collapse to one canonical "Myocardial
  infarction" concept in normalization, losing a real clinical distinction a
  UMLS-backed system would keep.
- Medication normalization keeps only the drug name, stripping dose/form; a
  combination drug's secondary ingredients (e.g. the oxycodone in a combo
  acetaminophen/oxycodone product) are tracked in the vocabulary but the medication
  is canonicalized under its first-listed ingredient.
- Drug-interaction checking in the visit agent relies on the model's general medical
  knowledge, not a real pharmacological interaction database — fine for a project
  demo, not for actual clinical use.
- The allergy sentence patterns in script 4 are shaped around this project's own
  note templates (script 2); they wouldn't generalize to arbitrary clinical text
  without a real relation-extraction step.
- **Not yet tracked at all:** procedures/treatments (Synthea's `procedures.csv` was
  never pulled into the pipeline) and a general severity rating for diagnoses/symptoms
  (the NER model has an unused `Severity` label, and note generation doesn't phrase
  severity explicitly enough for it to extract anything meaningful yet). Allergy
  severity/reaction *is* tracked (script 6), since that data already existed
  structurally in Synthea and just needed wiring through.
- `09_patient_history_view.py`'s relevance highlighting hit the same bug class as
  script 5's normalization: a short entity fragment (e.g. "et", "Nat" — NER
  truncation artifacts) can score a coincidentally high `partial_ratio` match against
  any long enough description text. Fixed with the same `MIN_FUZZY_LENGTH`-style
  guard used in script 5 -- but a second, more interesting case of the same root
  problem showed up against real free text: fuzzy-matching an entity's canonical name
  against the *model's own prose* (not a fixed vocabulary) is unreliable in both
  directions. A coincidental match ("Norinyl," a birth-control pill, scored 71 against
  an unrelated ACE-inhibitor interaction description) and a missed match (the model's
  phrasing didn't happen to name "lisinopril" literally enough to clear the threshold)
  both occurred in real runs. Fixed properly by adding a `related_canonical` field to
  the agent's structured output (08_visit_agent.py) -- the model now names the exact
  canonical term itself, so 09 matches it exactly instead of fuzzy-guessing from
  prose, with fuzzy matching kept only as a fallback for untagged entries.
- **The visit agent was citing drug interactions with medications the patient was
  never recorded as taking** (e.g. flagging methotrexate/warfarin/allopurinol
  interactions with amoxicillin when none of those three appeared anywhere in the
  patient's actual medication history) -- generically-true pharmacology trivia, not
  a finding grounded in this patient's chart. Fixed by tightening both the system
  prompt and the tool schema's field descriptions to require every red flag to name
  a medication/fact that actually appears in the provided patient data, with "no
  interaction apparent from the recorded medications" as an explicit, often-correct
  answer -- verified the model will still correctly flag a *real* interaction
  (lisinopril/losartan vs. TMP-SMX) when the drug genuinely is on the patient's chart.

## Troubleshooting notes

- **`pip-system-certs` conflicts with the Anthropic SDK** (infinite recursion in
  SSL setup) because both it and that SDK's own `httpx2` transport try to inject
  `truststore` into `ssl.SSLContext` -- only relevant if you switch script 8 back
  to Anthropic; it doesn't affect the current OpenAI-based version. It was needed
  once, transiently, to get HuggingFace Hub downloads working in script 4's
  initial run on this network; it is **not** installed by default and isn't in
  `requirements.txt`. Only reinstall it if you need to re-run script 4 and hit a
  HuggingFace SSL error.
- **gpt-5 needs headroom in `max_completion_tokens` beyond what you'd expect.**
  As a reasoning model, its hidden reasoning tokens count against that budget --
  too tight a limit (we initially tried 2000) can exhaust it before any tool-call
  output is emitted, failing with no error beyond "model did not return the
  expected tool call". Script 8 uses 6000 for this reason.
- **Neo4j Aura Free instances auto-pause after inactivity.** If script 7 or 8
  fails with a TLS connection reset (not a DNS or auth error), check
  console.neo4j.io and resume the instance — it can take a minute to come back,
  and DNS for a *newly created* instance can take a couple of minutes to
  propagate the first time.
- **This Neo4j instance is shared with the `propcrm` project** (a separate,
  unrelated app using `User`/`Lead`/`Property`/`Account`/`Payment`/`EmailMessage`
  labels) — a deliberate choice to keep sharing rather than provision a fully
  separate instance. Script 7 only ever touches its own `PTA_`-prefixed labels,
  so it can't damage propcrm's data, but the reverse isn't guaranteed: if
  propcrm's code ever does a full database wipe, this project's graph data goes
  with it (this happened once already, cause unconfirmed). Recovery is just
  `python scripts/07_load_neo4j.py` again — it rebuilds everything from
  `data/unified_timelines/*.json`, which isn't affected by anything that happens
  in Neo4j.

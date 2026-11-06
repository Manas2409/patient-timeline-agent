# Unified Patient History & Smart Visit Agent

Simulates the real-world problem of a patient's medical history being scattered
across multiple hospitals that don't share records, then builds a system that
(A) re-links and reconciles that history, and (B) briefs a doctor on what's
relevant to a *new* visit instead of dumping the full record.

All data is synthetic (Synthea-generated). No real patient data is involved.

## Pipeline

Run from the project root with the venv's Python, in order:

| # | Script | What it does | Key output |
|---|---|---|---|
| 1 | `01_shatter_and_corrupt.py` | Splits Synthea's 108 unified patients across 13 simulated regional hospital networks (grouped from Synthea's 278 real orgs by MA ZIP geography). Deliberately corrupts each hospital's view of patient identity (name formatting, DOB format, address completeness, SSN presence) so the same person looks different at every hospital — this is the entity-resolution problem to solve. | `data/hospitals/<slug>/patients.csv` + `data/ground_truth/patient_crosswalk.csv` (answer key) |
| 2 | `02_generate_notes.py` | Synthea's sample data has no free-text notes, only structured tables. This synthesizes a realistic clinical note per encounter from the structured fields, injecting lay/clinical synonym variation ("heart attack" vs "myocardial infarction") and separating genuine clinical conditions from Synthea's social/administrative findings (employment status, housing, etc.). | `data/hospitals/<slug>/notes.csv` + `notes_gold_entities.csv` (NER answer key) |
| 3 | `03_entity_resolution.py` | Fuzzy-matches (rapidfuzz) every hospital-local patient record against every other, using only the corrupted name/DOB/address/SSN fields, to re-link records belonging to the same real patient. Blocks by (gender, birth year) for efficiency. | `data/entity_resolution/predicted_clusters.csv`, `pair_scores.csv` — **scores 100% precision/recall against ground truth** on this dataset |
| 4 | `04_extract_entities.py` | Runs a pretrained biomedical NER model (`d4data/biomedical-ner-all`, BERT/MACCROBAT) over every note to extract DIAGNOSIS, SYMPTOM, MEDICATION, ALLERGY, DATE. Includes negation detection and generic-term filtering. | `data/hospitals/<slug>/extracted_entities.csv` |
| 5 | `05_normalize_entities.py` | Maps extracted surface text to one canonical concept per idea (so "heart attack"/"myocardial infarction"/"NSTEMI" collapse together) via exact synonym lookup + fuzzy matching against Synthea's own vocabulary. Explicitly flags genuinely ambiguous matches instead of guessing. | `data/hospitals/<slug>/normalized_entities.csv` |
| 6 | `06_merge_timeline.py` | Uses the *predicted* entity-resolution clusters (not the ground truth) to build one timeline per real patient across all their hospitals, sorted by date, with source attribution. Flags any fact reported by only one of a patient's hospitals as `unconfirmed_single_source`. Also enriches DIAGNOSIS/MEDICATION entities with an `outcome` (resolved-by-date vs ongoing, from each record's STOP date) and ALLERGY entities with `reaction`/`severity` (from Synthea's own structured allergy fields) — both matched back to extracted entities via fuzzy text matching. | `data/unified_timelines/patient_<id>.json` |
| 7 | `07_load_neo4j.py` | Loads every patient timeline into Neo4j as a graph (`PTA_Patient`, `PTA_Hospital`, `PTA_Visit`, `PTA_Concept` nodes; `HAD_VISIT`/`AT_HOSPITAL`/`MENTIONS`/`HAS_FACT` relationships), so later steps can query instead of scanning JSON. Also picks one representative `display_name`/`display_dob` per patient (the most complete-looking of their hospital-local records) and stores it on the `PTA_Patient` node, purely so reports can show a human name instead of a bare cluster id. | Graph in the Neo4j instance configured in `.env` |
| 8 | `08_visit_agent.py` | **Part B.** Given a patient + today's complaint (+ optional proposed treatment), queries the graph for that patient's compact fact/medication summary, and asks the model (OpenAI `gpt-5`) to produce a relevance-filtered, confidence-aware doctor-facing briefing — or ask a clarifying question if the complaint is too ambiguous to act on. | Printed briefing |
| 9 | `09_patient_history_view.py` | **The combined report.** If `--complaint` is given, calls script 8's agent (reused, not duplicated) and prints its briefing + red flags first — "what matters right now" — then the *complete* chronological history below it as reference: every visit, every entity, with confidence/outcome/severity/reaction and confirmed-vs-single-source status on everything, with an inline `RELEVANT TODAY -- <why>` note on items the briefing called out. Visits with no extracted clinical findings and non-clinical SDOH/administrative noise are filtered out of the display (not the data) so "complete" doesn't mean "cluttered." Without `--complaint`, it's just the full history (no briefing section). | Printed combined report |
| 10 | `10_lookup_patient.py` | **The missing front door.** Every other script above assumes you already know a patient's `cluster_id` — but a real intake process starts with a name and DOB, not a database key. Takes a name/DOB (+ optional middle name/address/SSN) and fuzzy-matches it against every hospital-local record using script 3's exact scoring functions (reused, not reimplemented), returning the matching `cluster_id` or "no match — new patient." | Printed match (or no-match) result |

`vocab.py` holds the shared synonym dictionary and admin/SDOH exclusion list used by
both script 2 (note generation) and script 5 (normalization), so the two can't drift
out of sync.

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

**Both parts complete and verified end-to-end.**

- **Part A (unified record builder):** Steps 1-7 all run end-to-end; entity
  resolution is 100% accurate against ground truth, and extraction/normalization
  numbers are honestly measured (see below).
- **Part B (visit agent):** Uses OpenAI (`gpt-5`, function calling) rather than
  Claude -- a provider switch made for this project's own billing reasons, not a
  technical requirement; swapping back just means rewriting `call_agent()`.
  Verified against several scenarios: correctly surfaced a single-source
  penicillin allergy as a HIGH-severity red flag when amoxicillin was proposed
  (and correctly downgraded that same allergy to LOW relevance for an unrelated
  chest-pain complaint), correctly asked a clarifying question for a vague
  complaint ("not feeling well") instead of guessing, correctly incorporated
  the clarification once given (chest pain -> flagged a PDE5-inhibitor/nitrate
  interaction caveat on its own), and correctly inferred a likely treatment
  (TMP-SMX for a UTI complaint) and caught a real ACE-inhibitor/ARB interaction
  against it without being told what the treatment would be.
- **Patient lookup + combined report:** `10_lookup_patient.py` resolves a name+DOB
  to a `cluster_id` the way a real front desk would need to, and
  `09_patient_history_view.py` is the single combined entry point -- briefing on
  top, full annotated history below, patient's name shown instead of a bare
  cluster id.

## Honest results (not cherry-picked)

**Entity resolution** (script 3): 176 hospital-local records → 108 predicted
patients (exact match to ground truth), precision = recall = F1 = 1.000. Two real
bugs were found and fixed during development: a DOB-corruption step that could
generate calendar-invalid dates (e.g. day 32), and a case-sensitivity bug in fuzzy
address matching that caused a coincidental birthday match between two different
real patients to almost merge incorrectly.

**NER extraction** (script 4), against self-generated gold entities:

| Category | Precision | Recall | F1 |
|---|---|---|---|
| CLINICAL_FINDING (diagnosis+symptom) | 0.862 | 0.533 | 0.659 |
| MEDICATION | 0.736 | 0.995 | 0.847 |
| ALLERGY | 1.000 | 1.000 | 1.000 |

ALLERGY's perfect score is a rule-based regex matched against this project's own
note templates, not a demonstration of the NER model's own ability — worth being
precise about that distinction. The CLINICAL_FINDING ceiling is limited by the
model being used zero-shot (no fine-tuning) and being measurably weaker on lay
phrasing ("a chest cold") than clinical terminology ("bronchitis"), plus a blind
spot on dental terminology.

**Normalization** (script 5): of ~7,900 extracted entities, 1.7% matched the exact
synonym dictionary, 54.4% matched one specific canonical concept confidently, 29.4%
were honestly flagged as ambiguous (e.g. "fracture" alone can't tell you which bone),
and 14.5% were left unmapped (mostly sub-word NER fragments too short to trust).

**Timeline conflicts** (script 6): of 239 facts for patients seen at 2+ hospitals,
only 14 (5.9%) were independently corroborated by a second hospital. This mostly
reflects how Synthea logs a condition once at whichever visit it was diagnosed
rather than restating it at every subsequent encounter — so "unconfirmed" here
mostly means "only one hospital ever had a reason to document this," not "hospitals
actively disagree." Worth being precise about that distinction in any write-up.

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

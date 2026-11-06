"""
Biomedical NER over the synthesized clinical notes.

Model: d4data/biomedical-ner-all -- a BERT-base token-classification model fine-tuned
on MACCROBAT clinical text. It recognizes ~40 clinical entity types (Disease_disorder,
Medication, Sign_symptom, Date, Dosage, ...), which we collapse down to the categories
Part A actually needs: DIAGNOSIS, MEDICATION, SYMPTOM, ALLERGY, DATE. This is a
standard pretrained-biomedical-NER approach (scispaCy was tried first but its pinned
spaCy/thinc/blis stack has no prebuilt wheels for this Python version and fails to
build from source; a transformers-based BioBERT model sidesteps that entirely).

The model has no dedicated "allergy" label, so ALLERGY is produced by a small rule
layer on top of it: regex patterns matched against the note's own allergy sentence
shapes, pulling out the allergen phrase directly rather than trying to reclassify
whatever the base NER model happened to tag nearby (an earlier version did that by
just flagging every entity in a sentence containing "allerg" -- it also caught the
literal word "allergy" itself and stray sub-token fragments, which tanked precision).
This mirrors how real clinical NLP systems layer context/assertion/relation detection
on top of a base NER model rather than expecting the base model to encode every
attribute itself; the specific patterns here are shaped around this project's own
note templates, which is a reasonable trade at this scale but wouldn't generalize to
arbitrary clinical text without a real relation-extraction step.

DIAGNOSIS and SYMPTOM are graded together as one CLINICAL_FINDING category. Synthea's
conditions.csv doesn't distinguish "this is a diagnosis" from "this is a symptom" --
they're all just "conditions" -- so our gold set has no real symptom/diagnosis
ground-truth split either. Scoring them separately would penalize the model every time
it correctly recognizes something (e.g. "fever", "cough") as Sign_symptom rather than
Disease_disorder, which is the model being *more* clinically correct than our gold
set, not less.

Because notes were synthesized from structured fields by 02_generate_notes.py, which
also wrote the literal gold phrases embedded in each note
(notes_gold_entities.csv), this script can score extraction precision/recall for real
instead of only producing entities with no way to check they're right.

Input:  data/hospitals/<slug>/{notes,notes_gold_entities}.csv
Output: data/hospitals/<slug>/extracted_entities.csv
        printed precision/recall/F1 per category
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz
from transformers import pipeline

DATA = Path(__file__).resolve().parent.parent / "data"
HOSPITALS_DIR = DATA / "hospitals"

MODEL_NAME = "d4data/biomedical-ner-all"

LABEL_TO_CATEGORY = {
    "Disease_disorder": "DIAGNOSIS",
    "History": "DIAGNOSIS",  # this checkpoint uses History for "diagnosed with X"-style mentions
    "Sign_symptom": "SYMPTOM",
    "Medication": "MEDICATION",
    "Date": "DATE",
    "Time": "DATE",
    "Duration": "DATE",
    "Frequency": "DATE",
}
FUZZY_MATCH_THRESHOLD = 70  # partial_ratio 0-100; gold phrases are near-verbatim in the note
MERGE_GAP_CHARS = 1  # merge same-category spans separated by nothing or a single space/subword join

# A few generic, non-specific words this checkpoint sometimes tags as Medication even
# though they don't name a drug (e.g. "No new medications prescribed" -> "medications"
# gets tagged). Filtered the same way a real pipeline would filter any NER output
# before treating it as a specific entity mention.
GENERIC_MEDICATION_TERMS = {"medication", "medications", "medicine", "medicines", "drug", "drugs"}
GENERIC_CLINICAL_TERMS = {
    "symptom", "symptoms", "complaint", "complaints", "acute complaints",
    "finding", "findings", "issue", "issues", "concern", "concerns",
    "condition", "conditions", "problem", "problems",
}

# Simple negation check: the model has no assertion/negation awareness of its own, so
# it tags "acute complaints" as a positive Sign_symptom inside "no acute complaints
# voiced at check-in" just as readily as a real complaint. A window of a few words
# immediately before the entity span is checked for a negation cue -- a lightweight
# version of what NegEx does in real clinical NLP pipelines.
NEGATION_CUES = re.compile(r"\b(no|not|without|denies|deny|denied)\b", re.IGNORECASE)
NEGATION_WINDOW_CHARS = 25


def is_negated(note_text: str, start: int) -> bool:
    window = note_text[max(0, start - NEGATION_WINDOW_CHARS):start]
    return bool(NEGATION_CUES.search(window))

# Regex patterns shaped around this project's own ALLERGY_TEMPLATES sentence shapes
# (see 02_generate_notes.py) -- pulls the allergen phrase out directly instead of
# reclassifying nearby NER output.
ALLERGY_PATTERNS = [
    re.compile(r"allergy to (.+?)(?: prior to any new prescriptions)?\.", re.IGNORECASE),
    re.compile(r"allergy history notable for (.+?)\.", re.IGNORECASE),
    re.compile(r"existing (.+?) allergy\.", re.IGNORECASE),
]


def extract_allergy_mentions(note_text: str) -> list:
    mentions = []
    for pattern in ALLERGY_PATTERNS:
        for m in pattern.finditer(note_text):
            mentions.append({"text": m.group(1).strip(), "start": m.start(1), "score": 1.0})
    return mentions


def merge_adjacent_entities(entities: list, note_text: str) -> list:
    """aggregation_strategy="simple" doesn't reliably merge subword continuations for
    this checkpoint (e.g. "an" + "##emia" from "anemia" comes back as two entities
    with the same label). Re-merge ourselves using character offsets against the
    original text, which also sidesteps "##" artifacts in the model's own `word`
    field -- the merged text is a real substring of the note."""
    mapped = []
    for ent in entities:
        category = LABEL_TO_CATEGORY.get(ent["entity_group"], "OTHER")
        mapped.append({"start": ent["start"], "end": ent["end"], "category": category, "score": float(ent["score"])})
    mapped.sort(key=lambda e: e["start"])

    merged = []
    for ent in mapped:
        if merged:
            gap = ent["start"] - merged[-1]["end"]
            # gap == 0 means these are literally subword pieces of one original word
            # (e.g. "hyper" + "glycemia" from "hyperglycemia") -- the model sometimes
            # labels those pieces with different entity types, but they're one word
            # and must stay one entity regardless of category disagreement. A gap of
            # exactly one space only merges when both sides agree on category.
            same_word = gap == 0
            same_category = merged[-1]["category"] == ent["category"]
            if same_word or (same_category and gap <= MERGE_GAP_CHARS):
                merged[-1]["end"] = max(merged[-1]["end"], ent["end"])
                merged[-1]["scores"].append(ent["score"])
                continue
        merged.append({"start": ent["start"], "end": ent["end"], "category": ent["category"], "scores": [ent["score"]]})

    return [
        {
            "category": m["category"],
            "text": note_text[m["start"]:m["end"]],
            "start": m["start"],
            "score": sum(m["scores"]) / len(m["scores"]),
        }
        for m in merged
        if m["category"] != "OTHER"
    ]


def extract_for_hospital(ner_pipe, hdir: Path):
    notes_path = hdir / "notes.csv"
    if not notes_path.exists():
        return []
    notes = list(csv.DictReader(open(notes_path, encoding="utf-8")))

    rows = []
    texts = [n["NOTE_TEXT"] for n in notes]
    batch_results = ner_pipe(texts, batch_size=16)

    for note, entities in zip(notes, batch_results):
        for ent in merge_adjacent_entities(entities, note["NOTE_TEXT"]):
            category = ent["category"]
            text_lower = ent["text"].strip().lower()
            if category == "MEDICATION" and text_lower in GENERIC_MEDICATION_TERMS:
                continue
            if category in ("DIAGNOSIS", "SYMPTOM"):
                if text_lower in GENERIC_CLINICAL_TERMS:
                    continue
                if is_negated(note["NOTE_TEXT"], ent["start"]):
                    continue
            rows.append({
                "LOCAL_ENCOUNTER_ID": note["LOCAL_ENCOUNTER_ID"],
                "LOCAL_PATIENT_ID": note["LOCAL_PATIENT_ID"],
                "CATEGORY": category,
                "ENTITY_TEXT": ent["text"],
                "SCORE": round(ent["score"], 3),
            })
        for allergy_ent in extract_allergy_mentions(note["NOTE_TEXT"]):
            rows.append({
                "LOCAL_ENCOUNTER_ID": note["LOCAL_ENCOUNTER_ID"],
                "LOCAL_PATIENT_ID": note["LOCAL_PATIENT_ID"],
                "CATEGORY": "ALLERGY",
                "ENTITY_TEXT": allergy_ent["text"],
                "SCORE": allergy_ent["score"],
            })
    return rows


# Synthea's conditions.csv has no diagnosis/symptom split, so neither does our gold
# set -- score them as one bucket rather than penalizing the model for correctly
# telling "fever" (Sign_symptom) apart from "pneumonia" (Disease_disorder).
EVAL_CATEGORY = {"DIAGNOSIS": "CLINICAL_FINDING", "SYMPTOM": "CLINICAL_FINDING"}


def evaluate(hdir: Path, extracted_rows: list):
    gold_path = hdir / "notes_gold_entities.csv"
    if not gold_path.exists():
        return None
    gold_rows = list(csv.DictReader(open(gold_path, encoding="utf-8")))

    gold_by_enc = defaultdict(list)
    for g in gold_rows:
        category = EVAL_CATEGORY.get(g["CATEGORY"], g["CATEGORY"])
        gold_by_enc[(g["LOCAL_ENCOUNTER_ID"], category)].append(g["PHRASE"])

    extracted_by_enc = defaultdict(list)
    for e in extracted_rows:
        category = EVAL_CATEGORY.get(e["CATEGORY"], e["CATEGORY"])
        extracted_by_enc[(e["LOCAL_ENCOUNTER_ID"], category)].append(e["ENTITY_TEXT"])

    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    all_keys = set(gold_by_enc) | set(extracted_by_enc)

    for key in all_keys:
        enc_id, category = key
        gold_phrases = list(gold_by_enc.get(key, []))
        pred_phrases = list(extracted_by_enc.get(key, []))
        matched_gold = set()
        for pred in pred_phrases:
            hit = None
            for i, gold in enumerate(gold_phrases):
                if i in matched_gold:
                    continue
                if fuzz.partial_ratio(pred.lower(), gold.lower()) >= FUZZY_MATCH_THRESHOLD:
                    hit = i
                    break
            if hit is not None:
                matched_gold.add(hit)
                stats[category]["tp"] += 1
            else:
                stats[category]["fp"] += 1
        stats[category]["fn"] += len(gold_phrases) - len(matched_gold)

    return stats


def main():
    print(f"Loading {MODEL_NAME} ...")
    ner_pipe = pipeline(
        "ner",
        model=MODEL_NAME,
        aggregation_strategy="simple",
    )

    hospital_dirs = sorted(d for d in HOSPITALS_DIR.iterdir() if d.is_dir())
    overall_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total_entities = 0

    for hdir in hospital_dirs:
        rows = extract_for_hospital(ner_pipe, hdir)
        if not rows:
            continue
        total_entities += len(rows)

        with open(hdir / "extracted_entities.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        stats = evaluate(hdir, rows)
        if stats:
            for category, s in stats.items():
                overall_stats[category]["tp"] += s["tp"]
                overall_stats[category]["fp"] += s["fp"]
                overall_stats[category]["fn"] += s["fn"]

        print(f"  {hdir.name}: {len(rows)} entities extracted")

    print(f"\nTotal entities extracted: {total_entities} across {len(hospital_dirs)} hospitals")

    print("\n--- Extraction precision/recall vs. self-generated gold entities ---")
    print(f"{'CATEGORY':<12} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    for category in ["CLINICAL_FINDING", "MEDICATION", "ALLERGY"]:
        s = overall_stats.get(category, {"tp": 0, "fp": 0, "fn": 0})
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"{category:<12} {tp:>5} {fp:>5} {fn:>5} {precision:>10.3f} {recall:>10.3f} {f1:>8.3f}")


if __name__ == "__main__":
    main()

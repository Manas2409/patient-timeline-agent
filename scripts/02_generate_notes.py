"""
Synthesize free-text clinical notes per encounter from Synthea's structured fields.

Synthea's downloadable sample data ships structured tables only (conditions,
medications, allergies as discrete rows) -- no unstructured note text. Part A's NER
step needs something to extract *from*, so this script turns each encounter's
structured facts into a short narrative note, the way a real progress note would
read. Two things are deliberately injected for downstream stages to deal with:

  1. Lay/clinical synonym variation (e.g. "heart attack" vs "myocardial infarction")
     so the normalization step in Part A has real synonym-mapping work to do.
  2. Sentence-level template variety + reproducible per-encounter randomness, so notes
     aren't uniform boilerplate a regex could trivially parse.

Input:  data/hospitals/<slug>/{encounters,conditions,medications,allergies}.csv
Output: data/hospitals/<slug>/notes.csv  (LOCAL_ENCOUNTER_ID, LOCAL_PATIENT_ID, DATE, NOTE_TEXT)
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

from vocab import SYNONYMS, clean, is_admin_sdoh

HOSPITALS_DIR = Path(__file__).resolve().parent.parent / "data" / "hospitals"

SOCIAL_HISTORY_TEMPLATES = [
    "Social history noted: {x}.",
    "Care-coordination note: {x}.",
]

REASON_TEMPLATES = [
    "Patient presents today with {x}.",
    "Seen in clinic for {x}.",
    "Patient reports symptoms consistent with {x}.",
    "Follow-up visit regarding {x}.",
    "Presenting complaint: {x}.",
    "Patient was evaluated for {x}.",
]
GENERIC_VISIT_TEMPLATES = [
    "Routine {cls} visit, no acute complaints voiced at check-in.",
    "Patient seen for a scheduled {cls} encounter.",
    "{cls_cap} visit for general evaluation.",
]
CONDITION_TEMPLATES = [
    "Assessment: {x}.",
    "New findings on exam include {x}.",
    "Diagnosed with {x} at this visit.",
    "Patient continues to be followed for {x}.",
    "Clinical impression: {x}.",
]
MEDICATION_TEMPLATES = [
    "Started on {x}.",
    "Medications prescribed today: {x}.",
    "Plan: initiate {x}.",
    "Patient given {x} during this encounter.",
]
NO_MED_TEMPLATES = [
    "No new medications prescribed at this time.",
    "Medication list unchanged.",
]
ALLERGY_TEMPLATES = [
    "Patient reports a known allergy to {x}.",
    "Allergy history notable for {x}.",
    "Nursing staff flagged an allergy to {x} prior to any new prescriptions.",
    "Chart notes an existing {x} allergy.",
]

def humanize_condition(desc: str, rng: random.Random) -> str:
    clean_desc = clean(desc)
    options = SYNONYMS.get(clean_desc)
    if options:
        return rng.choice(options)
    return clean_desc[0].lower() + clean_desc[1:] if clean_desc else clean_desc


def clean_medication(desc: str) -> str:
    # Trim dose-form boilerplate a bit; keep the drug name + strength, which is what
    # a NER model should learn to pull out anyway.
    return desc.strip()


def join_list(items: list) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def load_csv(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_note(enc, conditions, medications, allergy, rng: random.Random):
    """Returns (note_text, gold_entities) where gold_entities is a list of
    (category, phrase) exactly as the phrase was embedded in the text -- i.e. the
    real answer key for the NER step, not the canonical Synthea description."""
    sentences = []
    gold = []

    clinical_conditions = [c for c in conditions if not is_admin_sdoh(c["DESCRIPTION"])]
    admin_conditions = [c for c in conditions if is_admin_sdoh(c["DESCRIPTION"])]

    reason_raw = enc.get("REASONDESCRIPTION", "") or ""
    reason = clean(reason_raw)
    if reason and not is_admin_sdoh(reason_raw):
        reason_phrase = humanize_condition(reason, rng)
        sentences.append(rng.choice(REASON_TEMPLATES).format(x=reason_phrase))
        gold.append(("DIAGNOSIS", reason_phrase))
    elif not clinical_conditions:
        cls = enc.get("ENCOUNTERCLASS", "outpatient") or "outpatient"
        sentences.append(rng.choice(GENERIC_VISIT_TEMPLATES).format(cls=cls, cls_cap=cls.capitalize()))

    if clinical_conditions:
        phrases = [humanize_condition(c["DESCRIPTION"], rng) for c in clinical_conditions]
        sentences.append(rng.choice(CONDITION_TEMPLATES).format(x=join_list(phrases)))
        gold.extend(("DIAGNOSIS", p) for p in phrases)

    if admin_conditions:
        # Mentioned in the note for realism (SDOH context matters clinically) but not
        # graded -- a biomedical NER model correctly won't tag these as diagnoses.
        admin_phrases = [clean(c["DESCRIPTION"]) for c in admin_conditions]
        sentences.append(rng.choice(SOCIAL_HISTORY_TEMPLATES).format(x=join_list(admin_phrases)))

    if medications:
        med_names = [clean_medication(m["DESCRIPTION"]) for m in medications]
        sentences.append(rng.choice(MEDICATION_TEMPLATES).format(x=join_list(med_names)))
        gold.extend(("MEDICATION", m) for m in med_names)
    elif rng.random() < 0.3:
        sentences.append(rng.choice(NO_MED_TEMPLATES))

    if allergy:
        allergy_phrase = clean(allergy["DESCRIPTION"])
        sentences.append(rng.choice(ALLERGY_TEMPLATES).format(x=allergy_phrase))
        gold.append(("ALLERGY", allergy_phrase))

    return " ".join(sentences), gold


def main():
    hospital_dirs = [d for d in HOSPITALS_DIR.iterdir() if d.is_dir()]
    total_notes = 0

    for hdir in hospital_dirs:
        encounters = load_csv(hdir / "encounters.csv")
        conditions = load_csv(hdir / "conditions.csv")
        medications = load_csv(hdir / "medications.csv")
        allergies = load_csv(hdir / "allergies.csv")

        conditions_by_enc = defaultdict(list)
        for c in conditions:
            conditions_by_enc[c["LOCAL_ENCOUNTER_ID"]].append(c)
        medications_by_enc = defaultdict(list)
        for m in medications:
            medications_by_enc[m["LOCAL_ENCOUNTER_ID"]].append(m)
        allergy_by_patient_date = {}
        for a in allergies:
            allergy_by_patient_date[(a["LOCAL_PATIENT_ID"], a["START"])] = a

        rows = []
        gold_rows = []
        for enc in encounters:
            eid = enc["LOCAL_ENCOUNTER_ID"]
            rng = random.Random(hash(eid) & 0xFFFFFFFF)
            date = enc["START"][:10]
            allergy = allergy_by_patient_date.get((enc["LOCAL_PATIENT_ID"], date))
            note_text, gold = build_note(
                enc,
                conditions_by_enc.get(eid, []),
                medications_by_enc.get(eid, []),
                allergy,
                rng,
            )
            rows.append({
                "LOCAL_ENCOUNTER_ID": eid,
                "LOCAL_PATIENT_ID": enc["LOCAL_PATIENT_ID"],
                "DATE": date,
                "NOTE_TEXT": note_text,
            })
            for category, phrase in gold:
                gold_rows.append({
                    "LOCAL_ENCOUNTER_ID": eid,
                    "LOCAL_PATIENT_ID": enc["LOCAL_PATIENT_ID"],
                    "CATEGORY": category,
                    "PHRASE": phrase,
                })

        if rows:
            out_path = hdir / "notes.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            total_notes += len(rows)

        if gold_rows:
            gold_path = hdir / "notes_gold_entities.csv"
            with open(gold_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(gold_rows[0].keys()))
                writer.writeheader()
                writer.writerows(gold_rows)

    print(f"Wrote {total_notes} synthesized notes across {len(hospital_dirs)} hospitals")


if __name__ == "__main__":
    main()

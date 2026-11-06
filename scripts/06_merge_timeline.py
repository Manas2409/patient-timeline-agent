"""
Merge everything built so far into one per-patient timeline: entity resolution says
which hospital-local records belong to the same real person, extraction says what
happened at each visit, normalization says what it actually means. This script pulls
those three together, sorts each patient's visits by date, and flags disagreements
between sources instead of silently picking one hospital as ground truth.

Deliberately uses the *predicted* entity-resolution clusters (data/entity_resolution/
predicted_clusters.csv), not the ground-truth crosswalk -- a real deployed system
would only ever have its own predictions to work with. We already know from 03's eval
that those predictions are 100% correct on this dataset, so this is a fair thing to
build on top of, but it's the honest dependency to take.

Conflict/corroboration rule: for a given patient, a clinical fact (a normalized
concept under DIAGNOSIS/SYMPTOM/ALLERGY) is "confirmed" if 2+ of that patient's
hospitals independently reported it, and "single_source" if only 1 of 2+ hospitals
did. This generalizes a common real-world scenario ("penicillin allergy noted at
Hospital A only -- unconfirmed at other visits") -- our synthetic data doesn't
generate explicit contradictions (e.g. two hospitals disagreeing on blood type), only
omissions, so corroboration count is the honest signal available here. A patient only
ever seen at one hospital has nothing to corroborate against, so no fact for them gets
flagged either way.

Also enriches DIAGNOSIS/MEDICATION entities with an "outcome" (resolved-by-date vs
ongoing, from each record's STOP date) and ALLERGY entities with "reaction"/"severity"
(from Synthea's own structured allergy fields) -- all data that already existed
upstream but wasn't being carried through until now, matched back to each extracted
entity via fuzzy text matching since NER output doesn't share a row-level key with the
original structured record.

Input:  data/entity_resolution/predicted_clusters.csv
        data/hospitals/<slug>/{encounters,conditions,medications,allergies,normalized_entities}.csv
Output: data/unified_timelines/patient_<cluster_id>.json (one per patient)
        data/unified_timelines/index.csv (cluster_id -> hospital count, conflict count)
        printed summary + one example patient
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz

DATA = Path(__file__).resolve().parent.parent / "data"
HOSPITALS_DIR = DATA / "hospitals"
ENTITY_RES_DIR = DATA / "entity_resolution"
OUT_DIR = DATA / "unified_timelines"

# Matches the ZIP_PREFIX_TO_REGION values in 01_shatter_and_corrupt.py -- kept as a
# literal list here rather than imported, since 01's module does real work (reads
# data/raw) on any import and this script only needs the slug -> display name map.
HOSPITAL_DISPLAY_NAMES = {
    "pioneer_valley_health_network": "Pioneer Valley Health Network",
    "berkshires_regional_health": "Berkshires Regional Health",
    "north_central_mass_health": "North Central Mass Health",
    "worcester_regional_medical_center": "Worcester Regional Medical Center",
    "metrowest_health_partners": "MetroWest Health Partners",
    "merrimack_valley_health_network": "Merrimack Valley Health Network",
    "north_shore_health_alliance": "North Shore Health Alliance",
    "norfolk_county_health_network": "Norfolk County Health Network",
    "greater_boston_medical_center": "Greater Boston Medical Center",
    "west_suburban_boston_health": "West Suburban Boston Health",
    "south_shore_health_alliance": "South Shore Health Alliance",
    "cape_cod_islands_health": "Cape Cod & Islands Health",
    "southcoast_health": "Southcoast Health",
}


def load_csv(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


OUTCOME_MATCH_THRESHOLD = 75


def find_outcome(entity_text: str, structured_rows: list) -> str:
    """Best-effort match of an extracted entity against the structured rows logged at
    the same encounter, to pull its STOP date -- i.e. an encounter-level approximation,
    not a guaranteed per-entity match, since e.g. a 9-medication encounter can't always
    be disambiguated from NER text alone. Falls back to 'not available' rather than
    guessing when no row matches well."""
    if not structured_rows:
        return "not available"
    best = max(structured_rows, key=lambda r: fuzz.partial_ratio(entity_text.lower(), r["DESCRIPTION"].lower()))
    if fuzz.partial_ratio(entity_text.lower(), best["DESCRIPTION"].lower()) < OUTCOME_MATCH_THRESHOLD:
        return "not available"
    return f"resolved {best['STOP']}" if best.get("STOP") else "ongoing"


def find_allergy_detail(canonical: str, allergy_rows: list) -> dict:
    """Same best-effort matching, for the structured severity/reaction fields Synthea
    tracks on the allergy record itself (SEVERITY1/DESCRIPTION1) rather than in note
    text -- see vocab.py's clean() for why these don't always exactly match the
    normalized canonical name."""
    if not allergy_rows:
        return {}
    best = max(allergy_rows, key=lambda r: fuzz.partial_ratio(canonical.lower(), r["DESCRIPTION"].lower()))
    if fuzz.partial_ratio(canonical.lower(), best["DESCRIPTION"].lower()) < OUTCOME_MATCH_THRESHOLD:
        return {}
    return {"reaction": best.get("REACTION", ""), "severity": best.get("SEVERITY", "")}


def main():
    clusters = load_csv(ENTITY_RES_DIR / "predicted_clusters.csv")
    records_by_cluster = defaultdict(list)
    for r in clusters:
        records_by_cluster[r["CLUSTER_ID"]].append(r)

    # Cache each hospital's encounters/normalized-entities once, keyed by local patient id.
    hospital_encounters = {}
    hospital_entities_by_enc = {}
    hospital_conditions_by_enc = {}
    hospital_medications_by_enc = {}
    hospital_allergies_by_patient = {}
    for hdir in HOSPITALS_DIR.iterdir():
        if not hdir.is_dir():
            continue
        slug = hdir.name
        hospital_encounters[slug] = load_csv(hdir / "encounters.csv")
        ents_by_enc = defaultdict(list)
        for e in load_csv(hdir / "normalized_entities.csv"):
            ents_by_enc[e["LOCAL_ENCOUNTER_ID"]].append(e)
        hospital_entities_by_enc[slug] = ents_by_enc

        conditions_by_enc = defaultdict(list)
        for c in load_csv(hdir / "conditions.csv"):
            conditions_by_enc[c["LOCAL_ENCOUNTER_ID"]].append(c)
        hospital_conditions_by_enc[slug] = conditions_by_enc

        medications_by_enc = defaultdict(list)
        for m in load_csv(hdir / "medications.csv"):
            medications_by_enc[m["LOCAL_ENCOUNTER_ID"]].append(m)
        hospital_medications_by_enc[slug] = medications_by_enc

        allergies_by_patient = defaultdict(list)
        for a in load_csv(hdir / "allergies.csv"):
            allergies_by_patient[a["LOCAL_PATIENT_ID"]].append(a)
        hospital_allergies_by_patient[slug] = allergies_by_patient

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = []
    total_conflicts = 0
    example_patient = None

    for cluster_id, members in records_by_cluster.items():
        hospitals_touched = sorted({m["HOSPITAL_SLUG"] for m in members})

        # 1. Build the raw timeline: every encounter across every hospital this
        # predicted patient touches, each carrying its source hospital and entities.
        timeline = []
        for m in members:
            slug, local_id = m["HOSPITAL_SLUG"], m["LOCAL_PATIENT_ID"]
            for enc in hospital_encounters[slug]:
                if enc["LOCAL_PATIENT_ID"] != local_id:
                    continue
                eid = enc["LOCAL_ENCOUNTER_ID"]
                entities = []
                for e in hospital_entities_by_enc[slug].get(eid, []):
                    entity = {
                        "category": e["CATEGORY"],
                        "text": e["ENTITY_TEXT"],
                        "canonical": e["CANONICAL"],
                        "match_method": e["MATCH_METHOD"],
                    }
                    if e["CATEGORY"] == "DIAGNOSIS":
                        entity["outcome"] = find_outcome(e["ENTITY_TEXT"], hospital_conditions_by_enc[slug].get(eid, []))
                    elif e["CATEGORY"] == "MEDICATION":
                        entity["outcome"] = find_outcome(e["ENTITY_TEXT"], hospital_medications_by_enc[slug].get(eid, []))
                    elif e["CATEGORY"] == "ALLERGY":
                        entity.update(find_allergy_detail(e["CANONICAL"], hospital_allergies_by_patient[slug].get(local_id, [])))
                    entities.append(entity)
                timeline.append({
                    "date": enc["START"][:10],
                    "hospital": HOSPITAL_DISPLAY_NAMES.get(slug, slug),
                    "encounter_class": enc["ENCOUNTERCLASS"],
                    "reason": enc.get("REASONDESCRIPTION", ""),
                    "seen_at": enc.get("SEEN_AT", ""),
                    "entities": entities,
                })
        timeline.sort(key=lambda v: v["date"])

        # 2. Corroboration check: for each normalized concept, which hospitals
        # independently reported it (only counts confident matches -- ambiguous/
        # unmapped entities aren't reliable enough to corroborate or contradict
        # anything on their own).
        hospitals_reporting = defaultdict(set)
        for visit in timeline:
            for ent in visit["entities"]:
                if ent["category"] in ("DIAGNOSIS", "SYMPTOM", "ALLERGY") and ent["match_method"] in ("exact_synonym", "fuzzy"):
                    hospitals_reporting[(ent["category"], ent["canonical"])].add(visit["hospital"])

        conflicts = []
        if len(hospitals_touched) >= 2:
            for (category, canonical), reporting_hospitals in hospitals_reporting.items():
                if len(reporting_hospitals) == 1:
                    conflicts.append({
                        "category": category,
                        "canonical": canonical,
                        "reported_by": sorted(reporting_hospitals),
                        "status": "unconfirmed_single_source",
                        "total_hospitals_for_patient": len(hospitals_touched),
                    })
        total_conflicts += len(conflicts)

        patient_record = {
            "cluster_id": cluster_id,
            "hospitals": [HOSPITAL_DISPLAY_NAMES.get(s, s) for s in hospitals_touched],
            "timeline": timeline,
            "conflicts": conflicts,
        }

        with open(OUT_DIR / f"patient_{cluster_id}.json", "w", encoding="utf-8") as f:
            json.dump(patient_record, f, indent=2)

        index_rows.append({
            "CLUSTER_ID": cluster_id,
            "NUM_HOSPITALS": len(hospitals_touched),
            "NUM_VISITS": len(timeline),
            "NUM_CONFLICTS": len(conflicts),
        })

        if example_patient is None and len(conflicts) >= 1 and len(hospitals_touched) >= 3:
            example_patient = patient_record

    with open(OUT_DIR / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)

    multi_hospital_patients = sum(1 for r in index_rows if int(r["NUM_HOSPITALS"]) >= 2)
    print(f"Built {len(index_rows)} unified patient timelines in {OUT_DIR}")
    print(f"  {multi_hospital_patients} patients have records at 2+ hospitals (corroboration possible)")
    print(f"  {total_conflicts} single-source (unconfirmed) facts flagged across all patients")

    if example_patient:
        print(f"\n--- Example: patient {example_patient['cluster_id']} "
              f"({len(example_patient['hospitals'])} hospitals, {len(example_patient['conflicts'])} conflicts) ---")
        print("Hospitals:", ", ".join(example_patient["hospitals"]))
        for c in example_patient["conflicts"][:5]:
            print(f"  UNCONFIRMED: {c['category']} '{c['canonical']}' -- only reported by {c['reported_by']}"
                  f" (out of {c['total_hospitals_for_patient']} hospitals this patient visited)")


if __name__ == "__main__":
    main()

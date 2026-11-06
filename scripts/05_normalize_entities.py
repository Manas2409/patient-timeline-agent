"""
Map every extracted entity's surface text to one canonical medical concept, so
"heart attack" / "myocardial infarction" / "NSTEMI" collapse into a single concept
instead of being treated as three unrelated facts. This is the project's simplified
stand-in for what UMLS / SNOMED CT do in a real interoperability system: every
synonym and abbreviation resolves to one standard concept ID. We use a small
hand-built dictionary instead of the real UMLS.

Two lookup strategies, tried in order:
  1. Exact reverse lookup against the same SYNONYMS dictionary note generation used
     (see vocab.py) to inject lay-phrasing variation in the first place -- catches
     every synonym we deliberately built in, cheaply and reliably.
  2. Fuzzy match (rapidfuzz) against a canonical vocabulary built from Synthea's own
     raw conditions/medications/allergies descriptions -- catches everything else,
     including truncated NER output ("hyperlipide" -> "Hyperlipidemia") that an exact
     lookup would miss.
Anything below the fuzzy threshold is left unmapped rather than force-matched to the
nearest vocabulary entry -- a wrong normalization is worse than an honest "couldn't
map this."

Medication text is additionally stripped of dose/form/brand-name boilerplate
("1 ML Epoetin Alfa 4000 UNT/ML Injection [Epogen]" -> "Epoetin Alfa") before
canonicalization -- otherwise every distinct dose of the same drug would count as a
different "medication". This drops secondary ingredients in combination drugs (e.g.
"Acetaminophen 325 MG / Oxycodone ... [Percocet]" -> "Acetaminophen" only), a known
simplification at this scale.

Note: STEMI / NSTEMI / "history of MI" all collapse to the single canonical concept
"Myocardial infarction" here, losing the real clinical distinction between an acute
event and a past one. A real UMLS-backed system would keep these as related-but-
distinct concepts; this project's dictionary is intentionally simplified.

Input:  data/hospitals/<slug>/extracted_entities.csv
        data/raw/{conditions,medications,allergies}.csv (canonical vocabulary source)
Output: data/hospitals/<slug>/normalized_entities.csv
        printed before/after distinct-surface-form counts + example collapses
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz, process, utils

from vocab import SYNONYMS, clean, is_admin_sdoh

DATA = Path(__file__).resolve().parent.parent / "data"
HOSPITALS_DIR = DATA / "hospitals"
RAW = DATA / "raw"

FUZZY_THRESHOLD = 85
MIN_FUZZY_LENGTH = 5  # below this, a substring-based fuzzy score is unreliable --
# e.g. "et" scores a perfect partial_ratio match against "Diabetes" purely by
# coincidence. Short fragments are left unmapped rather than force-matched.

# Every known surface form (lowercased) -> canonical concept name (the SYNONYMS key,
# in its original casing).
EXACT_LOOKUP = {}
for canonical, variants in SYNONYMS.items():
    EXACT_LOOKUP[canonical.lower()] = canonical
    for v in variants:
        EXACT_LOOKUP[v.lower()] = canonical

LEADING_DOSE_PREFIX_RE = re.compile(r"^(?:\d+(?:\.\d+)?\s+(?:ML|MG|HR|ACTUAT|UNT)\s+)+", re.IGNORECASE)
BRAND_SUFFIX_RE = re.compile(r"\s*\[[^\]]+\]\s*$")
FIRST_DIGIT_RE = re.compile(r"\d")


def core_drug_name(desc: str) -> str:
    text = BRAND_SUFFIX_RE.sub("", desc).strip()
    text = LEADING_DOSE_PREFIX_RE.sub("", text)
    m = FIRST_DIGIT_RE.search(text)
    if m:
        text = text[:m.start()]
    return re.sub(r"\s+", " ", text).strip().rstrip("/").strip()


def build_diagnosis_vocab() -> set:
    vocab = set()
    for row in csv.DictReader(open(RAW / "conditions.csv", encoding="utf-8")):
        if not is_admin_sdoh(row["DESCRIPTION"]):
            vocab.add(clean(row["DESCRIPTION"]))
    return vocab


def build_medication_vocab() -> set:
    # Combination drugs list multiple active ingredients separated by "/"
    # (e.g. "Acetaminophen 325 MG / Oxycodone Hydrochloride 10 MG Oral Tablet"). The
    # note text mentions the whole string, so the NER model can legitimately extract
    # any one of the ingredients -- the vocabulary needs all of them, not just
    # whichever core_drug_name() would keep from the unsplit string.
    vocab = set()
    for row in csv.DictReader(open(RAW / "medications.csv", encoding="utf-8")):
        desc = BRAND_SUFFIX_RE.sub("", row["DESCRIPTION"]).strip()
        for segment in desc.split("/"):
            name = core_drug_name(segment)
            if name:
                vocab.add(name)
    return vocab


def build_allergy_vocab() -> set:
    vocab = set()
    for row in csv.DictReader(open(RAW / "allergies.csv", encoding="utf-8")):
        vocab.add(clean(row["DESCRIPTION"]))
    return vocab


VOCAB_BY_CATEGORY = None  # populated in main()


def normalize(text: str, category: str):
    text_lower = text.strip().lower()

    if category == "MEDICATION":
        text_for_match = core_drug_name(text)
    else:
        text_for_match = text.strip()

    hit = EXACT_LOOKUP.get(text_lower) or EXACT_LOOKUP.get(text_for_match.lower())
    if hit:
        return hit, "exact_synonym", 1.0

    vocab = VOCAB_BY_CATEGORY.get(category)
    if vocab and len(text_for_match) >= MIN_FUZZY_LENGTH:
        # partial_ratio (best-matching substring) rather than token_sort_ratio, since
        # the common case here is a short/truncated NER span that's a genuine
        # substring of a much longer canonical vocabulary entry (e.g. "kidney
        # disease" inside "Chronic kidney disease stage 1") -- token_sort_ratio
        # penalizes that length mismatch even though the match is correct.
        #
        # But a generic fragment like "fracture" is an equally perfect substring of
        # "Fracture of ankle", "Fracture of wrist", "Fracture of rib", etc. -- there's
        # no way to tell which one is meant from the fragment alone. Picking one of
        # several tied top-scorers would look confident while actually being a coin
        # flip, so a tie among top scorers is reported as ambiguous instead of
        # resolved arbitrarily.
        # processor=utils.default_process lowercases (and strips punctuation) before
        # scoring -- without it, "fracture" scores a perfect 100 against
        # "Osteoporotic fracture of bone" (lowercase "fracture" already inside) but
        # slightly under 100 against "Fracture of ankle" (capital F mismatch), which
        # would pick a winner based on Synthea's capitalization quirks rather than
        # genuine specificity.
        results = process.extract(
            text_for_match, vocab, scorer=fuzz.partial_ratio, processor=utils.default_process, limit=5
        )
        if results and results[0][1] >= FUZZY_THRESHOLD:
            top_score = results[0][1]
            tied = sorted({r[0] for r in results if r[1] >= top_score})
            if len(tied) == 1:
                return tied[0], "fuzzy", round(top_score / 100, 3)
            return " | ".join(tied[:4]), "ambiguous", round(top_score / 100, 3)

    return text_for_match or text, "unmapped", 0.0


def main():
    global VOCAB_BY_CATEGORY
    diagnosis_vocab = build_diagnosis_vocab()
    VOCAB_BY_CATEGORY = {
        "DIAGNOSIS": diagnosis_vocab,
        "SYMPTOM": diagnosis_vocab,  # same underlying concept space, see 04's rationale
        "MEDICATION": build_medication_vocab(),
        "ALLERGY": build_allergy_vocab(),
    }

    all_surface_forms = defaultdict(set)
    all_canonical_forms = defaultdict(set)
    method_counts = defaultdict(int)
    total = 0

    for hdir in sorted(HOSPITALS_DIR.iterdir()):
        ext_path = hdir / "extracted_entities.csv"
        if not ext_path.exists():
            continue
        rows = list(csv.DictReader(open(ext_path, encoding="utf-8")))

        out_rows = []
        for r in rows:
            category = r["CATEGORY"]
            if category == "DATE":
                continue
            canonical, method, score = normalize(r["ENTITY_TEXT"], category)
            out_rows.append({
                "LOCAL_ENCOUNTER_ID": r["LOCAL_ENCOUNTER_ID"],
                "LOCAL_PATIENT_ID": r["LOCAL_PATIENT_ID"],
                "CATEGORY": category,
                "ENTITY_TEXT": r["ENTITY_TEXT"],
                "CANONICAL": canonical,
                "MATCH_METHOD": method,
                "MATCH_SCORE": score,
            })
            all_surface_forms[category].add(r["ENTITY_TEXT"].strip().lower())
            if method != "ambiguous":
                all_canonical_forms[category].add(canonical.strip().lower())
            method_counts[method] += 1
            total += 1

        if out_rows:
            with open(hdir / "normalized_entities.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
                writer.writeheader()
                writer.writerows(out_rows)

    print(f"Normalized {total} entities across {len(list(HOSPITALS_DIR.iterdir()))} hospitals\n")

    print("Match method breakdown:")
    for method in ["exact_synonym", "fuzzy", "ambiguous", "unmapped"]:
        n = method_counts.get(method, 0)
        print(f"  {method:<15} {n:>5}  ({n / total * 100:.1f}%)")

    print("\nDistinct surface forms -> distinct canonical concepts (dedup effect):")
    for category in ["DIAGNOSIS", "SYMPTOM", "MEDICATION", "ALLERGY"]:
        before = len(all_surface_forms.get(category, set()))
        after = len(all_canonical_forms.get(category, set()))
        print(f"  {category:<12} {before:>4} surface forms  ->  {after:>4} canonical concepts")


if __name__ == "__main__":
    main()

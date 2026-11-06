"""
Entity resolution across hospital silos: pool every hospital's local patient records
and figure out, from corrupted/inconsistent identity fields alone, which ones belong
to the same real person -- with no shared ID to rely on.

Approach: fuzzy-score every candidate pair on name / DOB / address / SSN (rapidfuzz +
a format-agnostic DOB parser), block on (gender, birth year) to cut down comparisons,
then union-find the pairs that score above threshold into predicted patient clusters.

Because data/ground_truth/patient_crosswalk.csv tells us which local records *should*
link, we can score this properly instead of eyeballing it: pairwise precision/recall/F1
against the true REAL_PATIENT_ID groupings.

Input:  data/hospitals/<slug>/patients.csv (all hospitals)
        data/ground_truth/patient_crosswalk.csv (for evaluation only)
Output: data/entity_resolution/predicted_clusters.csv
        data/entity_resolution/pair_scores.csv (every scored pair, for inspection)
        printed precision/recall/F1 + example errors
"""

import csv
import itertools
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rapidfuzz import fuzz

DATA = Path(__file__).resolve().parent.parent / "data"
HOSPITALS_DIR = DATA / "hospitals"
OUT_DIR = DATA / "entity_resolution"

MATCH_THRESHOLD = 0.72

DOB_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%b %d, %Y"]


def parse_dob(dob_str: str):
    for fmt in DOB_FORMATS:
        try:
            return datetime.strptime(dob_str, fmt).date()
        except ValueError:
            continue
    return None


def load_all_records():
    """Returns list of dicts, one per hospital-local patient record."""
    records = []
    for hdir in sorted(HOSPITALS_DIR.iterdir()):
        if not hdir.is_dir():
            continue
        patients_csv = hdir / "patients.csv"
        if not patients_csv.exists():
            continue
        with open(patients_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["HOSPITAL_SLUG"] = hdir.name
                row["_dob_parsed"] = parse_dob(row["DOB"])
                records.append(row)
    return records


def name_score(a: dict, b: dict) -> float:
    first_a, first_b = a["FIRST"].strip(), b["FIRST"].strip()
    last_a, last_b = a["LAST"].strip(), b["LAST"].strip()

    # An initial ("C.") should count as a strong partial match against a full first
    # name ("Cole") rather than scoring near-zero on string similarity.
    def first_match(x, y):
        x_bare, y_bare = x.rstrip("."), y.rstrip(".")
        if len(x_bare) == 1 or len(y_bare) == 1:
            return 100.0 if x_bare[:1].lower() == y_bare[:1].lower() else 0.0
        return fuzz.ratio(x_bare.lower(), y_bare.lower())

    f_score = first_match(first_a, first_b) if first_a and first_b else 50.0
    l_score = fuzz.ratio(last_a.lower(), last_b.lower()) if last_a and last_b else 50.0
    # Last name carries more weight -- it's less often abbreviated, and a mismatch
    # here (e.g. a maiden-name swap) is the harder case matching needs to survive on
    # DOB/address instead.
    return (0.4 * f_score + 0.6 * l_score) / 100.0


def dob_score(a: dict, b: dict) -> float:
    da, db = a["_dob_parsed"], b["_dob_parsed"]
    if da is None or db is None:
        return 0.5  # unparseable -- don't let it veto or confirm a match alone
    if da == db:
        return 1.0
    # Same year+month, transposed day digits (typo injected at generation time)
    if da.year == db.year and da.month == db.month:
        return 0.6
    return 0.0


def address_score(a: dict, b: dict) -> float:
    # Score the street separately from city/state. Scoring the concatenated string
    # (street + city + state) together lets a shared "boston ma" suffix inflate the
    # similarity of two genuinely different streets -- most of this dataset lives in
    # a handful of MA cities, so city agreement alone is weak, not strong, evidence.
    same_city_state = a["CITY"] == b["CITY"] and a["STATE"] == b["STATE"]
    if a["ADDRESS"] and b["ADDRESS"]:
        street_sim = fuzz.ratio(a["ADDRESS"].lower(), b["ADDRESS"].lower()) / 100.0
        return 0.85 * street_sim + 0.15 * (1.0 if same_city_state else 0.0)
    return 0.55 if same_city_state else 0.25


def ssn_score(a: dict, b: dict):
    """Returns (score, is_decisive). SSN is definitive when both sides have it."""
    if a["SSN"] and b["SSN"]:
        return (1.0, True) if a["SSN"] == b["SSN"] else (0.0, True)
    return (0.5, False)


def pair_score(a: dict, b: dict) -> float:
    if a["GENDER"] != b["GENDER"]:
        return 0.0
    ssn_s, decisive = ssn_score(a, b)
    if decisive:
        # Exact SSN match/mismatch overrides everything else -- it's the one field
        # that, when present on both sides, was never corrupted at generation time.
        return ssn_s
    n_s = name_score(a, b)
    d_s = dob_score(a, b)
    addr_s = address_score(a, b)
    return 0.40 * n_s + 0.35 * d_s + 0.25 * addr_s


def block_key(rec: dict):
    dob = rec["_dob_parsed"]
    return (rec["GENDER"], dob.year if dob else "unknown")


class UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def main():
    records = load_all_records()
    for r in records:
        r["_uid"] = f"{r['HOSPITAL_SLUG']}::{r['LOCAL_PATIENT_ID']}"

    blocks = defaultdict(list)
    for r in records:
        blocks[block_key(r)].append(r)
    # Safety net: a record whose DOB failed to parse has no reliable year to block on.
    # Rather than isolate it in an "unknown" bucket where it can never be compared
    # against anyone, fan it out into every year-block for its gender so name/address/
    # SSN evidence still gets a chance to match it.
    unparseable = [r for r in records if r["_dob_parsed"] is None]
    for r in unparseable:
        for key, block_records in blocks.items():
            if key[0] == r["GENDER"] and key[1] != "unknown" and r not in block_records:
                block_records.append(r)

    uf = UnionFind([r["_uid"] for r in records])
    pair_rows = []
    n_compared = 0

    for block_records in blocks.values():
        for a, b in itertools.combinations(block_records, 2):
            n_compared += 1
            score = pair_score(a, b)
            pair_rows.append({
                "A": a["_uid"], "B": b["_uid"], "SCORE": round(score, 3),
                "PREDICTED_MATCH": score >= MATCH_THRESHOLD,
            })
            if score >= MATCH_THRESHOLD:
                uf.union(a["_uid"], b["_uid"])

    # Build predicted clusters
    clusters = defaultdict(list)
    for r in records:
        clusters[uf.find(r["_uid"])].append(r)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "predicted_clusters.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["CLUSTER_ID", "HOSPITAL_SLUG", "LOCAL_PATIENT_ID", "FIRST", "LAST", "DOB"])
        for cid, members in enumerate(clusters.values()):
            for m in members:
                writer.writerow([cid, m["HOSPITAL_SLUG"], m["LOCAL_PATIENT_ID"], m["FIRST"], m["LAST"], m["DOB"]])

    with open(OUT_DIR / "pair_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["A", "B", "SCORE", "PREDICTED_MATCH"])
        writer.writeheader()
        writer.writerows(pair_rows)

    print(f"Loaded {len(records)} local patient records across {len(list(HOSPITALS_DIR.iterdir()))} hospitals")
    print(f"Blocking cut pairwise comparisons to {n_compared} (vs {len(records) * (len(records) - 1) // 2} unblocked)")
    print(f"Predicted {len(clusters)} distinct patients (true count: 108)")

    evaluate(records)


def evaluate(records):
    crosswalk = list(csv.DictReader(open(DATA / "ground_truth" / "patient_crosswalk.csv", encoding="utf-8")))
    true_patient_of = {
        f"{row['HOSPITAL_SLUG']}::{row['LOCAL_PATIENT_ID']}": row["REAL_PATIENT_ID"]
        for row in crosswalk
    }

    uf = UnionFind([r["_uid"] for r in records])
    # Rebuild predicted pairs from the pair_scores.csv we just wrote (keeps evaluate()
    # self-contained / re-runnable independent of main()'s in-memory state).
    pair_rows = list(csv.DictReader(open(OUT_DIR / "pair_scores.csv", encoding="utf-8")))
    for row in pair_rows:
        if row["PREDICTED_MATCH"] == "True":
            uf.union(row["A"], row["B"])

    uids = [r["_uid"] for r in records]
    tp = fp = fn = 0
    errors_fp, errors_fn = [], []

    for a, b in itertools.combinations(uids, 2):
        same_true = true_patient_of.get(a) == true_patient_of.get(b)
        same_pred = uf.find(a) == uf.find(b)
        if same_pred and same_true:
            tp += 1
        elif same_pred and not same_true:
            fp += 1
            errors_fp.append((a, b))
        elif same_true and not same_pred:
            fn += 1
            errors_fn.append((a, b))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("\n--- Pairwise entity-resolution evaluation vs. ground truth ---")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    if errors_fp:
        print(f"\nExample false positives (predicted same patient, actually different) [{len(errors_fp)} total]:")
        for a, b in errors_fp[:3]:
            print(f"  {a}  <->  {b}")
    if errors_fn:
        print(f"\nExample false negatives (should have matched, didn't) [{len(errors_fn)} total]:")
        for a, b in errors_fn[:3]:
            print(f"  {a}  <->  {b}")


if __name__ == "__main__":
    main()

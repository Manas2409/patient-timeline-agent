"""
The missing front door: every other script in this pipeline assumes you already know
a patient's internal cluster_id, but a real intake process starts with a patient
giving their name and DOB at the front desk -- not a database key. This takes that
query identity and fuzzy-matches it against every hospital-local patient record,
using the exact same scoring functions as 03_entity_resolution.py (reused via dynamic
import, not reimplemented), to find which existing cluster_id (if any) it belongs to.

This is really the same problem script 03 already solves (reconciling inconsistent
identity fields with no shared ID), just applied to one new incoming query instead of
reconciling the whole dataset against itself.

Usage:
  python scripts/10_lookup_patient.py --first "Jane" --last "Doe" --dob "1985-03-02" \\
      --gender F [--middle "A"] [--address "123 Main St"] [--city Boston] [--state MA] [--ssn "999-00-1234"]

DOB accepts any of: YYYY-MM-DD, MM/DD/YYYY, MM-DD-YYYY, "Mon D, YYYY" (same formats
script 01 generates across hospitals).
"""

import argparse
import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_entity_resolution_module():
    spec = importlib.util.spec_from_file_location("entity_resolution", Path(__file__).parent / "03_entity_resolution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lookup_patient(first: str, last: str, dob: str, gender: str, middle: str = "",
                    address: str = "", city: str = "", state: str = "", ssn: str = "") -> dict:
    """The reusable core, with no argparse/printing -- used by the CLI below and by
    the Streamlit app, so the two front ends can't ever disagree on matching logic.

    Returns one of:
      {"status": "invalid_dob", "accepted_formats": [...]}
      {"status": "not_found", "closest": [(record, score), ...]}
      {"status": "found", "cluster_id": str, "matches": [(record, score), ...]}
      {"status": "ambiguous", "clusters": {cluster_id: [(record, score), ...]}}
    """
    er = load_entity_resolution_module()
    query = {
        "FIRST": first, "MIDDLE": middle, "LAST": last, "DOB": dob, "GENDER": gender,
        "ADDRESS": address, "CITY": city, "STATE": state, "SSN": ssn,
    }
    query["_dob_parsed"] = er.parse_dob(dob)
    if query["_dob_parsed"] is None:
        return {"status": "invalid_dob", "accepted_formats": er.DOB_FORMATS}

    records = er.load_all_records()
    for r in records:
        r["_uid"] = f"{r['HOSPITAL_SLUG']}::{r['LOCAL_PATIENT_ID']}"

    scored = [(r, er.pair_score(query, r)) for r in records]
    scored.sort(key=lambda rs: rs[1], reverse=True)
    matches = [(r, s) for r, s in scored if s >= er.MATCH_THRESHOLD]

    if not matches:
        return {"status": "not_found", "closest": scored[:3]}

    uid_to_cluster = {}
    for row in csv.DictReader(open(DATA / "entity_resolution" / "predicted_clusters.csv", encoding="utf-8")):
        uid_to_cluster[f"{row['HOSPITAL_SLUG']}::{row['LOCAL_PATIENT_ID']}"] = row["CLUSTER_ID"]

    clusters_matched = defaultdict(list)
    for r, s in matches:
        clusters_matched[uid_to_cluster.get(r["_uid"], "UNKNOWN")].append((r, s))

    if len(clusters_matched) == 1:
        return {"status": "found", "cluster_id": next(iter(clusters_matched)), "matches": matches}
    return {"status": "ambiguous", "clusters": dict(clusters_matched)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--last", required=True)
    parser.add_argument("--dob", required=True, help="YYYY-MM-DD, MM/DD/YYYY, MM-DD-YYYY, or 'Mon D, YYYY'")
    parser.add_argument("--gender", required=True, choices=["M", "F"])
    parser.add_argument("--middle", default="")
    parser.add_argument("--address", default="")
    parser.add_argument("--city", default="")
    parser.add_argument("--state", default="")
    parser.add_argument("--ssn", default="")
    args = parser.parse_args()

    result = lookup_patient(args.first, args.last, args.dob, args.gender, args.middle,
                             args.address, args.city, args.state, args.ssn)

    if result["status"] == "invalid_dob":
        print(f"Could not parse DOB {args.dob!r}. Accepted formats: {result['accepted_formats']}")
    elif result["status"] == "not_found":
        print(f"No existing patient found matching {args.first} {args.last} (DOB {args.dob}).")
        print("This would be registered as a new patient.")
        print("\nClosest non-matching records on file, for a human to double-check:")
        for r, s in result["closest"]:
            print(f"  score={s:.3f}  {r['FIRST']} {r['LAST']}  DOB={r['DOB']}  ({r['HOSPITAL_SLUG']})")
    elif result["status"] == "found":
        cluster_id = result["cluster_id"]
        print(f"Found existing patient -- cluster_id = {cluster_id}")
        print(f"Matched against {len(result['matches'])} hospital-local record(s):")
        for r, s in sorted(result["matches"], key=lambda rs: rs[1], reverse=True):
            print(f"  score={s:.3f}  {r['FIRST']} {r['LAST']}  DOB={r['DOB']}  ({r['HOSPITAL_SLUG']})")
        print(f"\nRun: python scripts/09_patient_history_view.py --patient {cluster_id} ...")
    elif result["status"] == "ambiguous":
        print(f"AMBIGUOUS: matched records across {len(result['clusters'])} different existing patients -- "
              "needs a human to disambiguate rather than guessing:")
        for cluster_id, members in result["clusters"].items():
            print(f"\n  cluster_id = {cluster_id}")
            for r, s in members:
                print(f"    score={s:.3f}  {r['FIRST']} {r['LAST']}  DOB={r['DOB']}  ({r['HOSPITAL_SLUG']})")


if __name__ == "__main__":
    main()

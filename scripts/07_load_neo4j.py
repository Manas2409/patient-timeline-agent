"""
Load the merged per-patient timelines (data/unified_timelines/patient_*.json) into
Neo4j as a graph, so the eventual visit agent (Part B) can query "what does this
patient's history look like" as a graph traversal instead of scanning JSON files.

This Neo4j instance is shared with the sibling graph-triage project (same Aura Free
database -- Aura Free only gives one DB per instance). Every node/relationship this
script writes uses a PTA_ prefix so it can never collide with or be confused for
graph-triage's own TriageNode/Symptom/Check/Cause/Resolution/Escalation labels, and
this script only ever deletes nodes carrying its own PTA_ labels before a fresh load.

Schema:
  (:PTA_Patient {cluster_id, display_name, display_dob})  -- display_name/dob are a
      -- representative identity for showing a human name in reports (picked as the
      -- most complete-looking of this patient's hospital-local records); the real,
      -- inconsistent-per-hospital identity records live in data/hospitals/*/patients.csv
      -- and 10_lookup_patient.py matches against those directly, not this field.
  (:PTA_Hospital {name})
  (:PTA_Visit {visit_id, date, encounter_class, reason, seen_at})
  (:PTA_Concept {category, canonical})  -- one node per distinct normalized concept

  (:PTA_Patient)-[:HAD_VISIT]->(:PTA_Visit)
  (:PTA_Visit)-[:AT_HOSPITAL]->(:PTA_Hospital)
  (:PTA_Visit)-[:MENTIONS {text, match_method}]->(:PTA_Concept)
  (:PTA_Patient)-[:HAS_FACT {status, reported_by, num_hospitals}]->(:PTA_Concept)
      -- the corroboration-aware summary computed in 06_merge_timeline.py: one edge
      -- per (patient, concept) carrying whether it's confirmed or single_source, and
      -- by which hospitals. This is what the visit agent queries for red flags,
      -- rather than re-deriving corroboration status at query time.

Input:  data/unified_timelines/patient_*.json
Output: graph written to Neo4j (see .env for connection)
"""

import csv
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT / ".env")

TIMELINES_DIR = ROOT / "data" / "unified_timelines"
ENTITY_RES_DIR = ROOT / "data" / "entity_resolution"

PTA_LABELS = ["PTA_Patient", "PTA_Hospital", "PTA_Visit", "PTA_Concept"]


def clear_existing(session):
    for label in PTA_LABELS:
        session.run(f"MATCH (n:{label}) DETACH DELETE n")


def load_entity_resolution_module():
    spec = importlib.util.spec_from_file_location("entity_resolution", Path(__file__).parent / "03_entity_resolution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_display_identities() -> dict:
    """One representative name/DOB per cluster, for showing a human-readable name in
    reports instead of a bare cluster_id -- picked as whichever hospital-local record
    has the longest first+last name (a simple proxy for "least truncated/abbreviated",
    e.g. prefers "Thuy VonRueden" over "T. VonRueden")."""
    er = load_entity_resolution_module()
    by_cluster = defaultdict(list)
    for row in csv.DictReader(open(ENTITY_RES_DIR / "predicted_clusters.csv", encoding="utf-8")):
        by_cluster[row["CLUSTER_ID"]].append(row)

    display = {}
    for cluster_id, members in by_cluster.items():
        best = max(members, key=lambda m: len(m["FIRST"]) + len(m["LAST"]))
        dob_parsed = er.parse_dob(best["DOB"])
        display[cluster_id] = {
            "name": f"{best['FIRST']} {best['LAST']}".strip(),
            "dob": dob_parsed.isoformat() if dob_parsed else best["DOB"],
        }
    return display


def load_patient(session, record: dict, display: dict):
    """Batches everything for one patient into a handful of UNWIND queries instead of
    one round-trip per visit/entity -- with ~5,500 visits and ~7,900 entity mentions
    total, a query-per-row approach would mean tens of thousands of network round
    trips to Aura's cloud instance, which is the dominant cost at this data size."""
    cluster_id = record["cluster_id"]
    display_name = display.get("name", "")
    display_dob = display.get("dob", "")

    visit_rows = []
    mention_rows = []
    for i, visit in enumerate(record["timeline"]):
        visit_id = f"{cluster_id}_{i}"
        visit_rows.append({
            "visit_id": visit_id,
            "hospital": visit["hospital"],
            "date": visit["date"],
            "encounter_class": visit["encounter_class"],
            "reason": visit["reason"],
            "seen_at": visit["seen_at"],
        })
        for ent in visit["entities"]:
            mention_rows.append({
                "visit_id": visit_id,
                "category": ent["category"],
                "canonical": ent["canonical"],
                "text": ent["text"],
                "match_method": ent["match_method"],
                "outcome": ent.get("outcome", ""),
                "reaction": ent.get("reaction", ""),
                "severity": ent.get("severity", ""),
            })

    from collections import defaultdict
    hospitals_reporting = defaultdict(set)
    for visit in record["timeline"]:
        for ent in visit["entities"]:
            if ent["category"] in ("DIAGNOSIS", "SYMPTOM", "ALLERGY") and ent["match_method"] in ("exact_synonym", "fuzzy"):
                hospitals_reporting[(ent["category"], ent["canonical"])].add(visit["hospital"])

    num_hospitals = len(record["hospitals"])
    fact_rows = []
    for (category, canonical), reporting_hospitals in hospitals_reporting.items():
        if num_hospitals >= 2:
            status = "confirmed" if len(reporting_hospitals) >= 2 else "single_source"
        else:
            status = "single_hospital_patient"
        fact_rows.append({
            "category": category,
            "canonical": canonical,
            "status": status,
            "reported_by": sorted(reporting_hospitals),
            "num_hospitals": num_hospitals,
        })

    session.run(
        """
        MERGE (p:PTA_Patient {cluster_id: $cluster_id})
        SET p.display_name = $display_name, p.display_dob = $display_dob
        WITH p
        UNWIND $hospitals AS name
        MERGE (:PTA_Hospital {name: name})
        """,
        cluster_id=cluster_id, display_name=display_name, display_dob=display_dob,
        hospitals=record["hospitals"],
    )

    session.run(
        """
        MATCH (p:PTA_Patient {cluster_id: $cluster_id})
        UNWIND $rows AS row
        MERGE (v:PTA_Visit {visit_id: row.visit_id})
        SET v.date = row.date, v.encounter_class = row.encounter_class,
            v.reason = row.reason, v.seen_at = row.seen_at
        MERGE (p)-[:HAD_VISIT]->(v)
        WITH v, row
        MATCH (h:PTA_Hospital {name: row.hospital})
        MERGE (v)-[:AT_HOSPITAL]->(h)
        """,
        cluster_id=cluster_id, rows=visit_rows,
    )

    if mention_rows:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (v:PTA_Visit {visit_id: row.visit_id})
            MERGE (c:PTA_Concept {category: row.category, canonical: row.canonical})
            MERGE (v)-[m:MENTIONS]->(c)
            SET m.text = row.text, m.match_method = row.match_method,
                m.outcome = row.outcome, m.reaction = row.reaction, m.severity = row.severity
            """,
            rows=mention_rows,
        )

    if fact_rows:
        session.run(
            """
            MATCH (p:PTA_Patient {cluster_id: $cluster_id})
            UNWIND $rows AS row
            MERGE (c:PTA_Concept {category: row.category, canonical: row.canonical})
            MERGE (p)-[f:HAS_FACT]->(c)
            SET f.status = row.status, f.reported_by = row.reported_by, f.num_hospitals = row.num_hospitals
            """,
            cluster_id=cluster_id, rows=fact_rows,
        )


def main():
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USERNAME"]
    pwd = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    patient_files = sorted(TIMELINES_DIR.glob("patient_*.json"))

    with driver.session(database=database) as session:
        print("Clearing any previous PTA_ nodes from this database...")
        clear_existing(session)

        print("Creating constraints/indexes...")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:PTA_Patient) REQUIRE p.cluster_id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (h:PTA_Hospital) REQUIRE h.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (v:PTA_Visit) REQUIRE v.visit_id IS UNIQUE")
        # Composite uniqueness constraints need Enterprise; a composite index is
        # Community/Aura-Free-compatible and still speeds up the MERGE lookups here.
        session.run("CREATE INDEX IF NOT EXISTS FOR (c:PTA_Concept) ON (c.category, c.canonical)")

        display_identities = build_display_identities()

        print(f"Loading {len(patient_files)} patient timelines...")
        for i, path in enumerate(patient_files, 1):
            record = json.load(open(path, encoding="utf-8"))
            display = display_identities.get(record["cluster_id"], {})
            load_patient(session, record, display)
            if i % 20 == 0 or i == len(patient_files):
                print(f"  {i}/{len(patient_files)} patients loaded")

        counts = {}
        for label in PTA_LABELS:
            counts[label] = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
        rel_counts = {}
        for rel in ["HAD_VISIT", "AT_HOSPITAL", "MENTIONS", "HAS_FACT"]:
            rel_counts[rel] = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]

    driver.close()

    print("\nLoad complete. Node counts:")
    for label, n in counts.items():
        print(f"  {label:<15} {n}")
    print("Relationship counts:")
    for rel, n in rel_counts.items():
        print(f"  {rel:<15} {n}")


if __name__ == "__main__":
    main()

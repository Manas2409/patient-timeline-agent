"""
The combined patient visit report: today's actionable briefing (reusing
08_visit_agent.py's judgment, not duplicating it) up top, followed by the complete
chronological history -- every visit, every diagnosis/symptom/medication/allergy,
with confidence/outcome/severity/reaction and confirmed-vs-single-source status on
each item -- as supporting reference below. The briefing answers "what matters right
now"; the history below it answers "show me everything, so I can check your work and
see the bigger picture." One report, not two things a doctor has to mentally merge.

If no --complaint is given, this is just the full history (no briefing section) --
useful on its own for reviewing a patient's complete record outside of a specific
visit.

Fields shown, and where each comes from:
  date, hospital/source       -- PTA_Visit / PTA_Hospital (script 7)
  diagnosis/condition, symptom, medication, allergy -- PTA_Concept via MENTIONS (script 4/5)
  confidence                  -- MENTIONS.match_method (exact_synonym/fuzzy/ambiguous/unmapped)
  conflicts (confirmed vs single-source) -- PTA_Patient-[:HAS_FACT]->PTA_Concept (script 6/7)
  outcome (resolved/ongoing)  -- MENTIONS.outcome, added in script 6 from each record's
                                  own STOP date (approximate: matched to the entity by
                                  fuzzy text match against the structured record logged
                                  at the same encounter, not a guaranteed exact link)
  allergy reaction/severity   -- MENTIONS.reaction / MENTIONS.severity, from Synthea's
                                  own structured allergy fields (script 6)

"Complete" means every clinical finding, not every row: visits with nothing
extracted, Synthea's social/administrative findings (employment status, etc. --
see vocab.py's ADMIN_SDOH_DESCRIPTIONS), and bare unmapped NER fragments too short
to carry meaning on their own ("sin", "bro") are all omitted as noise, not signal.

Known gap (not shown per-run, but real): procedures/treatments and a general
severity rating for diagnoses/symptoms are not in this view -- deferred (Tier 3),
see README.

Usage:
  python scripts/09_patient_history_view.py --patient <cluster_id> [--complaint "..." [--treatment "..."]]
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from rapidfuzz import fuzz

from vocab import ADMIN_SDOH_DESCRIPTIONS, is_admin_sdoh

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT / ".env")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

HIGHLIGHT_THRESHOLD = 85  # higher than 05's fuzzy-match threshold (70-80) because the
# targets here are long free-text sentences, not short structured vocab terms -- more
# candidate windows means more chances for a coincidental match (observed: "Norinyl"
# scored 71 against an unrelated red flag description purely by coincidental letter
# overlap, while genuine matches -- the actual drug name named in the flag -- score 100).
MIN_HIGHLIGHT_LENGTH = 5  # same reasoning as MIN_FUZZY_LENGTH in 05_normalize_entities.py --
# a short fragment like "et" or "Nat" can score a coincidentally high partial_ratio
# against any long enough description text, which would look like a confident match
# while actually being meaningless.


def load_visit_agent_module():
    spec = importlib.util.spec_from_file_location("visit_agent", Path(__file__).parent / "08_visit_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_full_history(session, cluster_id: str):
    summary = session.run(
        """
        MATCH (p:PTA_Patient {cluster_id: $cluster_id})-[:HAD_VISIT]->(v:PTA_Visit)-[:AT_HOSPITAL]->(h:PTA_Hospital)
        RETURN p.display_name AS display_name, p.display_dob AS display_dob,
               count(DISTINCT v) AS num_visits, count(DISTINCT h) AS num_hospitals,
               min(v.date) AS earliest, max(v.date) AS latest, collect(DISTINCT h.name) AS hospitals
        """,
        cluster_id=cluster_id,
    ).single()
    if summary is None or summary["num_visits"] == 0:
        return None, None, None

    visits = session.run(
        """
        MATCH (p:PTA_Patient {cluster_id: $cluster_id})-[:HAD_VISIT]->(v:PTA_Visit)-[:AT_HOSPITAL]->(h:PTA_Hospital)
        OPTIONAL MATCH (v)-[m:MENTIONS]->(c:PTA_Concept)
        RETURN v.date AS date, v.encounter_class AS encounter_class, v.reason AS reason,
               h.name AS hospital,
               collect({category: c.category, canonical: c.canonical, match_method: m.match_method,
                         outcome: m.outcome, reaction: m.reaction, severity: m.severity}) AS mentions
        ORDER BY v.date
        """,
        cluster_id=cluster_id,
    ).data()

    facts = session.run(
        """
        MATCH (p:PTA_Patient {cluster_id: $cluster_id})-[f:HAS_FACT]->(c:PTA_Concept)
        RETURN c.category AS category, c.canonical AS canonical, f.status AS status, f.reported_by AS reported_by
        ORDER BY f.status, c.category
        """,
        cluster_id=cluster_id,
    ).data()

    return dict(summary), visits, facts


def run_visit_agent(cluster_id: str, complaint: str, treatment: str, clarification: str = ""):
    """Reuses the visit agent's own judgment rather than re-implementing it, so the
    briefing and the highlights below it can never disagree with each other."""
    va = load_visit_agent_module()
    client = va.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        patient_context = va.fetch_patient_context(session, cluster_id)
    driver.close()

    user_message = va.build_user_message(patient_context, complaint, treatment, clarification)
    return va.call_agent(client, user_message)


def build_highlight_entries(agent_result: dict) -> list:
    """One entry per thing the agent called out, each carrying the short reason so the
    full-history view can show *why* something is marked relevant, not just that it is.
    related_canonical is the model's own exact-name reference (added after discovering
    that fuzzy-matching free-text descriptions back to canonical names was flaky --
    model phrasing varies call to call, so a description that didn't happen to name the
    drug verbatim would silently fail to highlight). Fuzzy matching is kept only as a
    fallback for whatever the model leaves untagged."""
    entries = []
    for h in agent_result.get("relevant_history", []):
        entries.append({"phrase": h["fact"], "why": h["why_relevant"], "related_canonical": h.get("related_canonical", "")})
    for r in agent_result.get("red_flags", []):
        entries.append({
            "phrase": r["description"],
            "why": f"RED FLAG ({r['severity'].upper()}, {r['flag_type']})",
            "related_canonical": r.get("related_canonical", ""),
        })
    return entries


def match_highlight(canonical: str, entries: list):
    if not entries:
        return None

    # Exact match first (deterministic): canonical may be a pipe-joined list of
    # candidates for an "ambiguous" normalization match, so check each part.
    canonical_parts = {part.strip().lower() for part in canonical.split("|")}
    for e in entries:
        related = e.get("related_canonical", "").strip().lower()
        if related and related in canonical_parts:
            return e

    # Fuzzy fallback, for entries the model didn't tag with related_canonical.
    if len(canonical) < MIN_HIGHLIGHT_LENGTH:
        return None
    best, best_score = None, 0
    for e in entries:
        score = fuzz.partial_ratio(canonical.lower(), e["phrase"].lower())
        if score >= HIGHLIGHT_THRESHOLD and score > best_score:
            best, best_score = e, score
    return best


def print_todays_visit(complaint: str, treatment: str, agent_result: dict):
    print("--- TODAY'S VISIT ---")
    print(f"Complaint: {complaint}")
    if treatment:
        print(f"Treatment under consideration: {treatment}")

    if agent_result["status"] == "needs_clarification":
        print(f"\nThe agent needs more information before briefing confidently:")
        print(f"  {agent_result.get('clarifying_question', '(no question provided)')}")
        print("(Full history is still shown below for reference.)")
        return

    print(f"\n{agent_result.get('briefing', '').strip()}")
    if agent_result.get("red_flags"):
        print("\nRed flags:")
        for r in agent_result["red_flags"]:
            print(f"  ({r['severity'].upper()}) [{r['flag_type']}] {r['description']}")
    if agent_result.get("uncertainty_notes"):
        print("\nTransparency notes:")
        for n in agent_result["uncertainty_notes"]:
            print(f"  - {n}")


MIN_DISPLAY_LENGTH = 5  # same reasoning as MIN_HIGHLIGHT_LENGTH -- a bare 2-4 char
# unmapped NER fragment ("sin", "bro", "Risk") carries no real clinical meaning on its
# own and is just noise in a report meant to be comprehensive, not exhaustive.


SDOH_FRAGMENT_THRESHOLD = 85


def is_admin_sdoh_fragment(canonical: str) -> bool:
    """is_admin_sdoh (vocab.py) only catches an exact phrase match, but NER output is
    often a truncated fragment of the original Synthea description ("higher
    education" instead of "received higher education", "Limited" instead of "limited
    social contact") -- these still need filtering out of a clinical report, so this
    falls back to a fuzzy substring check against the same exclusion list."""
    if is_admin_sdoh(canonical):
        return True
    if len(canonical) < MIN_DISPLAY_LENGTH:
        return False
    return any(fuzz.partial_ratio(canonical.lower(), desc) >= SDOH_FRAGMENT_THRESHOLD for desc in ADMIN_SDOH_DESCRIPTIONS)


def is_worth_showing(m: dict) -> bool:
    if m["category"] is None:
        return False
    if is_admin_sdoh_fragment(m["canonical"]):
        return False  # SDOH/administrative, not a clinical fact -- see vocab.py
    if m["match_method"] == "unmapped" and len(m["canonical"]) < MIN_DISPLAY_LENGTH:
        return False
    return True


def print_full_history(summary: dict, visits: list, facts: list, highlight_entries: list):
    print("\n--- COMPLETE CHRONOLOGICAL HISTORY ---")
    print(f"{summary['num_visits']} visits across {summary['num_hospitals']} hospitals "
          f"({summary['earliest']} to {summary['latest']})")
    print("Hospitals: " + ", ".join(summary["hospitals"]))

    shown_visits = 0
    for visit in visits:
        mentions = [m for m in visit["mentions"] if is_worth_showing(m)]
        if not mentions:
            continue  # a visit with nothing extracted adds no information here
        shown_visits += 1
        print(f"\n[{visit['date']}] {visit['hospital']} -- {visit['encounter_class']}"
              + (f" (reason: {visit['reason']})" if visit["reason"] else ""))
        for m in mentions:
            line = f"    [{m['category']}] {m['canonical']}  (confidence: {m['match_method']})"
            if m.get("outcome"):
                line += f"  outcome: {m['outcome']}"
            if m.get("severity"):
                line += f"  severity: {m['severity']}"
            if m.get("reaction"):
                line += f"  reaction: {m['reaction']}"
            match = match_highlight(m["canonical"], highlight_entries)
            if match:
                line += f"  <<< RELEVANT TODAY -- {match['why']}"
            print(line)

    skipped = len(visits) - shown_visits
    if skipped:
        print(f"\n({skipped} visits with no extracted clinical findings omitted)")

    print("\n--- CROSS-HOSPITAL AGREEMENT (all known facts, not just today's) ---")
    for f in facts:
        tag = {"confirmed": "[CONFIRMED]", "single_source": "[UNCONFIRMED - single source]",
               "single_hospital_patient": "[ONLY HOSPITAL ON FILE]"}.get(f["status"], f["status"])
        print(f"  {tag} [{f['category']}] {f['canonical']}  (reported by: {', '.join(f['reported_by'])})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True, help="Entity-resolution cluster id")
    parser.add_argument("--complaint", default="", help="Today's complaint -- adds a briefing section and highlights")
    parser.add_argument("--treatment", default="", help="Treatment under consideration today")
    parser.add_argument("--clarification", default="", help="Answer to a previous clarifying question")
    args = parser.parse_args()

    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        summary, visits, facts = fetch_full_history(session, args.patient)
    driver.close()

    if summary is None:
        print(f"No patient found with cluster id {args.patient!r}.")
        sys.exit(1)

    print("=" * 78)
    name = summary.get("display_name") or f"cluster {args.patient}"
    dob_part = f"  (DOB {summary['display_dob']})" if summary.get("display_dob") else ""
    print(f"PATIENT VISIT REPORT -- {name}{dob_part}  [cluster_id={args.patient}]")
    print("=" * 78)

    highlight_entries = []
    if args.complaint:
        agent_result = run_visit_agent(args.patient, args.complaint, args.treatment, args.clarification)
        print_todays_visit(args.complaint, args.treatment, agent_result)
        highlight_entries = build_highlight_entries(agent_result)

    print_full_history(summary, visits, facts, highlight_entries)


if __name__ == "__main__":
    main()

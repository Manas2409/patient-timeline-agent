"""
Part B: the Smart Visit Agent. Given a patient (by entity-resolution cluster id) and
their current complaint/reason for visit, produces a short doctor-facing briefing
instead of a dump of the full history -- the thing Part A's merge/graph work was
built to make possible.

Why this counts as "agentic" rather than a summarizer: it reasons over the patient's
history *conditioned on the current complaint* (not every fact is relevant to every
visit), decides for itself whether it has enough information to give a confident
briefing or needs to ask a clarifying question first, and treats confirmed vs
single-source facts differently rather than presenting everything with equal
confidence.

Context size note: patients in this dataset average ~52 visits (max 565) but only
~4 distinct clinical facts and ~9.5 distinct medications (max 20 / 28) once
corroboration-aware normalization has done its job (see 05/06/07). So the agent is
handed the compact fact/medication summary, never the raw visit list -- dumping raw
visits would defeat the entire point of this being a "smart" briefing rather than a
records printout.

Known limitation: drug-interaction checking here relies on the model's own medical
knowledge, not a real pharmacological interaction database (e.g. a real system would
call something like a First Databank / Micromedex API). Fine for a project demo, not
a substitute for one in an actual clinical tool.

Uses OpenAI (function calling) for the reasoning step rather than Claude -- a provider
choice made for this project's own billing reasons, not a technical requirement.

Usage:
  python scripts/08_visit_agent.py --patient <cluster_id> --complaint "..." [--treatment "..."] [--clarification "..."]
"""

import argparse
import json
import os
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252,
    # which can't render the curly quotes/dashes the model's output often contains

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT / ".env")

MODEL = "gpt-5"

BRIEFING_TOOL = {
    "type": "function",
    "function": {
    "name": "emit_briefing",
    "description": "Emit the visit briefing or a clarifying question.",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["ready", "needs_clarification"],
                "description": "'needs_clarification' if the complaint is too vague/ambiguous to relate to the history confidently.",
            },
            "clarifying_question": {
                "type": "string",
                "description": "One specific question that would resolve the ambiguity. Required if status is needs_clarification.",
            },
            "relevant_history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact": {"type": "string"},
                        "related_canonical": {
                            "type": "string",
                            "description": "The exact canonical name this refers to, copied verbatim (same spelling/case) "
                                           "from the KNOWN DIAGNOSES/SYMPTOMS/ALLERGIES or MEDICATION HISTORY list below. "
                                           "This is used for exact programmatic lookup, not read by a human, so it must "
                                           "match one of those listed names character-for-character.",
                        },
                        "confirmation_status": {"type": "string", "enum": ["confirmed", "single_source", "single_hospital_patient"]},
                        "why_relevant": {"type": "string"},
                    },
                    "required": ["fact", "related_canonical", "confirmation_status", "why_relevant"],
                },
                "description": "Only facts actually pertinent to today's complaint -- not the full history.",
            },
            "red_flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "flag_type": {"type": "string", "enum": ["drug_interaction", "recurring_symptom", "relevant_allergy"]},
                        "related_canonical": {
                            "type": "string",
                            "description": "The exact canonical name of the medication/fact causing this flag, copied "
                                           "verbatim (same spelling/case) from the MEDICATION HISTORY or KNOWN "
                                           "DIAGNOSES/SYMPTOMS/ALLERGIES list below -- must match one of those listed "
                                           "names character-for-character. Leave empty only if this flag genuinely isn't "
                                           "tied to one specific listed item.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Must name the specific fact from this patient's data that makes it a flag "
                                           "(e.g. the exact medication from MEDICATION HISTORY, or the exact allergy/symptom "
                                           "from KNOWN DIAGNOSES/SYMPTOMS/ALLERGIES). Do not describe a hypothetical "
                                           "interaction with a drug that isn't actually listed for this patient.",
                        },
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["flag_type", "related_canonical", "description", "severity"],
                },
                "description": "Only flags grounded in this patient's actual recorded data -- an empty list is the "
                                "correct answer when nothing on file actually interacts with today's complaint/treatment.",
            },
            "uncertainty_notes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Plain-language callouts of anything unconfirmed/single-source that the doctor should know is not fully corroborated.",
            },
            "briefing": {
                "type": "string",
                "description": "3-6 sentence doctor-facing summary. Only when status is ready.",
            },
        },
        "required": ["status"],
    },
    },
}

SYSTEM_PROMPT = """You are a clinical decision-support assistant. A doctor is about to see a \
patient for a new complaint. You have that patient's history pulled from multiple \
hospitals, already deduplicated and normalized -- but hospitals don't always agree, \
so each fact is tagged with a confirmation status:
  - confirmed: reported independently by 2+ of the patient's hospitals
  - single_source: reported by only 1 hospital, even though the patient has visited others
  - single_hospital_patient: this patient has only ever been seen at one hospital, so there's nothing to corroborate against

Your job is NOT to summarize the entire history. Only surface what's actually relevant \
to today's complaint: recurring/unresolved symptoms matching it, allergies relevant to \
likely treatments, and drug interactions with anything the doctor is considering \
prescribing today. Never present a single_source fact with the same confidence as a \
confirmed one -- say so explicitly if it matters to today's visit.

Stick to this patient's actual data. A drug_interaction red flag must name a specific \
medication that appears in the MEDICATION HISTORY list you were given below -- if \
nothing in that list interacts with what's being considered, say plainly that no \
interaction is apparent from the recorded medications, rather than listing generic \
interactions with drugs that aren't actually in this patient's history. The same rule \
applies to relevant_history and uncertainty_notes: ground every claim in a fact you \
were actually given, not general background knowledge about the condition or drug \
class. It's fine, and often correct, for the briefing to come back short.

If the complaint is too vague to confidently connect to the history (e.g. it could \
plausibly relate to several very different past conditions and the right course of \
action genuinely depends on which), set status to needs_clarification and ask ONE \
specific question rather than guessing. Otherwise set status to ready and give a \
short, concrete briefing.

You are not a substitute for a real drug-interaction database -- flag interactions \
you're aware of between what's being considered and medications actually on this \
patient's chart, but note this is based on general medical knowledge, not a live \
pharmacology lookup. Do not use this as license to list hypothetical interactions \
with drugs the patient isn't recorded as taking."""


def fetch_patient_context(session, cluster_id: str) -> dict:
    facts = session.run(
        """
        MATCH (p:PTA_Patient {cluster_id: $cluster_id})-[f:HAS_FACT]->(c:PTA_Concept)
        RETURN c.category AS category, c.canonical AS canonical,
               f.status AS status, f.reported_by AS reported_by
        ORDER BY c.category, c.canonical
        """,
        cluster_id=cluster_id,
    ).data()

    medications = session.run(
        """
        MATCH (p:PTA_Patient {cluster_id: $cluster_id})-[:HAD_VISIT]->(v:PTA_Visit)
              -[:MENTIONS]->(c:PTA_Concept {category: "MEDICATION"})
        MATCH (v)-[:AT_HOSPITAL]->(h:PTA_Hospital)
        WITH c.canonical AS medication, min(v.date) AS first_seen, max(v.date) AS last_seen,
             collect(DISTINCT h.name) AS hospitals
        RETURN medication, first_seen, last_seen, hospitals
        ORDER BY last_seen DESC
        """,
        cluster_id=cluster_id,
    ).data()

    summary = session.run(
        """
        MATCH (p:PTA_Patient {cluster_id: $cluster_id})-[:HAD_VISIT]->(v:PTA_Visit)-[:AT_HOSPITAL]->(h:PTA_Hospital)
        RETURN count(DISTINCT v) AS num_visits, count(DISTINCT h) AS num_hospitals,
               min(v.date) AS earliest, max(v.date) AS latest
        """,
        cluster_id=cluster_id,
    ).single()

    if summary is None or summary["num_visits"] == 0:
        return None

    return {
        "facts": facts,
        "medications": medications,
        "summary": dict(summary),
    }


def build_user_message(patient_context: dict, complaint: str, treatment: str, clarification: str) -> str:
    parts = [
        f"PATIENT SUMMARY: {patient_context['summary']['num_visits']} visits across "
        f"{patient_context['summary']['num_hospitals']} hospitals, "
        f"{patient_context['summary']['earliest']} to {patient_context['summary']['latest']}.",
        "\nKNOWN DIAGNOSES/SYMPTOMS/ALLERGIES:",
    ]
    for f in patient_context["facts"]:
        parts.append(f"  [{f['category']}] {f['canonical']} -- {f['status']} (reported by: {', '.join(f['reported_by'])})")

    parts.append("\nMEDICATION HISTORY:")
    for m in patient_context["medications"]:
        parts.append(f"  {m['medication']} -- seen {m['first_seen']} to {m['last_seen']} at {', '.join(m['hospitals'])}")

    parts.append(f"\nTODAY'S COMPLAINT: {complaint}")
    if treatment:
        parts.append(f"PROPOSED TREATMENT UNDER CONSIDERATION TODAY: {treatment}")
    if clarification:
        parts.append(f"\n(Following up on your clarifying question, the doctor answered: {clarification})")

    return "\n".join(parts)


def call_agent(client, user_message: str):
    response = client.chat.completions.create(
        model=MODEL,
        # gpt-5 is a reasoning model -- its hidden reasoning tokens count against
        # max_completion_tokens, so a tight budget can get fully consumed by
        # reasoning before any tool-call output is emitted (finish_reason="length").
        max_completion_tokens=6000,
        tools=[BRIEFING_TOOL],
        tool_choice={"type": "function", "function": {"name": "emit_briefing"}},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    choice = response.choices[0]
    tool_calls = choice.message.tool_calls
    if tool_calls:
        return json.loads(tool_calls[0].function.arguments)
    raise RuntimeError(
        f"Model did not return the expected tool call "
        f"(finish_reason={choice.finish_reason!r}, content={choice.message.content!r})"
    )


def print_briefing(result: dict):
    if result["status"] == "needs_clarification":
        print("\nThe agent needs more information before it can brief confidently:")
        print(f"  {result.get('clarifying_question', '(no question provided)')}")
        print("\nRerun with --clarification \"<answer>\" to continue.")
        return

    print("\n" + "=" * 70)
    print("VISIT BRIEFING")
    print("=" * 70)
    print(result.get("briefing", "").strip())

    if result.get("relevant_history"):
        print("\nRELEVANT HISTORY:")
        for h in result["relevant_history"]:
            tag = "[CONFIRMED]" if h["confirmation_status"] == "confirmed" else "[UNCONFIRMED - single source]"
            print(f"  {tag} {h['fact']}")
            print(f"      why: {h['why_relevant']}")

    if result.get("red_flags"):
        print("\nRED FLAGS:")
        for r in result["red_flags"]:
            print(f"  ({r['severity'].upper()}) [{r['flag_type']}] {r['description']}")

    if result.get("uncertainty_notes"):
        print("\nTRANSPARENCY NOTES:")
        for n in result["uncertainty_notes"]:
            print(f"  - {n}")

    print("\n(Drug-interaction flags are based on the model's general medical knowledge, "
          "not a live pharmacology database -- verify independently.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True, help="Entity-resolution cluster id")
    parser.add_argument("--complaint", required=True, help="Today's reason for visit / chief complaint")
    parser.add_argument("--treatment", default="", help="Treatment being considered today, if any")
    parser.add_argument("--clarification", default="", help="Answer to a previous clarifying question")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )

    with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        patient_context = fetch_patient_context(session, args.patient)
    driver.close()

    if patient_context is None:
        print(f"No patient found with cluster id {args.patient!r}.")
        sys.exit(1)

    user_message = build_user_message(patient_context, args.complaint, args.treatment, args.clarification)
    result = call_agent(client, user_message)
    print_briefing(result)


if __name__ == "__main__":
    main()

"""
Streamlit front end for the realistic check-in -> report workflow. Reuses every
piece of logic from the CLI scripts (dynamic import, same pattern those scripts
already use for each other) rather than reimplementing any of it -- this app is a
presentation layer on top of 10_lookup_patient.py and 09_patient_history_view.py,
nothing more.

Flow:
  1. Front-desk form (name + DOB [+ optional details]) -> resolves to a cluster_id
     via the same fuzzy matching as script 10.
  2. Once a patient is resolved, a complaint (+ optional treatment) form generates
     the combined report -- the same briefing + full annotated history as script 09,
     rendered as a page instead of printed text. Handles the agent's clarifying-
     question path interactively (answer in a box, resubmit) instead of requiring a
     second CLI invocation.

Run with: streamlit run streamlit_app.py
"""

import importlib.util
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env")

# scripts/*.py import each other with bare `from vocab import ...` / dynamic
# importlib.util loads of sibling scripts, which only resolves if scripts/ is on
# sys.path -- true automatically when Python runs one of those files directly (it
# adds the script's own directory), but not when this app loads them from the
# project root.
sys.path.insert(0, str(ROOT / "scripts"))

st.set_page_config(page_title="Unified Patient History", layout="wide")


@st.cache_resource
def load_module(filename: str):
    spec = importlib.util.spec_from_file_location(filename, ROOT / "scripts" / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource
def get_driver():
    return GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))


lk = load_module("10_lookup_patient")
hv = load_module("09_patient_history_view")

for key, default in [
    ("cluster_id", None), ("patient_label", None), ("lookup_result", None),
    ("agent_result", None), ("history", None), ("complaint", ""), ("treatment", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def reset_patient():
    for key in ["cluster_id", "patient_label", "lookup_result", "agent_result", "history", "complaint", "treatment"]:
        st.session_state[key] = None if key not in ("complaint", "treatment") else ""


st.title("Unified Patient History & Smart Visit Agent")

# ---------------------------------------------------------------------------
# Step 1: Check-in -- resolve a patient from name + DOB, the way a front desk would.
# ---------------------------------------------------------------------------
if st.session_state.cluster_id is None:
    st.header("Patient Check-In")
    with st.form("checkin_form"):
        col1, col2, col3 = st.columns(3)
        first = col1.text_input("First name *")
        last = col2.text_input("Last name *")
        gender = col3.selectbox("Gender *", ["F", "M"])
        dob = st.text_input("Date of birth *", placeholder="YYYY-MM-DD, MM/DD/YYYY, or 'Mon D, YYYY'")

        with st.expander("More details (improves match confidence, all optional)"):
            mcol1, mcol2 = st.columns(2)
            middle = mcol1.text_input("Middle name")
            ssn = mcol2.text_input("SSN")
            acol1, acol2, acol3 = st.columns(3)
            address = acol1.text_input("Street address")
            city = acol2.text_input("City")
            state = acol3.text_input("State")

        submitted = st.form_submit_button("Look up patient")

    if submitted:
        if not first or not last or not dob:
            st.error("First name, last name, and DOB are required.")
        else:
            with st.spinner("Searching existing records..."):
                result = lk.lookup_patient(first, last, dob, gender, middle, address, city, state, ssn)
            st.session_state.lookup_result = result

    result = st.session_state.lookup_result
    if result:
        if result["status"] == "invalid_dob":
            st.error(f"Could not parse that DOB. Accepted formats: {result['accepted_formats']}")

        elif result["status"] == "not_found":
            st.warning("No existing patient found. This would be registered as a new patient.")
            st.caption("Closest non-matching records on file, for a human to double-check:")
            st.table([
                {"score": round(s, 3), "name": f"{r['FIRST']} {r['LAST']}", "dob": r["DOB"], "hospital": r["HOSPITAL_SLUG"]}
                for r, s in result["closest"]
            ])

        elif result["status"] == "found":
            cluster_id = result["cluster_id"]
            st.success(f"Found existing patient -- cluster_id = {cluster_id}")
            st.caption(f"Matched against {len(result['matches'])} hospital-local record(s):")
            st.table([
                {"score": round(s, 3), "name": f"{r['FIRST']} {r['LAST']}", "dob": r["DOB"], "hospital": r["HOSPITAL_SLUG"]}
                for r, s in sorted(result["matches"], key=lambda rs: rs[1], reverse=True)
            ])
            if st.button("Continue to visit"):
                st.session_state.cluster_id = cluster_id
                st.rerun()

        elif result["status"] == "ambiguous":
            st.warning(f"Matched records across {len(result['clusters'])} different existing patients -- "
                       "please confirm which one:")
            for cid, members in result["clusters"].items():
                best = max(members, key=lambda rs: rs[1])[0]
                if st.button(f"{best['FIRST']} {best['LAST']} (DOB {best['DOB']}) -- cluster_id {cid}", key=f"amb_{cid}"):
                    st.session_state.cluster_id = cid
                    st.rerun()

# ---------------------------------------------------------------------------
# Step 2: Today's visit -- the combined briefing + full history report.
# ---------------------------------------------------------------------------
else:
    driver = get_driver()
    with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        summary, visits, facts = hv.fetch_full_history(session, st.session_state.cluster_id)

    name = summary.get("display_name") or f"cluster {st.session_state.cluster_id}"
    dob_part = f" (DOB {summary['display_dob']})" if summary.get("display_dob") else ""
    header_col, button_col = st.columns([5, 1])
    header_col.header(f"{name}{dob_part}")
    header_col.caption(f"cluster_id={st.session_state.cluster_id} -- {summary['num_visits']} visits across "
                        f"{summary['num_hospitals']} hospitals ({summary['earliest']} to {summary['latest']})")
    if button_col.button("New check-in"):
        reset_patient()
        st.rerun()

    st.subheader("Today's Visit")
    with st.form("visit_form"):
        complaint = st.text_area("Complaint / reason for visit *", value=st.session_state.complaint)
        treatment = st.text_input("Treatment under consideration (optional)", value=st.session_state.treatment)
        go = st.form_submit_button("Generate report")

    if go and complaint:
        st.session_state.complaint = complaint
        st.session_state.treatment = treatment
        with st.spinner("Reviewing history and drafting briefing..."):
            st.session_state.agent_result = hv.run_visit_agent(st.session_state.cluster_id, complaint, treatment)

    agent_result = st.session_state.agent_result
    highlight_entries = []

    if agent_result:
        if agent_result["status"] == "needs_clarification":
            st.info(f"**The agent needs more information:** {agent_result.get('clarifying_question', '')}")
            with st.form("clarify_form"):
                clarification = st.text_input("Your answer")
                clarify_go = st.form_submit_button("Submit clarification")
            if clarify_go and clarification:
                with st.spinner("Incorporating your answer..."):
                    st.session_state.agent_result = hv.run_visit_agent(
                        st.session_state.cluster_id, st.session_state.complaint, st.session_state.treatment, clarification
                    )
                st.rerun()
        else:
            st.markdown(f"> {agent_result.get('briefing', '').strip()}")

            if agent_result.get("red_flags"):
                st.markdown("**Red flags**")
                for r in agent_result["red_flags"]:
                    box = {"high": st.error, "medium": st.warning, "low": st.info}.get(r["severity"], st.info)
                    box(f"[{r['flag_type']}] {r['description']}")

            if agent_result.get("uncertainty_notes"):
                st.markdown("**Transparency notes**")
                for n in agent_result["uncertainty_notes"]:
                    st.caption(f"- {n}")

            highlight_entries = hv.build_highlight_entries(agent_result)

    st.divider()
    st.subheader("Complete Chronological History")
    if highlight_entries:
        st.caption("Visits with something relevant to today's complaint are pinned to the top (🔴), "
                   "in date order; the rest follow below, also in date order.")

    # Filter out empty visits first, then stable-sort by relevance so the 🔴 ones
    # float to the top -- a stable sort preserves each group's original (date) order,
    # so it's "relevant visits chronologically, then the rest chronologically," not a
    # full re-shuffle.
    filtered = []
    for visit in visits:
        mentions = [m for m in visit["mentions"] if hv.is_worth_showing(m)]
        if not mentions:
            continue
        any_relevant = any(hv.match_highlight(m["canonical"], highlight_entries) for m in mentions) if highlight_entries else False
        filtered.append((visit, mentions, any_relevant))
    filtered.sort(key=lambda row: not row[2])

    shown = len(filtered)
    for visit, mentions, any_relevant in filtered:
        title = f"[{visit['date']}] {visit['hospital']} -- {visit['encounter_class']}"
        if visit["reason"]:
            title += f" (reason: {visit['reason']})"
        with st.expander(("🔴 " if any_relevant else "") + title, expanded=any_relevant):
            for m in mentions:
                line = f"**[{m['category']}]** {m['canonical']}  _{m['match_method']}_"
                if m.get("outcome"):
                    line += f"  ·  outcome: {m['outcome']}"
                if m.get("severity"):
                    line += f"  ·  severity: {m['severity']}"
                if m.get("reaction"):
                    line += f"  ·  reaction: {m['reaction']}"
                match = hv.match_highlight(m["canonical"], highlight_entries) if highlight_entries else None
                if match:
                    st.markdown(f"🔴 {line}")
                    st.caption(f"RELEVANT TODAY -- {match['why']}")
                else:
                    st.markdown(line)

    skipped = len(visits) - shown
    if skipped:
        st.caption(f"({skipped} visits with no extracted clinical findings omitted)")

    st.subheader("Cross-Hospital Agreement")
    tag_map = {"confirmed": "CONFIRMED", "single_source": "UNCONFIRMED (single source)", "single_hospital_patient": "ONLY HOSPITAL ON FILE"}
    st.table([
        {"status": tag_map.get(f["status"], f["status"]), "category": f["category"], "fact": f["canonical"],
         "reported_by": ", ".join(f["reported_by"])}
        for f in facts
    ])

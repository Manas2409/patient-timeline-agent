"""
Shatter Synthea's unified patient records across a small number of consolidated
regional "hospitals" (built by grouping the 278 real Synthea organizations by
Massachusetts ZIP region), each with its own local patient ID and its own
(corrupted, inconsistent) view of that patient's identity. This simulates the
real-world entity resolution problem: the same person shows up at Hospital A as
"John A. Smith, DOB 1985-03-02" and at Hospital B as "J. Smith, 03/02/1985".

Regional consolidation (rather than one silo per real organization) exists because
108 sample patients spread across 278 real orgs leaves most orgs with only 1-2
patients -- too sparse to look like an actual hospital roster. Grouping by region
approximates how a real regional health system operates: a patient's visits to
different departments/clinics within the same network land in one chart, not one
per building.

Ground truth (which local records belong to the same real patient) is preserved in
data/ground_truth/patient_crosswalk.csv so entity-resolution accuracy can be scored.

Input:  data/raw/{patients,encounters,conditions,medications,allergies,organizations}.csv
Output: data/hospitals/<hospital_slug>/{patients,encounters,conditions,medications,allergies}.csv
        data/ground_truth/patient_crosswalk.csv
"""

import calendar
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_HOSPITALS = Path(__file__).resolve().parent.parent / "data" / "hospitals"
OUT_GT = Path(__file__).resolve().parent.parent / "data" / "ground_truth"

random.seed(42)  # reproducible corruption

# Consolidate Synthea's 278 real orgs into regional hospital networks, based on the
# geography their ZIP prefixes actually cluster into. Deliberately kept finer-grained
# than a naive "6 MA regions" split: merging too aggressively (e.g. all of Greater
# Boston into one bucket) also merges the exact clinics a given patient would visit
# repeatedly, which collapses the cross-hospital signal entity resolution needs to
# demonstrate anything. This split was chosen by checking, empirically, that it keeps
# ~47% of patients touching 2+ hospitals while avoiding near-empty hospital rosters.
ZIP_PREFIX_TO_REGION = {
    "010": "Pioneer Valley Health Network",
    "011": "Pioneer Valley Health Network",
    "012": "Berkshires Regional Health",
    "014": "North Central Mass Health",
    "015": "Worcester Regional Medical Center",
    "016": "Worcester Regional Medical Center",
    "017": "MetroWest Health Partners",
    "018": "Merrimack Valley Health Network",
    "019": "North Shore Health Alliance",
    "020": "Norfolk County Health Network",
    "021": "Greater Boston Medical Center",
    "022": "Greater Boston Medical Center",
    "024": "West Suburban Boston Health",
    "023": "South Shore Health Alliance",
    "025": "Cape Cod & Islands Health",
    "026": "Cape Cod & Islands Health",
    "027": "Southcoast Health",
}
DEFAULT_REGION = "Regional Community Health Network"


def region_for_org(org: dict) -> str:
    return ZIP_PREFIX_TO_REGION.get(org["ZIP"][:3], DEFAULT_REGION)

STREET_ABBREV = {
    "Street": "St", "Avenue": "Ave", "Boulevard": "Blvd", "Drive": "Dr",
    "Lane": "Ln", "Road": "Rd", "Court": "Ct", "Circle": "Cir",
    "Place": "Pl", "Way": "Way", "Terrace": "Ter", "Square": "Sq",
    "Tunnel": "Tun", "Underpass": "Underpass", "Ville": "Vl",
    "Lodge": "Lodge", "Pike": "Pike", "Path": "Path",
}


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug[:40] or "unknown_org"


def load_csv(name):
    with open(RAW / f"{name}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def strip_number_suffix(name: str) -> str:
    """Synthea appends digits to names for uniqueness, e.g. 'Guillermo498' -> 'Guillermo'."""
    return re.sub(r"\d+$", "", name)


def abbreviate_address(addr: str) -> str:
    for full, abbr in STREET_ABBREV.items():
        addr = re.sub(rf"\b{full}\b", abbr, addr)
    return addr


def corrupt_dob(dob: str, fmt: str, typo: bool) -> str:
    y, m, d = dob.split("-")
    if typo:
        # Transpose day digits (03 -> 30) to simulate a data-entry slip. Only apply
        # if the transposed value is still a real day-of-month for this exact
        # year/month (e.g. "23" -> "32" is never valid; "29" -> "92" is invalid, but
        # also "30" -> "03" would be fine in April while "31" -> "13" needs a 31-day
        # month) -- a data-entry error still produces a valid-looking date, it just
        # points at the wrong day.
        reversed_d = d[::-1]
        max_day = calendar.monthrange(int(y), int(m))[1]
        if reversed_d != d and 1 <= int(reversed_d) <= max_day:
            d = reversed_d
    if fmt == "iso":
        return f"{y}-{m}-{d}"
    if fmt == "us_slash":
        return f"{m}/{d}/{y}"
    if fmt == "us_dash":
        return f"{m}-{d}-{y}"
    if fmt == "written":
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        return f"{months[int(m) - 1]} {int(d)}, {y}"
    return dob


def build_hospital_identity(patient: dict, rng: random.Random) -> dict:
    """Produce one hospital's corrupted view of a patient's identity fields."""
    first = strip_number_suffix(patient["FIRST"])
    middle = strip_number_suffix(patient["MIDDLE"]) if patient["MIDDLE"] else ""
    last = strip_number_suffix(patient["LAST"])
    maiden = strip_number_suffix(patient["MAIDEN"]) if patient["MAIDEN"] else ""

    name_style = rng.choice(["full", "initial_first", "no_middle", "no_middle", "maiden_swap"])
    if name_style == "maiden_swap" and not (maiden and patient["GENDER"] == "F" and patient["MARITAL"] == "M"):
        name_style = "no_middle"

    if name_style == "full":
        first_out, middle_out, last_out = first, middle, last
    elif name_style == "initial_first":
        first_out, middle_out, last_out = f"{first[0]}.", "", last
    elif name_style == "no_middle":
        first_out, middle_out, last_out = first, "", last
    elif name_style == "maiden_swap":
        first_out, middle_out, last_out = first, middle, maiden
    else:
        first_out, middle_out, last_out = first, middle, last

    dob_fmt = rng.choice(["iso", "us_slash", "us_dash", "written"])
    dob_typo = rng.random() < 0.08
    dob_out = corrupt_dob(patient["BIRTHDATE"], dob_fmt, dob_typo)

    addr_style = rng.choice(["full", "abbreviated", "city_state_only"])
    if addr_style == "full":
        address_out = patient["ADDRESS"]
    elif addr_style == "abbreviated":
        address_out = abbreviate_address(patient["ADDRESS"])
    else:
        address_out = ""

    return {
        "FIRST": first_out,
        "MIDDLE": middle_out,
        "LAST": last_out,
        "DOB": dob_out,
        "DOB_HAD_TYPO": dob_typo,
        "GENDER": patient["GENDER"],
        "ADDRESS": address_out,
        "CITY": patient["CITY"],
        "STATE": patient["STATE"],
        "ZIP": patient["ZIP"] if addr_style != "city_state_only" else "",
        "SSN": patient["SSN"] if rng.random() > 0.4 else "",
    }


def main():
    patients = {p["Id"]: p for p in load_csv("patients")}
    encounters = load_csv("encounters")
    conditions = load_csv("conditions")
    medications = load_csv("medications")
    allergies = load_csv("allergies")
    organizations = {o["Id"]: o for o in load_csv("organizations")}

    # Group encounters by (patient, consolidated region) rather than (patient, raw org),
    # so a patient's visits to different clinics within the same regional network land
    # in one chart -- see ZIP_PREFIX_TO_REGION above.
    org_region = {org_id: region_for_org(org) for org_id, org in organizations.items()}
    enc_by_patient_region = defaultdict(list)
    for e in encounters:
        region = org_region.get(e["ORGANIZATION"], DEFAULT_REGION)
        enc_by_patient_region[(e["PATIENT"], region)].append(e)

    conditions_by_enc = defaultdict(list)
    for c in conditions:
        conditions_by_enc[c["ENCOUNTER"]].append(c)
    medications_by_enc = defaultdict(list)
    for m in medications:
        medications_by_enc[m["ENCOUNTER"]].append(m)

    # Allergies aren't tied to an encounter column consistently across Synthea versions,
    # so attribute them to whichever (patient, org) has an encounter on the same START date.
    allergies_by_patient = defaultdict(list)
    for a in allergies:
        allergies_by_patient[a["PATIENT"]].append(a)

    rng = random.Random(42)

    crosswalk_rows = []  # real_patient_id, hospital_org_id, hospital_slug, local_patient_id
    hospital_patients_out = defaultdict(list)
    hospital_encounters_out = defaultdict(list)
    hospital_conditions_out = defaultdict(list)
    hospital_medications_out = defaultdict(list)
    hospital_allergies_out = defaultdict(list)

    for (patient_id, region_name), enc_list in enc_by_patient_region.items():
        patient = patients[patient_id]
        slug = slugify(region_name)
        local_patient_id = f"{slug[:6].upper()}-{rng.randrange(16**6):06x}"
        contributing_orgs = sorted({organizations[e["ORGANIZATION"]]["NAME"] for e in enc_list})

        identity = build_hospital_identity(patient, rng)
        hospital_patients_out[slug].append({
            "LOCAL_PATIENT_ID": local_patient_id,
            **identity,
        })
        crosswalk_rows.append({
            "REAL_PATIENT_ID": patient_id,
            "HOSPITAL_SLUG": slug,
            "HOSPITAL_NAME": region_name,
            "LOCAL_PATIENT_ID": local_patient_id,
            "CONTRIBUTING_ORGS": "; ".join(contributing_orgs),
        })

        for e in sorted(enc_list, key=lambda r: r["START"]):
            hospital_encounters_out[slug].append({
                "LOCAL_ENCOUNTER_ID": e["Id"],
                "LOCAL_PATIENT_ID": local_patient_id,
                "START": e["START"],
                "STOP": e["STOP"],
                "ENCOUNTERCLASS": e["ENCOUNTERCLASS"],
                "DESCRIPTION": e["DESCRIPTION"],
                "REASONDESCRIPTION": e["REASONDESCRIPTION"],
                "SEEN_AT": organizations[e["ORGANIZATION"]]["NAME"],
            })
            for c in conditions_by_enc.get(e["Id"], []):
                hospital_conditions_out[slug].append({
                    "LOCAL_ENCOUNTER_ID": e["Id"],
                    "LOCAL_PATIENT_ID": local_patient_id,
                    "START": c["START"],
                    "STOP": c["STOP"],
                    "DESCRIPTION": c["DESCRIPTION"],
                })
            for m in medications_by_enc.get(e["Id"], []):
                hospital_medications_out[slug].append({
                    "LOCAL_ENCOUNTER_ID": e["Id"],
                    "LOCAL_PATIENT_ID": local_patient_id,
                    "START": m["START"],
                    "STOP": m["STOP"],
                    "DESCRIPTION": m["DESCRIPTION"],
                    "REASONDESCRIPTION": m["REASONDESCRIPTION"],
                })

        enc_dates = {e["START"][:10] for e in enc_list}
        for a in allergies_by_patient.get(patient_id, []):
            if a["START"] in enc_dates or not a["START"]:
                hospital_allergies_out[slug].append({
                    "LOCAL_PATIENT_ID": local_patient_id,
                    "START": a["START"],
                    "DESCRIPTION": a["DESCRIPTION"],
                    "REACTION": a.get("DESCRIPTION1", ""),
                    "SEVERITY": a.get("SEVERITY1", ""),
                })

    # Write per-hospital CSVs
    def write_rows(path: Path, rows: list):
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    all_slugs = set(hospital_patients_out) | set(hospital_encounters_out)
    for slug in all_slugs:
        base = OUT_HOSPITALS / slug
        write_rows(base / "patients.csv", hospital_patients_out[slug])
        write_rows(base / "encounters.csv", hospital_encounters_out[slug])
        write_rows(base / "conditions.csv", hospital_conditions_out[slug])
        write_rows(base / "medications.csv", hospital_medications_out[slug])
        write_rows(base / "allergies.csv", hospital_allergies_out[slug])

    write_rows(OUT_GT / "patient_crosswalk.csv", crosswalk_rows)

    print(f"Wrote {len(all_slugs)} hospital silos to {OUT_HOSPITALS}")
    print(f"Wrote ground-truth crosswalk ({len(crosswalk_rows)} rows) to {OUT_GT / 'patient_crosswalk.csv'}")
    print(f"Real patients: {len(patients)}, shattered into avg "
          f"{len(crosswalk_rows) / len(patients):.2f} hospital records each")


if __name__ == "__main__":
    main()

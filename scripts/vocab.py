"""
Shared vocabulary used by both note generation (02) and entity normalization (05).
Kept in one place so the two stages can't drift out of sync -- normalization needs to
reverse exactly the same synonym mapping that note generation used to inject
variation in the first place.
"""

import re

# Clinical term -> alternate phrasings (index 0 is always the canonical clinical term).
SYNONYMS = {
    "Myocardial infarction": ["myocardial infarction", "a heart attack"],
    "Acute ST segment elevation myocardial infarction": ["an acute ST-elevation myocardial infarction (STEMI)", "a severe heart attack"],
    "Acute non-ST segment elevation myocardial infarction": ["a non-ST-elevation myocardial infarction (NSTEMI)", "a mild heart attack"],
    "History of myocardial infarction": ["a prior myocardial infarction", "a previous heart attack"],
    "Ischemic heart disease": ["ischemic heart disease", "coronary artery disease"],
    "Essential hypertension": ["essential hypertension", "high blood pressure"],
    "Diabetes mellitus type 2": ["type 2 diabetes mellitus", "type 2 diabetes"],
    "Prediabetes": ["prediabetes", "borderline blood sugar"],
    "Viral sinusitis": ["viral sinusitis", "a sinus infection"],
    "Chronic sinusitis": ["chronic sinusitis", "long-standing sinus infection"],
    "Acute viral pharyngitis": ["acute viral pharyngitis", "a sore throat"],
    "Acute bronchitis": ["acute bronchitis", "a chest cold"],
    "Otitis media": ["otitis media", "a middle-ear infection"],
    "Gingivitis": ["gingivitis", "gum inflammation"],
    "Gingival disease": ["gingival disease", "gum disease"],
    "Anemia": ["anemia", "a low blood count"],
    "Asthma": ["asthma", "reactive airway disease"],
    "Childhood asthma": ["childhood asthma", "pediatric asthma"],
    "Severe anxiety (panic)": ["severe anxiety with panic features", "panic attacks"],
    "Body mass index 30+ - obesity": ["obesity (BMI 30+)", "significant weight gain"],
    "Chronic pain": ["chronic pain", "ongoing pain"],
}

# Synthea's conditions.csv mixes real clinical diagnoses in with social-determinants-
# of-health / administrative findings (employment status, education level, housing,
# screening reminders, care-coordination events...). A biomedical NER model trained on
# clinical text correctly does NOT tag "full-time employment" as a Disease_disorder --
# that's the model being right, not wrong. Keeping these in the DIAGNOSIS gold set
# would make the extraction eval fail for a reason that has nothing to do with
# extraction quality, so they get their own non-graded "social history" sentence
# instead. Matched against the cleaned (tag-stripped) description, lowercased.
ADMIN_SDOH_DESCRIPTIONS = {
    "full-time employment", "part-time employment", "not in labor force", "unemployed",
    "social isolation", "limited social contact", "received higher education",
    "educated to high school level", "only received primary school education",
    "medication review due", "victim of intimate partner abuse",
    "reports of violence in the environment", "risk activity involvement",
    "homeless", "housing unsatisfactory", "lack of access to transportation",
    "has a criminal record", "refugee (person)", "serving in military service",
    "stress", "transport problem", "transition from acute care to home-health care",
    "unhealthy alcohol drinking behavior", "misuses drugs",
    "patient referral for dental care", "patient transfer to skilled nursing facility",
    "contraception care (regime/therapy)", "screening for malignant neoplasm of breast",
    "screening for malignant neoplasm of colon", "sterilization requested",
}

STRIP_TAG_RE = re.compile(r"\s*\((disorder|finding|situation|procedure|substance)\)\s*$")


def clean(desc: str) -> str:
    return STRIP_TAG_RE.sub("", desc).strip()


def is_admin_sdoh(raw_description: str) -> bool:
    return clean(raw_description).lower() in ADMIN_SDOH_DESCRIPTIONS

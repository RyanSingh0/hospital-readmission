"""
Cohort construction and feature engineering for the Diabetes 130-US Hospitals dataset.

Three decisions in here are worth flagging, because they account for most of the gap between
this pipeline and the published ones it is compared against.

First, encounters ending in death or a transfer to hospice are removed. Those patients cannot
be readmitted — discharge disposition 11 carries a 30-day readmission rate of exactly zero
across 1,642 encounters — so a model that learns to recognise them is not stratifying risk, it
is recognising an outcome that was already fixed at discharge. Retaining them inflates AUC by
roughly 0.007.

Second, the three diagnosis fields are mapped to ICD-9 chapters rather than collapsed into a
handful of binary flags. Chapters keep more information, and the ablation in experiment 04
shows the difference is worth about 0.002 AUC.

Third, nothing here imputes a category. A question mark in `payer_code` or `medical_specialty`
means the field was never recorded, and whether a field was recorded is itself predictive, so
those values become an explicit "Unknown" level instead of being filled in.
"""

import os

import numpy as np
import pandas as pd

# Discharge dispositions that mean the patient died or entered hospice care. These encounters
# are dropped, because a readmission was never possible for them.
TERMINAL_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

# Drug columns that carry a single value for essentially the whole cohort. They cost memory,
# add nothing, and clutter the SHAP plots.
CONSTANT_DRUG_COLUMNS = [
    "acetohexamide",
    "troglitazone",
    "examide",
    "citoglipton",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Drug columns that do vary, and that the medication-dynamics features are built from.
DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone",
    "acarbose", "miglitol", "tolazamide", "insulin", "glyburide-metformin",
    "glipizide-metformin",
]

NUMERIC_FEATURES = [
    "age_mid", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "prior_visits", "had_inpatient", "had_emergency",
    "n_drug_changes", "n_drugs_on", "meds_per_day", "procs_per_day", "labs_per_day",
    "a1c_tested", "glu_tested", "insulin_on",
]

CATEGORICAL_FEATURES = [
    "race", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty", "payer_code", "max_glu_serum",
    "A1Cresult", "change", "diabetesMed", "diag_1_grp", "diag_2_grp", "diag_3_grp",
    "metformin", "insulin", "glipizide", "glyburide", "pioglitazone",
    "rosiglitazone", "glimepiride",
]

# Kept as a module-level name so the ablation scripts can temporarily empty it and reproduce
# the published cohort, which retains the terminal encounters.
DEAD = TERMINAL_DISPOSITIONS


def icd9_chapter(code):
    """Map a raw ICD-9 code onto its clinical chapter.

    The dataset holds 717 distinct codes across the three diagnosis fields. One-hot encoding
    them directly produces a very sparse matrix with little signal per column, so they are
    folded into the eighteen standard chapters instead. V-codes and E-codes get their own
    buckets rather than being forced into a numeric range they do not belong to.
    """
    if pd.isna(code) or code == "?":
        return "Missing"

    text = str(code)
    if text.startswith("V"):
        return "Supplementary"
    if text.startswith("E"):
        return "External"

    try:
        value = float(text)
    except ValueError:
        return "Other"

    if 250 <= value < 251:
        return "Diabetes"
    if 390 <= value < 460 or int(value) == 785:
        return "Circulatory"
    if 460 <= value < 520 or int(value) == 786:
        return "Respiratory"
    if 520 <= value < 580 or int(value) == 787:
        return "Digestive"
    if 580 <= value < 630 or int(value) == 788:
        return "Genitourinary"
    if 800 <= value < 1000:
        return "Injury"
    if 710 <= value < 740:
        return "Musculoskeletal"
    if 140 <= value < 240:
        return "Neoplasms"
    if 240 <= value < 280:
        return "Endocrine/Metabolic"
    if 280 <= value < 290:
        return "Blood"
    if 290 <= value < 320:
        return "Mental"
    if 320 <= value < 390:
        return "Nervous"
    if 630 <= value < 680:
        return "Pregnancy"
    if 680 <= value < 710:
        return "Skin"
    if 780 <= value < 800:
        return "Symptoms"
    return "Other"


def load(path=None, verbose=True):
    """Build the modelling cohort from the raw UCI export.

    Returns one row per encounter, with the binary outcome in `y`, the patient identifier
    preserved in `patient_nbr` so that cross-validation can group on it, and every engineered
    feature listed in NUMERIC_FEATURES and CATEGORICAL_FEATURES attached.
    """
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "data", "diabetic_data.csv")

    frame = pd.read_csv(path, low_memory=False)
    before = len(frame)

    frame = frame[~frame.discharge_disposition_id.isin(DEAD)].copy()
    after_terminal = len(frame)

    # Three encounters carry an unknown gender. They are dropped rather than imputed, because
    # three rows out of a hundred thousand are not worth a special case downstream.
    frame = frame[frame.gender != "Unknown/Invalid"].copy()

    if verbose:
        print(
            f"rows {before:,} -> {after_terminal:,} after removing terminal encounters "
            f"({before - after_terminal:,}) -> {len(frame):,} after removing unknown gender"
        )

    frame["y"] = (frame.readmitted == "<30").astype(int)

    # Age arrives as bracket labels such as "[60-70)". The midpoint keeps the ordinal
    # structure while letting the model treat age as continuous.
    frame["age_mid"] = frame.age.map(lambda s: int(s[1:s.index("-")]) + 5)

    for column in ["diag_1", "diag_2", "diag_3"]:
        frame[column + "_grp"] = frame[column].map(icd9_chapter)

    # Prior utilisation is consistently the strongest clinical signal in the readmission
    # literature, so it gets both a total and a pair of has-any indicators.
    frame["prior_visits"] = (
        frame.number_outpatient + frame.number_emergency + frame.number_inpatient
    )
    frame["had_inpatient"] = (frame.number_inpatient > 0).astype(int)
    frame["had_emergency"] = (frame.number_emergency > 0).astype(int)

    # Medication dynamics — how much the regimen moved during the stay, and how broad it is.
    frame["n_drug_changes"] = sum(
        frame[column].isin(["Up", "Down"]).astype(int) for column in DRUG_COLUMNS
    )
    frame["n_drugs_on"] = sum((frame[column] != "No").astype(int) for column in DRUG_COLUMNS)

    # Intensity ratios separate a long quiet admission from a short busy one, which the raw
    # counts cannot do on their own.
    frame["meds_per_day"] = frame.num_medications / frame.time_in_hospital
    frame["procs_per_day"] = frame.num_procedures / frame.time_in_hospital
    frame["labs_per_day"] = frame.num_lab_procedures / frame.time_in_hospital

    # Whether a test was ordered carries information even when the result does not, and both
    # A1C and glucose are missing for the large majority of encounters.
    frame["a1c_tested"] = (frame.A1Cresult != "None").astype(int)
    frame["glu_tested"] = (frame.max_glu_serum != "None").astype(int)
    frame["insulin_on"] = (frame.insulin != "No").astype(int)

    for column in ["race", "payer_code", "medical_specialty"]:
        frame[column] = frame[column].replace("?", "Unknown")

    # These three are identifiers, not magnitudes, so they are carried as strings and handled
    # as categoricals rather than being treated as ordered numbers.
    for column in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        frame[column] = frame[column].astype(str)

    return frame.drop(columns=["weight"] + CONSTANT_DRUG_COLUMNS)


def add_trajectory_features(frame):
    """Attach the patient-history features, computed strictly from earlier encounters.

    Every column built here uses `shift(1)` before any expanding aggregation, so an encounter
    only ever sees the encounters that preceded it for the same patient. That ordering comes
    from `encounter_id`, which is assigned chronologically in the source data.

    Note that no prior-outcome feature is included. An earlier version did add one — the count
    of the patient's previous 30-day readmissions — and it turned out to contribute nothing
    once the other trajectory columns were present, so it was dropped rather than defended.
    """
    frame = frame.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
    grouped = frame.groupby("patient_nbr")

    frame["enc_seq"] = grouped.cumcount()
    frame["is_repeat"] = (frame.enc_seq > 0).astype(int)

    rollups = [
        ("prior_los_mean", "time_in_hospital", "mean"),
        ("prior_los_max", "time_in_hospital", "max"),
        ("prior_meds_mean", "num_medications", "mean"),
        ("prior_dx_mean", "number_diagnoses", "mean"),
        ("prior_inpatient_max", "number_inpatient", "max"),
    ]
    for name, source, how in rollups:
        frame[name] = (
            grouped[source]
            .apply(lambda s: getattr(s.shift(1).expanding(), how)())
            .reset_index(level=0, drop=True)
            .fillna(-1)
        )

    # Deltas against the patient's own history, which is what a clinician would actually
    # notice: this stay is three days longer than the last two were.
    frame["los_vs_prior"] = np.where(
        frame.prior_los_mean > 0, frame.time_in_hospital - frame.prior_los_mean, 0
    )
    frame["meds_vs_prior"] = np.where(
        frame.prior_meds_mean > 0, frame.num_medications - frame.prior_meds_mean, 0
    )

    # The dataset carries no dates, so the gap between consecutive encounter identifiers is
    # the only available proxy for elapsed time between visits. It is crude, and it is
    # reported as such in the manuscript.
    frame["enc_id_gap"] = grouped.encounter_id.diff().fillna(-1)

    return frame


TRAJECTORY_FEATURES = [
    "enc_seq", "is_repeat", "prior_los_mean", "prior_los_max", "prior_meds_mean",
    "prior_dx_mean", "prior_inpatient_max", "los_vs_prior", "meds_vs_prior", "enc_id_gap",
]

# Aliases kept short because the experiment scripts import them constantly.
NUM = NUMERIC_FEATURES
CAT = CATEGORICAL_FEATURES


if __name__ == "__main__":
    cohort = add_trajectory_features(load())
    print(f"\nfinal cohort: {len(cohort):,} encounters, {cohort.patient_nbr.nunique():,} patients")
    print(f"30-day readmission rate: {cohort.y.mean():.4f}")
    print(f"repeat encounters: {cohort.is_repeat.sum():,} ({cohort.is_repeat.mean():.1%})")

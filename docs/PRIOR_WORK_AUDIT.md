# Audit of Prior Work + Honest Publication Assessment

**Date:** 16 Sep 2026. Every number here is either arithmetic on published values (reproducible with
a calculator) or output from the scripts in this folder.

---

## Part 1 — Is `prac 01` faking its results?

**The claim does not match the notebook's own output.**

| | "Key Insights" markdown cell | What the code actually printed |
|---|---|---|
| AUC-ROC | **0.82–0.85** ("strong discrimination") | **0.642** |
| Recall | **0.75** | **0.60** (class 1, from `classification_report`) |
| Top predictors | LOS, CHF, ICU days | XGBoost F-score ranks **age 389**, los_days 262, icu_days 135, chf 59 |

Both numbers in the narrative cell are materially higher than the printed `AUC-ROC: 0.642` and the
0.60 recall in the classification report directly above it, and the stated top-predictor ordering
contradicts the feature-importance chart in the same notebook.

I would not call this fraud — it reads like a narrative cell written from expectation and never
reconciled against the run. But as a benchmark it is worthless, and there is a second, larger
problem: **that notebook never loads the dataset it is attached to.** It calls
`generate_patient_data(n=30000)` and builds its own synthetic frame with different columns
(`chf`, `icu_days`, `los_days`) and a 29.6% readmission rate. Its 0.642 describes a different
problem entirely.

**Verdict: not a benchmark. Do not cite it, do not compare to it.**

---

## Part 2 — Audit of Salim & Ibrahim (*Healthcare* 2026, 14, 1185)

This is a real, competently executed paper, and its limitations section is unusually candid. Three
of the four issues below are ones **the authors themselves disclose**. One is not.

### 2.1 Tables 4 and 5 contradict each other at the same threshold

Both tables describe "the calibrated XGBoost model" at a 10% risk threshold:

| | Sensitivity | Specificity | PPV | High risk per 1000 |
|---|---|---|---|---|
| Table 4 | 0.723 | 0.530 | 0.162 | 497.9 |
| Table 5 (row 0.10) | **0.881** | 0.524 | **0.189** | **521.0** |

Each table is internally consistent — I verified PPV, NPV and flagged-count against the stated
prevalence of 11.16% and they reconcile to within rounding in both. So they are not typos; they are
**two different evaluations of two different models**, presented as one.

### 2.2 Table 5's operating points require a far stronger model than the paper reports

Converting each (sensitivity, specificity) pair to the binormal-equivalent AUC it implies:

| Source | Implied AUC |
|---|---|
| Table 4 @ 10% | **0.681** — consistent with the reported 0.688 |
| Table 5 @ 0.10 | 0.810 |
| Table 5 @ 0.15 | 0.795 |
| Table 5 @ 0.20 | 0.803 |
| Table 5 @ 0.25 | 0.816 |
| Table 5 @ 0.30 | 0.824 |

**Every Table 5 row implies AUC ≈ 0.80, against a reported 0.664 (nested) / 0.688 (full-dataset).**
Table 5's last row claims PPV 0.886 — i.e. 89% of flagged patients readmit, against an 11.16% base
rate, a lift of 7.9×. No model with AUC 0.688 can do that.

Empirical confirmation: I trained an XGBoost with the paper's stated configuration
(`scale_pos_weight` = 7.96, Platt scaling) on the paper's cohort.

| Evaluation | AUC |
|---|---|
| In-sample (train on all, evaluate on all) | **0.8775** |
| Out-of-fold (5-fold) | **0.6594** — closely reproduces their 0.664 |

The in-sample figure lands in the same 0.82–0.88 band Table 5 implies. The paper does state, for
decision curve analysis specifically, that it "was computed using full-dataset predictions rather
than strictly out-of-sample estimates." Table 5 carries no such caveat, and it is the table a
hospital would read to choose an operating threshold.

Two rows also fail their own arithmetic: at threshold 0.25 the stated PPV is 0.707 but the other
three columns imply 0.693; at 0.30 the stated PPV is 0.886 against an implied 0.827.

### 2.3 The model earns discrimination by identifying patients who died — not disclosed

The paper's Section 3.1.4 retains **"Discharge group: Home, Transfer, Expired, Other."** Their own
Figure 3 SHAP plot shows **"Discharge: Expired"** as the single largest-magnitude contributor,
with SHAP values near −6.

In the raw data:

| Discharge disposition | n | 30-day readmission rate |
|---|---|---|
| 11 — Expired | 1,642 | **0.00%** |
| 13 — Hospice / home | 399 | 4.76% |
| 14 — Hospice / medical facility | 372 | 6.45% |
| All expired + hospice | 2,423 | 1.77% |
| Everyone else | 99,340 | 11.39% |

Deceased patients cannot be readmitted. A model that identifies them is not stratifying risk; it is
recognising an outcome already determined at discharge. Measured effect on our own pipeline:

| Cohort | AUC |
|---|---|
| Expired/hospice retained (paper's cohort) | 0.6895 |
| Expired/hospice removed (clinically correct) | 0.6824 |

**≈0.007 AUC, roughly a third of the margin by which this paper leads the prior literature.**
This is the one issue the authors do not flag anywhere.

### 2.4 Three limitations the authors do disclose, which we close

> "The cross-validation strategy did not enforce patient-level splits… Future work should consider
> GroupKFold strategies." — §5.4

We did it. **Row-level split 0.6787 vs patient-grouped 0.6773 — a difference of +0.0014, p = 0.45.**
Not significant. With `patient_nbr` excluded from the design matrix, no feature identifies an
individual, so repeat encounters behave like ordinary correlated rows. This is a **negative result
that resolves their open question**, and it is worth reporting precisely because the field assumes
the opposite.

> "Decision curve analysis was computed using full-dataset predictions rather than strictly
> out-of-sample estimates." — §5.4

Ours is computed entirely out-of-fold.

> "No subgroup analysis was conducted to evaluate model performance across age groups, genders, or
> racial/ethnic categories." — §5.4

We ran it. See §3.3 below.

---

## Part 3 — Our results

### 3.1 Head-to-head

| Metric | Salim & Ibrahim | This work | Note |
|---|---|---|---|
| ROC AUC (nested CV) | 0.664 | **0.683** | clean cohort, patient-grouped |
| ROC AUC (comparable cohort) | 0.664 | **0.690** | expired retained, 5-fold grouped |
| PR-AUC | 0.215 | **0.239** | |
| Brier | 0.094 | 0.095 | essentially tied |
| Patient-level splits | no | **yes** | |
| Out-of-sample DCA | no | **yes** | |
| Subgroup analysis | no | **yes** | |
| Expired/hospice excluded | no | **yes** | |

Baseline comparisons under our grouped split: logistic regression 0.665, `number_inpatient` alone
with no model at all 0.607.

### 3.2 Calibration — post-hoc calibration was not needed

Isotonic calibration made Brier slightly **worse** (0.0953 → 0.0957) and AUC slightly worse
(0.6826 → 0.6814). LightGBM optimising log-loss is already calibrated; Platt/isotanic on top adds
variance. Out-of-fold reliability across deciles: predicted 0.042 / observed 0.037 at the bottom,
predicted 0.244 / observed 0.278 at the top — mild under-prediction in the top decile only.

### 3.3 Subgroup analysis — the finding that matters clinically

| Subgroup | n | Prevalence | AUC | Brier | Obs/Pred | Top-decile lift |
|---|---|---|---|---|---|---|
| **Overall** | 99,340 | 0.114 | 0.6813 | 0.0957 | 1.015 | 2.44× |
| Age < 45 | 15,870 | 0.109 | **0.7329** | 0.0889 | 1.039 | 3.04× |
| Age 45–65 | 39,118 | 0.106 | 0.6906 | 0.0898 | 0.998 | 2.57× |
| Age 65–80 | 25,329 | 0.121 | 0.6613 | 0.1015 | 1.020 | 2.23× |
| **Age 80+** | 19,023 | **0.125** | **0.6312** | 0.1058 | 1.024 | 2.09× |
| African American | 18,772 | 0.114 | 0.6760 | 0.0966 | 1.009 | 2.30× |
| Caucasian | 74,220 | 0.115 | 0.6804 | 0.0967 | 1.020 | 2.46× |
| Female | 53,454 | 0.115 | 0.6828 | 0.0962 | 1.014 | 2.45× |
| Male | 45,886 | 0.113 | 0.6795 | 0.0951 | 1.017 | 2.44× |

**Discrimination falls monotonically with age — 0.733 at under 45 down to 0.631 at 80+, a gap of
0.102 — while prevalence rises.** The model performs worst on the group at highest risk, which is
also the group a readmission-reduction programme most wants to target.

Race and sex show no meaningful disparity (African American 0.676 vs Caucasian 0.680; female 0.683
vs male 0.680), and calibration holds across every subgroup (observed/predicted 0.93–1.04). So this
is a **discrimination** disparity, not a calibration one — which has a different operational
remedy: age-stratified thresholds rather than recalibration.

### 3.4 Clinical utility, out-of-sample

Top 5% of risk: 34.1% PPV, 3.00× lift. Top 10%: 27.8% PPV, 2.44× lift, 24.4% of readmissions
captured. Decision curve net benefit exceeds both treat-all and treat-none at every threshold from
0.05 to 0.30.

---

## Part 4 — Publication assessment

### What you actually have

A defensible contribution, but **not** "we beat the state of the art." The AUC delta (0.664 → 0.683)
is modest and partly attributable to feature engineering rather than method. The real contributions
are methodological:

1. Quantifying that patient-level splitting does **not** matter on this dataset (negative result,
   answers an open question the prior authors posed).
2. Demonstrating that retaining deceased patients inflates AUC by ~0.007, with the mechanism shown.
3. The first subgroup analysis on this benchmark, showing a 0.10 AUC age gradient.
4. Out-of-sample decision curve and capacity-constrained targeting economics.
5. Showing post-hoc calibration is unnecessary for a log-loss-trained GBM here.

### Recommended route

**Do not write a takedown.** A paper whose thesis is "this recent paper's tables are inconsistent"
is a Comment, it invites a Reply, and for a job-seeking first author the downside risk badly
outweighs the upside. If you are wrong about any detail, it follows you.

Write the **positive** version instead:

> *"Evaluation pitfalls in administrative-data readmission prediction: outcome-determined features,
> patient-level splitting, and subgroup performance on Diabetes 130-US Hospitals"*

Cite Salim & Ibrahim as **one of several** benchmarks in a comparison table alongside Emi-Johnson
(0.667), Shang (0.661), and Liu. Report the expired-patient issue as a **general pitfall** of the
dataset — it affects most published work on it, not just theirs — and let the reader connect it.
State the Table 4/5 discrepancy, if at all, in a single neutral sentence in a footnote, or leave it
out of the paper and keep it in your repo as a reproducibility note. That framing is stronger
science, safer for you, and far more likely to survive review.

**Sequence:**

1. **medRxiv preprint first.** Free, immediate, citable, gives you a live link for the resume this
   month rather than in six. Post the repo alongside it.
2. **Then submit.** Realistic targets, in order: *JMIR Medical Informatics*, *BMC Medical Informatics
   and Decision Making*, *Diagnostics* (MDPI), or *Healthcare* (MDPI) itself. Note that MDPI APCs
   run roughly 2,000–2,700 CHF — budget for a waiver request, as you obtained for *Risks*.
3. **Conference alternative** if you want a faster, cheaper credit: IEEE ICHI or an AMIA workshop.

### What still needs doing before submission

- [ ] External validation, or an explicit statement that there is none. This is the single biggest
      reviewer objection and the paper you are building on has the same gap.
- [ ] Bootstrap confidence intervals on every headline metric (they did 1,000 resamples; match it).
- [ ] DeLong test for the AUC difference vs the logistic baseline — "0.683 vs 0.665" needs a p-value.
- [ ] Nested CV for the comparable-cohort number too, so both headline figures use the same protocol.
- [ ] SHAP with stability analysis across bootstrap resamples, to match their Section 3.2.7.
- [ ] TRIPOD+AI reporting checklist — most clinical-ML venues now require it.
- [ ] A clear statement that the data predates HRRP (2012) and is a historical artifact.

### Honest expected outcome

With the items above completed, this is a **plausible accept at JMIR Medical Informatics or BMC MIDM,
and a likely accept at an MDPI journal**. It is not a high-impact-factor paper. Its real value to you
is (a) a second first-author credit, (b) a genuine methodological story you can tell in an interview,
and (c) a repo that demonstrates you audit your own work. Those are worth more to your job search
than the venue's impact factor.

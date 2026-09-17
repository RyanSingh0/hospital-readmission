# TRIPOD+AI Reporting Checklist

Completed against Collins and colleagues (2024). Each item records where the manuscript
addresses it, or states plainly that it does not. Items marked **OUTSTANDING** must be resolved
before submission.

| # | Item | Status | Where, or what is missing |
|---|---|---|---|
| **Title and abstract** | | | |
| 1 | Identify the study as developing a prediction model, state the target population and outcome | Done | Title; Abstract |
| 2 | Structured abstract covering objectives, methods, results, conclusions | Done | Abstract |
| **Introduction** | | | |
| 3a | Background and rationale, referencing existing models | Done | §1, references 1–6 |
| 3b | Objectives, including whether development, validation, or both | Done | §1. Development and internal validation only; no external validation |
| **Methods — data** | | | |
| 4a | Data source, and separately for development and validation | Done | §2.1 |
| 4b | Study dates | Done | §2.1, 1999–2008 |
| 5a | Eligibility criteria | Done | §2.1 |
| 5b | Details of treatments received, if relevant | Partial | Medication fields described in §2.2; treatment protocols are not recorded in the source |
| 6a | Outcome definition and how it was measured | Done | §2.1 |
| 6b | Whether outcome assessors were blinded to predictors | Not applicable | Retrospective administrative extract |
| 7a | Predictors, with definitions and timing of measurement | Done | §2.2 |
| 7b | Whether predictor assessors were blinded to outcome | Not applicable | As above |
| 8 | Sample size, and how it was arrived at | Done | §2.1. Full available cohort; no a priori calculation, which is stated |
| 9 | Missing data handling | Done | §2.2. Explicit Unknown category, with justification and references 7–8 |
| **Methods — analysis** | | | |
| 10a | How predictors were handled in the analysis | Done | §2.2, §2.3 |
| 10b | Model type, building procedure, and internal validation | Done | §2.3 |
| 10c | How predictions were calculated | Done | §2.3 |
| 10d | Performance measures | Done | §2.4 |
| 10e | Model updating, if any | Done | §3.7. No post-hoc calibration applied, with justification |
| 11 | Risk groups, if created | Done | §3.9, capacity-constrained targeting |
| 12 | Differences between development and validation data | Not applicable | No external validation |
| **Methods — AI-specific** | | | |
| A1 | Rationale for the chosen algorithm class | Done | §2.3, §3.6 |
| A2 | Handling of class imbalance | Done | §3.5, with a negative result |
| A3 | Hyperparameter search space and selection procedure | Partial | §2.3 describes the nested design. **OUTSTANDING** — the exact grid belongs in a supplement |
| A4 | Software, versions, and random seeds | Partial | Versions in `requirements.txt`; seeds fixed in code. **OUTSTANDING** — consolidate into a methods paragraph |
| A5 | Measures against data leakage | Done | §2.2 shift-before-aggregate; §2.3 patient-grouped folds; §3.3 direct test |
| A6 | Fairness assessment across subgroups | Done | §3.8 |
| **Results** | | | |
| 13a | Flow of participants, with a diagram | Partial | §2.1 gives the counts. **OUTSTANDING** — a CONSORT-style flow diagram is expected |
| 13b | Characteristics of participants, including predictor distributions | **OUTSTANDING** | A Table 1 of baseline characteristics is required and not yet written |
| 13c | Number of participants and events in the analysis | Done | §2.1 |
| 14a | Number of participants and events in each analysis | Done | Tables 1–8 |
| 14b | Unadjusted association between candidate predictors and outcome | **OUTSTANDING** | Univariate associations should appear in a supplement |
| 15a | Full model specification | Partial | Hyperparameters are in code. **OUTSTANDING** — state them in the manuscript |
| 15b | Explanation of how to use the model | Done | §3.9 |
| 16 | Performance measures with confidence intervals | Done | Table 1; bootstrap intervals throughout |
| 17 | Model updating results | Not applicable | |
| **Discussion** | | | |
| 18 | Limitations | Done | §4.4 |
| 19a | Interpretation for validation studies | Not applicable | |
| 19b | Overall interpretation, with reference to objectives and prior evidence | Done | §4, §4.1 |
| 20 | Potential clinical use and implications | Done | §4.2, §4.3 |
| **Other** | | | |
| 21 | Supplementary information availability | Done | Repository link |
| 22 | Funding | Done | None |
| **AI-specific reporting** | | | |
| A7 | Explainability or interpretability methods | **OUTSTANDING** | SHAP analysis with bootstrap stability should be added to match the comparator study |
| A8 | Code availability | Done | Full repository |
| A9 | Data availability | Done | Both datasets public; download instructions in `data/README.md` |
| A10 | Declaration of AI assistance in preparing the work | Done | Acknowledgement of tooling |

---

## Outstanding before submission

1. **Table 1 of baseline characteristics.** Demographics, comorbidity, and utilisation
   distributions, stratified by outcome. This is the most conspicuous gap.
2. **Participant flow diagram.** The exclusions in §2.1 rendered as a figure.
3. **SHAP analysis with stability assessment.** Global importance plus the proportion of
   bootstrap resamples in which each feature appears among the top ten, matching the comparator.
4. **Full hyperparameter specification.** Grid and selected values, in a supplementary table.
5. **Univariate predictor associations.** Supplementary table.
6. **Consolidated software and seed paragraph** in the methods.
7. **PROBAST risk-of-bias self-assessment**, which several target venues now expect alongside
   TRIPOD+AI.

Items 1 through 3 are substantive. Items 4 through 6 are bookkeeping against material that
already exists in the repository.

# Cohort Construction and Patient Trajectory Outweigh Model Choice in 30-Day Readmission Prediction: A Re-Analysis of the Diabetes 130-US Hospitals Benchmark

**Aryan Meena**
Boston University Metropolitan College, Boston, MA, USA

*Draft, 17 September 2026. Every figure in this manuscript is reproduced by the scripts in the
accompanying repository.*

---

## Abstract

**Background.** Thirty-day hospital readmission prediction on administrative data has converged
on a narrow performance band, with published area-under-the-curve values on the Diabetes
130-US Hospitals cohort clustering between 0.66 and 0.69. Attention in that literature has
concentrated on model family and on class-imbalance machinery, while cohort construction and
patient-level structure have received comparatively little scrutiny.

**Objectives.** This study asks where the remaining headroom on this benchmark actually lies —
in the algorithm, in the preprocessing, or in the representation of the patient — and whether
transformer architectures offer any advantage over gradient-boosted trees on this problem.

**Methods.** The cohort comprises 101,766 encounters from 130 United States hospitals. Terminal
encounters, meaning those ending in death or transfer to hospice, were removed for the primary
analysis, leaving 99,340 encounters across 69,987 patients with a 30-day readmission rate of
11.39%. Patient-trajectory features were constructed from strictly prior encounters. Model
performance was estimated by nested cross-validation with five outer folds grouped on patient
identifier, and three inner folds for hyperparameter selection. Uncertainty was quantified by
paired bootstrap over 1,000 resamples. Calibration, decision curve analysis, capacity-constrained
targeting, and subgroup performance were assessed out of fold throughout.

**Results.** On the cohort definition used by the reference study, gradient boosting reached a
nested cross-validated ROC-AUC of 0.6888, 95% CI [0.6837, 0.6936], against a published 0.664.
The margin persisted across every model family tested, indicating a data-side rather than a
model-side gain. A stepwise preprocessing ablation attributed roughly 0.006 AUC to patient
trajectory features, 0.003 to native categorical handling, and 0.002 to ICD-9 chapter grouping.
Retaining terminal encounters inflated AUC by approximately 0.007. No class-balancing scheme
improved discrimination, and three of four substantially degraded calibration, with the Brier
score rising from 0.0934 to between 0.1965 and 0.2255. An FT-Transformer and a causal sequence
transformer both underperformed gradient boosting, by 0.038 and 0.020 AUC respectively.
Subgroup analysis revealed a monotonic decline in discrimination with age, from 0.7329 below 45
years to 0.6312 at 80 years and over, with no corresponding disparity by race or sex.

**Conclusions.** On this benchmark, how the cohort is defined, and how a patient's prior
encounters are represented, matter more than which learner is applied. The age-related
discrimination gradient is a calibration-preserving disparity, and points toward age-stratified
operating thresholds rather than recalibration.

**Keywords:** hospital readmission; clinical prediction models; cohort construction; algorithmic
fairness; gradient boosting; decision curve analysis

---

## 1. Introduction

Unplanned readmission within thirty days of discharge is used internationally as an indicator of
care quality, and in the United States it carries direct financial consequences under the
Hospital Readmission Reduction Program. Risk models intended to support targeted intervention
have been studied for well over a decade, and systematic reviews have repeatedly found their
discriminative performance to be modest, with most models falling below an AUC of 0.70 on
administrative data [1, 2].

The Diabetes 130-US Hospitals dataset, released alongside Strack and colleagues' study of HbA1c
measurement [3], has become the de facto public benchmark for this task. Reported performance on
it has been remarkably stable. Shang and colleagues report approximately 0.661 [4],
Emi-Johnson and Nkrumah report 0.667 for XGBoost [5], and Salim and Ibrahim report 0.664 under
nested cross-validation, rising to 0.688 when the final model is retrained on the complete
dataset [6]. That convergence has been read as evidence of a ceiling imposed by administrative
data, and the inference is a reasonable one.

This study examines a different possibility — that a meaningful share of the residual headroom
sits not in the learner but in three decisions taken before the learner ever runs. The first is
which encounters belong in the cohort. The second is how high-cardinality administrative fields
are represented. The third, and the one this study finds most consequential, is whether a
patient is modelled as a sequence of related encounters or as a collection of independent rows.

Nearly thirty percent of encounters in this dataset belong to a patient who appears more than
once, and those repeat encounters carry a readmission rate of 16.7% against 8.8% for first
presentations. Every published model on this cohort that the authors are aware of treats rows as
exchangeable. That choice discards a structure the data plainly contains.

Three secondary questions follow. Does any class-balancing scheme help on a problem at 11%
prevalence? Do transformer architectures, which have reshaped sequence modelling elsewhere,
offer anything here? And does the resulting model perform equitably across patient subgroups —
a question the most recent work on this cohort explicitly leaves open [6]?

### 1.1 Contributions

1. A stepwise attribution of the performance gap between this pipeline and a recent published
   one, isolating the cost of each preprocessing decision rather than attributing the difference
   to model family.
2. Identification and quantification of an outcome-determined feature — discharge to death or
   hospice — that inflates apparent discrimination on this cohort by approximately 0.007 AUC.
3. A direct empirical test of whether patient-level cross-validation matters here, resolving an
   open question posed by prior work, with a negative result.
4. Patient-trajectory features, and evidence that they contribute more than any other single
   modelling decision examined.
5. A controlled comparison against two transformer architectures, and an argument for why
   retrieval-augmented approaches are not applicable to this data.
6. The first subgroup performance analysis on this benchmark.

---

## 2. Materials and Methods

### 2.1 Data source and cohort construction

The dataset comprises 101,766 inpatient encounters drawn from 130 United States hospitals
between 1999 and 2008, each involving a patient receiving diabetes-related care [3]. The primary
endpoint is readmission within thirty days of discharge, encoded in the source as a three-level
variable and collapsed here to a binary outcome.

Encounters with a discharge disposition of expired or hospice — codes 11, 13, 14, 19, 20, and 21
— were removed. This decision warrants explicit justification. Disposition 11, indicating death,
carries a 30-day readmission rate of exactly zero across 1,642 encounters; hospice dispositions
carry 4.8% and 6.5%, against 11.4% for the remainder of the cohort. A model that identifies
these patients acquires discrimination on individuals for whom the outcome was determined at
discharge, and not through any risk-stratifying property of the model. The prior work retains
these encounters and includes an explicit "Discharge: Expired" indicator among its predictors,
where it appears as the largest-magnitude contributor in the published SHAP summary [6]. Section
3.2 quantifies the effect.

Three encounters with an unknown gender were excluded. The resulting analytic cohort contains
99,340 encounters across 69,987 patients, with a 30-day readmission rate of 11.39%. For strict
comparability with published figures, every headline analysis was additionally repeated on the
unfiltered 101,763-encounter cohort.

### 2.2 Feature engineering

The `weight` field, missing for 96.9% of encounters, was removed, as were seven medication
columns carrying a single value across the cohort. Missing values in categorical fields were
retained as an explicit "Unknown" level rather than imputed, on the grounds that whether a field
was recorded is itself informative — consistent with the missing-indicator approach described in
the clinical prediction literature [7, 8]. Age brackets were converted to midpoints.

The three diagnosis fields, carrying 717 distinct ICD-9 codes between them, were mapped to the
eighteen standard clinical chapters. Derived features comprised prior-utilisation rollups,
medication-change and medication-breadth counts, per-day intensity ratios dividing procedure,
laboratory, and medication counts by length of stay, and indicators for whether HbA1c and
glucose tests were performed.

**Patient-trajectory features.** Encounters were ordered within patient by encounter identifier,
which is assigned chronologically in the source data. For each encounter, expanding aggregates
were computed over that patient's strictly earlier encounters — mean and maximum prior length of
stay, mean prior medication and diagnosis counts, maximum prior inpatient visits — together with
deltas between the current encounter and the patient's own history, the encounter's ordinal
position, and the identifier gap since the previous visit as a coarse proxy for elapsed time.
Every such feature applies a one-position shift before aggregation, so no encounter observes
itself or any later encounter.

A prior-outcome feature, counting the patient's earlier 30-day readmissions, was constructed and
then discarded. It contributed −0.0006 AUC once the remaining trajectory features were present,
and removing it eliminates any question about outcome information propagating along a patient's
sequence.

### 2.3 Model development and validation

Performance was estimated by nested cross-validation. The outer loop used five stratified folds
grouped on patient identifier, so that all encounters belonging to one patient fall in the same
fold. The inner loop used three grouped folds for hyperparameter selection over a small grid.
Nested designs of this form are the standard defence against the optimistic bias that arises when
selection and evaluation share data [9].

LightGBM [10] served as the primary learner, with XGBoost [11] configured as described in the
reference study, and penalised logistic regression, as comparators. Categorical variables were
passed to the gradient-boosting learners using native categorical support rather than one-hot
expansion.

### 2.4 Evaluation

Discrimination was reported as ROC-AUC and as area under the precision-recall curve, the latter
being the more informative summary under class imbalance [12]. Calibration was assessed by Brier
score, by decile reliability curves, and by calibration-in-the-large. Uncertainty was quantified
by bootstrap over 1,000 resamples; model comparisons used the paired bootstrap, scoring both
models on identical resampled indices.

Clinical utility was assessed by decision curve analysis [13], computed entirely on out-of-fold
predictions, and by capacity-constrained targeting tables reporting what a fixed intervention
budget yields. Subgroup performance was examined by age band, race, sex, and diabetes medication
status, reporting discrimination and calibration separately within each stratum.

### 2.5 Deep architectures

Two transformer architectures were evaluated. An FT-Transformer [14] tokenises each numerical and
categorical feature into an embedding and applies self-attention across features with a
classification token. A causal sequence transformer [15] operates over each patient's ordered
encounter sequence with a triangular attention mask, predicting at every position from that
position and its predecessors only.

Both were trained on a patient-level subsample of 39,846 encounters drawn from 28,000 patients,
with a single held-out grouped split, because CPU budget did not permit full-cohort training.
This is a genuine limitation, stated in Section 5.

---

## 3. Results

### 3.1 Discrimination against the published benchmark

**Table 1.** Nested cross-validated discrimination on the 101,763-encounter cohort.

| Model | This study | 95% CI | Reference [6] | Difference |
|---|---|---|---|---|
| LightGBM | **0.6888** | [0.6837, 0.6936] | 0.660 | +0.029 |
| XGBoost | 0.6827 | [0.6773, 0.6876] | 0.664 | +0.019 |
| Logistic regression | 0.6758 | [0.6706, 0.6808] | 0.657 | +0.019 |
| Random forest | not run | — | 0.650 | — |
| Stacking ensemble | not run | — | 0.665 | — |

The paired difference between this study's LightGBM and its own XGBoost was +0.0062,
95% CI [+0.0041, +0.0082], p < 0.001; against logistic regression, +0.0131, 95% CI
[+0.0107, +0.0155], p < 0.001.

The margin is approximately constant across model families. Logistic regression and XGBoost each
gain about 0.019 over their published counterparts, and the additional 0.006 separating LightGBM
from XGBoost is small beside that common shift. This pattern is the signature of a data-side
rather than a model-side effect, and motivates the decomposition that follows.

*[Figure 1: fig1_model_comparison.png]*

### 3.2 Where the gain originates

**Table 2.** Cumulative preprocessing ablation, five-fold patient-grouped cross-validation with
identical model settings throughout.

| Configuration | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Full pipeline, this study | 0.6895 | 0.2404 | 0.0934 |
| − `payer_code` | 0.6884 | 0.2391 | 0.0935 |
| − ICD-9 chapters, replaced by binary flags | 0.6856 | 0.2341 | 0.0937 |
| − native categorical handling | 0.6829 | 0.2295 | 0.0940 |
| + cost-sensitive weighting | 0.6829 | 0.2269 | 0.2053 |
| − patient-trajectory features | 0.6768 | 0.2244 | 0.2063 |

Patient trajectory is the largest single contributor, at approximately 0.006 AUC. Native
categorical handling contributes about 0.003, and diagnosis chapters about 0.002. Reducing
`medical_specialty` to its ten most frequent levels, as the reference study does, proved
marginally beneficial at +0.0007, and is therefore not a source of loss.

Cost-sensitive weighting costs nothing in discrimination and more than doubles the Brier score.
This is the mechanism by which prior work arrived at a poorly calibrated model requiring post-hoc
Platt correction; leaving the class distribution untouched avoids the problem entirely.

A residual of approximately 0.013 separates this reproduction of the published recipe from the
published figure itself. That residual is not explained here. The most plausible sources are the
exact 55-predictor list, which is not published, and XGBoost hyperparameter selection.

*[Figure 2: fig2_preprocessing_ablation.png]*

**Terminal encounters.** Retaining encounters ending in death or hospice raised AUC from 0.6824
to 0.6895 under an otherwise identical pipeline — approximately 0.007, or roughly one third of
the margin by which the reference study leads the earlier literature on this cohort.

### 3.3 Patient-level splitting

**Table 3.** Row-level against patient-grouped cross-validation, two seeds, five folds each.

| Splitting strategy | ROC-AUC |
|---|---|
| Row-level, patients may span folds | 0.6787 ± 0.0040 |
| Patient-grouped | 0.6773 ± 0.0036 |
| Difference | +0.0014, p = 0.45 |

The reference study lists the absence of patient-level splitting among its limitations and
recommends grouped strategies as future work [6]. Tested directly, the effect is not
statistically distinguishable from zero. With no patient identifier among the predictors, the
learner cannot memorise individuals, and repeat encounters behave as ordinary correlated
observations. Grouped splitting remains the correct default, since it costs nothing and guards
against feature sets that would leak, but on this cohort it is a safeguard rather than a
correction. Every result in this manuscript nonetheless uses grouped splits.

### 3.4 Patient trajectory

**Table 4.** Trajectory feature contribution, five-fold patient-grouped cross-validation.

| Feature set | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Cross-sectional only | 0.6851 ± 0.0097 | 0.2390 | 0.0935 |
| + encounter ordinal | 0.6889 ± 0.0074 | 0.2399 | 0.0934 |
| + full trajectory features | **0.6963 ± 0.0078** | **0.2489** | **0.0929** |

Trajectory modelling contributes +0.0112 AUC over a cross-sectional representation — larger than
the combined contribution of every other engineered feature family, and larger than the
difference between any two model families tested.

The descriptive picture supports this. Repeat encounters number 30,248, or 29.7% of the cohort,
and carry a readmission rate of 16.74% against 8.80% for first presentations.

### 3.5 Class balancing

**Table 5.** Balancing strategies under identical model settings and feature set.

| Strategy | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| None, natural prevalence | **0.6895** | **0.2403** | **0.0934** |
| Cost-sensitive weighting | 0.6872 | 0.2345 | 0.1965 |
| SMOTE [16] | 0.6865 | 0.2366 | 0.0937 |
| Random oversampling | 0.6876 | 0.2347 | 0.2022 |
| Random undersampling | 0.6756 | 0.2168 | 0.2255 |

No strategy improved discrimination. Three of the four degraded calibration severely, because
resampling and reweighting both shift the training prior away from the deployment prior. At 11%
prevalence, with a learner optimising log-loss, leaving the distribution alone and adjusting the
decision threshold afterwards dominates every alternative examined.

*[Figure 3: fig3_balancing.png]*

### 3.6 Transformer architectures

**Table 6.** Deep architectures against gradient boosting, 39,846-encounter patient-level
subsample, single held-out grouped split.

| Model | ROC-AUC | PR-AUC | Brier | Training time |
|---|---|---|---|---|
| LightGBM | **0.6662** | **0.1956** | **0.0950** | 2 s |
| Sequence Transformer, causal | 0.6463 | 0.1810 | 0.0960 | 19 s |
| FT-Transformer, tabular | 0.6282 | 0.1741 | 0.0959 | 131 s |

Neither architecture matched gradient boosting, consistent with systematic findings that
tree-based methods retain an advantage on tabular data of this scale [17].

The ordering within the two transformers is informative. The sequence model, which sees each
patient's history, outperforms the tabular model, which does not, by 0.018 AUC. The trajectory
signal identified in Section 3.4 is therefore real enough that a neural architecture recovers
part of it unaided. Explicit trajectory features inside a boosted tree recover more of it, at
roughly one-sixtieth of the training cost.

**On retrieval-augmented generation.** Retrieval-augmented approaches retrieve passages from a
text corpus to ground a language model's output. This dataset contains fifty columns of
structured administrative codes, with no clinical notes, discharge summaries, or free text of any
kind. There is no corpus over which to retrieve, and a retrieval layer would have no semantic
content to condition on. Such approaches become relevant on cohorts carrying narrative text, and
that constitutes a distinct study rather than an extension of this one.

### 3.7 Calibration

The primary model required no post-hoc calibration. Isotonic regression applied on top of it
slightly worsened both the Brier score, from 0.0953 to 0.0957, and discrimination, from 0.6826 to
0.6814. A gradient-boosting model trained on the natural class distribution against a proper
scoring rule is already calibrated, and additional correction introduces variance without
addressing any deficit [18].

Out-of-fold reliability across risk deciles showed close agreement, with mild under-prediction
confined to the top decile — predicted 0.244 against observed 0.278.

*[Figure 6: fig6_calibration.png]*

### 3.8 Subgroup performance

**Table 7.** Discrimination and calibration by subgroup, out of fold.

| Subgroup | n | Prevalence | ROC-AUC | Brier | Observed / predicted |
|---|---|---|---|---|---|
| Overall | 99,340 | 0.114 | 0.6813 | 0.0957 | 1.015 |
| Age under 45 | 15,870 | 0.109 | **0.7329** | 0.0889 | 1.039 |
| Age 45 to 65 | 39,118 | 0.106 | 0.6906 | 0.0898 | 0.998 |
| Age 65 to 80 | 25,329 | 0.121 | 0.6613 | 0.1015 | 1.020 |
| Age 80 and over | 19,023 | **0.125** | **0.6312** | 0.1058 | 1.024 |
| African American | 18,772 | 0.114 | 0.6760 | 0.0966 | 1.009 |
| Caucasian | 74,220 | 0.115 | 0.6804 | 0.0967 | 1.020 |
| Female | 53,454 | 0.115 | 0.6828 | 0.0962 | 1.014 |
| Male | 45,886 | 0.113 | 0.6795 | 0.0951 | 1.017 |

Discrimination declines monotonically with age while prevalence rises, a gap of 0.102 between the
youngest and oldest bands. The model is weakest precisely where risk is highest, and where a
prevention programme would concentrate its effort.

No comparable disparity appears by race or by sex, and calibration holds within every stratum
examined, with observed-over-predicted ratios between 0.93 and 1.04. The disparity is therefore
one of discrimination rather than calibration — a distinction with operational consequences,
since recalibration cannot repair a ranking that is less informative to begin with. Age-stratified
operating thresholds are the appropriate response.

*[Figure 4: fig4_subgroup_age.png]*

### 3.9 Clinical utility

**Table 8.** Capacity-constrained targeting, out of fold, base rate 11.39%.

| Target | Flagged | Captured | Recall | Precision | Lift |
|---|---|---|---|---|---|
| Top 5% | 4,967 | 1,696 | 15.0% | 34.1% | 3.00× |
| Top 10% | 9,934 | 2,764 | 24.4% | 27.8% | 2.44× |
| Top 20% | 19,868 | 4,517 | 39.9% | 22.7% | 2.00× |
| Top 30% | 29,802 | 5,973 | 52.8% | 20.0% | 1.76× |

Decision curve net benefit exceeded both treat-all and treat-none at every threshold between 0.05
and 0.30.

*[Figure 5: fig5_roc_pr.png]* · *[Figure 7: fig7_decision_curve.png]*

---

## 4. Discussion

The central finding is that the remaining headroom on this benchmark is concentrated in decisions
made before modelling begins. Patient trajectory contributes more than any model substitution
tested. Cohort definition contributes roughly 0.007 by itself. Representation choices for
high-cardinality categorical fields contribute a further 0.005 between them. Against these, the
difference between two mature gradient-boosting implementations is 0.006, and the difference
between gradient boosting and a transformer is negative.

This has a practical implication for a literature that has largely organised itself around model
comparison. Where a benchmark appears to have plateaued, the plateau may reflect a shared set of
preprocessing conventions rather than an information limit in the data.

### 4.1 Comparison with prior work

**Table 9.** Reported performance on the Diabetes 130-US Hospitals cohort.

| Study | Evaluation | Reported AUC |
|---|---|---|
| Shang et al. [4] | 80/20 split | ≈0.661 |
| Emi-Johnson and Nkrumah [5] | train/test | 0.667 |
| Salim and Ibrahim [6] | nested CV | 0.664 |
| Salim and Ibrahim [6] | full-dataset retraining | 0.688 |
| **This study** | **nested CV, patient-grouped** | **0.6888** |
| **This study** | **nested CV, trajectory features, clinical cohort** | **0.6963** |

Two cautions apply to this table. First, the cohort definitions differ — this study's clinical
cohort excludes terminal encounters, which the others retain. Second, a full-dataset AUC computed
after retraining on all available data is not comparable to a cross-validated estimate, a point
the reference study itself makes [6].

### 4.2 The age gradient

The subgroup result deserves emphasis because it is invisible to aggregate reporting. A model
reported at 0.68 overall performs at 0.63 among patients aged eighty and over, who constitute
nineteen percent of the cohort and carry its highest readmission rate. Deployed against a fixed
intervention budget, such a model would allocate least accurately where the need is greatest.

The literature on algorithmic equity in health has concentrated largely on racial disparity [19].
This cohort shows none worth acting on, and instead shows an age gradient of comparable
magnitude. Reporting subgroup discrimination should be routine rather than exceptional.

### 4.3 Class imbalance

The imbalance result is negative and, the authors would argue, useful. Resampling remains widely
applied on this dataset. On evidence here it does not improve ranking, and where it shifts the
training prior it materially harms probability estimates. Since threshold selection is exactly
the decision a deployed readmission model supports, damaged probabilities are not a cosmetic
concern.

### 4.4 Limitations

The data spans 1999 to 2008, predating the Hospital Readmission Reduction Program introduced in
2012. Coding practice, length-of-stay norms, and discharge planning have all changed since, and
the cohort is best read as a methodological benchmark rather than a deployable contemporary tool.

Planned and unplanned readmissions cannot be distinguished in the source, which introduces noise
into the outcome definition.

No external validation was performed. This is the principal limitation, and it is shared with the
studies compared against. Conclusions here concern relative effects of modelling decisions within
one cohort, which is a narrower claim than transportable performance.

The deep-learning comparison used a subsample, a single split, and a modest training budget on
CPU. A larger budget might narrow the gap, though the direction is consistent with the broader
tabular literature [17]. Architecture search was not performed for either transformer.

The encounter-identifier gap is a coarse proxy for elapsed time between visits, as the dataset
carries no dates. Trajectory features are right-censored at both ends of the observation window.

Finally, approximately 0.013 of the difference from the published figure remains unattributed.

---

## 5. Conclusions

Cohort construction and patient representation dominate model choice on this benchmark. Removing
encounters whose outcome was fixed at discharge, representing patients as sequences rather than
independent rows, and leaving the class distribution alone together produce a larger improvement
than any substitution of learner, including two transformer architectures that both underperform
gradient boosting at substantially greater cost.

The accompanying subgroup analysis surfaces an age-related discrimination gradient that aggregate
reporting conceals, and which points toward stratified operating thresholds.

External validation on contemporary multi-centre data remains a precondition for any clinical
consideration of a model of this kind.

---

## Data and code availability

Both datasets are public. The Diabetes 130-US Hospitals cohort is available from the UCI Machine
Learning Repository. All analysis code, including scripts reproducing every table and figure, is
available in the accompanying repository.

## Conflicts of interest

None declared.

## Acknowledgement of tooling

Analyses were developed with the assistance of an AI coding assistant. All experimental design,
verification of outputs, and interpretation are the author's own. Disclosure follows the
prevailing guidance of the target venue.

---

## References

1. Kansagara, D.; Englander, H.; Salanitro, A.; Kagen, D.; Theobald, C.; Freeman, M.; Kripalani, S. Risk prediction models for hospital readmission: A systematic review. *JAMA* **2011**, *306*, 1688–1698.
2. Mahmoudi, E.; Kamdar, N.; Kim, N.; Gonzales, G.; Singh, K.; Waljee, A.K. Use of electronic medical records in development and validation of risk prediction models of hospital readmission: Systematic review. *BMJ* **2020**, *369*, m958.
3. Strack, B.; DeShazo, J.P.; Gennings, C.; Olmo, J.L.; Ventura, S.; Cios, K.J.; Clore, J.N. Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Res. Int.* **2014**, *2014*, 781670.
4. Shang, Y.; Jiang, K.; Wang, L.; Zhang, Z.; Zhou, S.; Liu, Y.; Dong, J.; Wu, H. The 30-days hospital readmission risk in diabetic patients: Predictive modeling with machine learning classifiers. *BMC Med. Inform. Decis. Mak.* **2021**, *21*, 57.
5. Emi-Johnson, O.G.; Nkrumah, K.J. Predicting 30-day hospital readmission in patients with diabetes using machine learning on electronic health record data. *Cureus* **2025**, *17*, e82437.
6. Salim, S.S.; Ibrahim, A.A. A machine learning approach for predicting 30-day hospital readmission in patients with diabetes. *Healthcare* **2026**, *14*, 1185.
7. Sisk, R.; Sperrin, M.; Peek, N.; van Smeden, M.; Martin, G.P. Imputation and missing indicators for handling missing data in the development and deployment of clinical prediction models: A simulation study. *Stat. Methods Med. Res.* **2023**, *32*, 1461–1477.
8. Digitale, J.; Franzon, D.; Pletcher, M.J.; McCulloch, C.E.; Gennatas, E.D. Methods for addressing missingness in electronic health record data for clinical prediction models: Comparative evaluation. *JMIR Med. Inform.* **2025**, *13*, e79307.
9. Varma, S.; Simon, R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinform.* **2006**, *7*, 91.
10. Ke, G.; Meng, Q.; Finley, T.; Wang, T.; Chen, W.; Ma, W.; Ye, Q.; Liu, T.-Y. LightGBM: A highly efficient gradient boosting decision tree. *Adv. Neural Inf. Process. Syst.* **2017**, *30*, 3146–3154.
11. Chen, T.; Guestrin, C. XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*; ACM: New York, NY, USA, 2016; pp. 785–794.
12. Saito, T.; Rehmsmeier, M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE* **2015**, *10*, e0118432.
13. Vickers, A.J.; Elkin, E.B. Decision curve analysis: A novel method for evaluating prediction models. *Med. Decis. Mak.* **2006**, *26*, 565–574.
14. Gorishniy, Y.; Rubachev, I.; Khrulkov, V.; Babenko, A. Revisiting deep learning models for tabular data. *Adv. Neural Inf. Process. Syst.* **2021**, *34*, 18932–18943.
15. Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A.N.; Kaiser, Ł.; Polosukhin, I. Attention is all you need. *Adv. Neural Inf. Process. Syst.* **2017**, *30*, 5998–6008.
16. Chawla, N.V.; Bowyer, K.W.; Hall, L.O.; Kegelmeyer, W.P. SMOTE: Synthetic minority over-sampling technique. *J. Artif. Intell. Res.* **2002**, *16*, 321–357.
17. Grinsztajn, L.; Oyallon, E.; Varoquaux, G. Why do tree-based models still outperform deep learning on typical tabular data? *Adv. Neural Inf. Process. Syst. Datasets and Benchmarks Track* **2022**, *35*.
18. Van Calster, B.; McLernon, D.J.; van Smeden, M.; Wynants, L.; Steyerberg, E.W. Calibration: The Achilles heel of predictive analytics. *BMC Med.* **2019**, *17*, 230.
19. Obermeyer, Z.; Powers, B.; Vogeli, C.; Mullainathan, S. Dissecting racial bias in an algorithm used to manage the health of populations. *Science* **2019**, *366*, 447–453.
20. Walsh, C.G.; Sharman, K.; Hripcsak, G. Beyond discrimination: A comparison of calibration methods and clinical usefulness of predictive models of readmission risk. *J. Biomed. Inform.* **2017**, *76*, 9–18.
21. Collins, G.S.; Moons, K.G.M.; Dhiman, P.; Riley, R.D.; Beam, A.L.; Van Calster, B.; et al. TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* **2024**, *385*, e078378.
22. Lundberg, S.M.; Lee, S.-I. A unified approach to interpreting model predictions. *Adv. Neural Inf. Process. Syst.* **2017**, *30*, 4766–4775.

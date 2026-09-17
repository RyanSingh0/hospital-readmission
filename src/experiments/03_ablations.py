"""Experiment 03 — feature families, and class-balancing strategies.

Two questions. Which engineered feature families actually earn their place, and does any
resampling scheme help? The answer to the second turns out to be no, and three of the four
schemes tested do real damage to calibration while leaving discrimination essentially alone.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb
import preprocessing as diab_prep
from preprocessing import NUM, CAT

orig=diab_prep.DEAD; diab_prep.DEAD=set()
d=diab_prep.load(verbose=False); diab_prep.DEAD=orig
d=d.sort_values('encounter_id')
d['enc_seq']=d.groupby('patient_nbr').cumcount(); d['is_repeat']=(d.enc_seq>0).astype(int)
y=d.y.values; g=d.patient_nbr.values
P=dict(n_estimators=600,learning_rate=0.03,num_leaves=31,min_child_samples=100,
       reg_lambda=5,colsample_bytree=0.7,subsample=0.8,subsample_freq=1,verbose=-1,random_state=0)

RAW_NUM=['age_mid','time_in_hospital','num_lab_procedures','num_procedures','num_medications',
         'number_outpatient','number_emergency','number_inpatient','number_diagnoses']
RAW_CAT=['race','gender','admission_type_id','discharge_disposition_id','admission_source_id',
         'medical_specialty','payer_code','max_glu_serum','A1Cresult','change','diabetesMed',
         'metformin','insulin','glipizide','glyburide','pioglitazone','rosiglitazone','glimepiride']
UTIL=['prior_visits','had_inpatient','had_emergency']
MEDS=['n_drug_changes','n_drugs_on']
RATE=['meds_per_day','procs_per_day','labs_per_day']
FLAG=['a1c_tested','glu_tested','insulin_on']
DXG=['diag_1_grp','diag_2_grp','diag_3_grp']
HIST=['enc_seq','is_repeat']

def run(cols,label,sampler=None,weight=False):
    X=d[cols].copy()
    for c in cols:
        if c in CAT or c in DXG: X[c]=X[c].astype('category')
    a,ap,br=[],[],[]
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=0).split(X,y,g):
        Xtr,ytr=X.iloc[tr],y[tr]
        p=dict(P)
        if weight: p['scale_pos_weight']=(ytr==0).sum()/(ytr==1).sum()
        if sampler is not None:
            Xd=pd.get_dummies(Xtr,dummy_na=True)
            Xd,ytr2=sampler.fit_resample(Xd,ytr)
            m=lgb.LGBMClassifier(**p); m.fit(Xd,ytr2)
            Xte=pd.get_dummies(X.iloc[te],dummy_na=True).reindex(columns=Xd.columns,fill_value=0)
            pr=m.predict_proba(Xte)[:,1]
        else:
            m=lgb.LGBMClassifier(**p); m.fit(Xtr,ytr)
            pr=m.predict_proba(X.iloc[te])[:,1]
        a.append(roc_auc_score(y[te],pr)); ap.append(average_precision_score(y[te],pr)); br.append(brier_score_loss(y[te],pr))
    print(f"  {label:<48}{np.mean(a):.4f} ±{np.std(a):.4f}   {np.mean(ap):.4f}   {np.mean(br):.4f}",flush=True)
    return np.mean(a)

print("FEATURE-FAMILY ABLATION  (5-fold patient-grouped CV, identical model settings)")
print(f"  {'feature set':<48}{'AUC':>8}{'':>9}{'PR-AUC':>9}{'Brier':>9}")
base=RAW_NUM+RAW_CAT
r={}
r['raw']=run(base,"A. Raw columns only (no engineering)")
r['dxg']=run(base+DXG,"B. + ICD-9 chapter grouping")
r['util']=run(base+DXG+UTIL,"C. + prior-utilisation rollups")
r['meds']=run(base+DXG+UTIL+MEDS,"D. + medication-change counts")
r['rate']=run(base+DXG+UTIL+MEDS+RATE,"E. + per-day intensity ratios")
r['flag']=run(base+DXG+UTIL+MEDS+RATE+FLAG,"F. + lab-performed flags")
r['hist']=run(base+DXG+UTIL+MEDS+RATE+FLAG+HIST,"G. + patient visit history  [FULL]")
print(f"\n  total gain from feature engineering: {r['hist']-r['raw']:+.4f} AUC")
print(f"  paper's reported XGBoost: 0.664   our raw-feature LightGBM: {r['raw']:.4f}")

print("\n\nCLASS-BALANCING ABLATION  (full feature set, identical model settings)")
print(f"  {'strategy':<48}{'AUC':>8}{'':>9}{'PR-AUC':>9}{'Brier':>9}")
FULL=base+DXG+UTIL+MEDS+RATE+FLAG+HIST
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
run(FULL,"1. None (natural prevalence)")
run(FULL,"2. scale_pos_weight (cost-sensitive)",weight=True)
run(FULL,"3. SMOTE oversampling",sampler=SMOTE(random_state=0))
run(FULL,"4. Random oversampling",sampler=RandomOverSampler(random_state=0))
run(FULL,"5. Random undersampling",sampler=RandomUnderSampler(random_state=0))

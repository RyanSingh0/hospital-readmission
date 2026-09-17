"""Experiment 05 — patient-trajectory features.

Every published model on this dataset treats encounters as independent rows, yet 29.7% of them
belong to a patient who appears more than once, and those repeat encounters carry nearly twice
the readmission rate. This experiment quantifies what modelling that history is worth.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb
import preprocessing as diab_prep
from preprocessing import NUM, CAT

orig=diab_prep.DEAD; diab_prep.DEAD=set()
d=diab_prep.load(verbose=False); diab_prep.DEAD=orig
d=d.sort_values(['patient_nbr','encounter_id']).reset_index(drop=True)

gb=d.groupby('patient_nbr')
d['enc_seq']=gb.cumcount()
d['is_repeat']=(d.enc_seq>0).astype(int)
# strictly causal: shift(1) then expanding, so row k sees only rows 1..k-1 of the same patient
d['prior_readmits']=gb.y.apply(lambda s:s.shift(1).expanding().sum()).reset_index(level=0,drop=True).fillna(0)
d['prior_readmit_rate']=(d.prior_readmits/d.enc_seq.replace(0,np.nan)).fillna(-1)
d['prior_los_mean']=gb.time_in_hospital.apply(lambda s:s.shift(1).expanding().mean()).reset_index(level=0,drop=True).fillna(-1)
d['prior_los_max']=gb.time_in_hospital.apply(lambda s:s.shift(1).expanding().max()).reset_index(level=0,drop=True).fillna(-1)
d['prior_meds_mean']=gb.num_medications.apply(lambda s:s.shift(1).expanding().mean()).reset_index(level=0,drop=True).fillna(-1)
d['prior_dx_mean']=gb.number_diagnoses.apply(lambda s:s.shift(1).expanding().mean()).reset_index(level=0,drop=True).fillna(-1)
d['prior_inpatient_max']=gb.number_inpatient.apply(lambda s:s.shift(1).expanding().max()).reset_index(level=0,drop=True).fillna(-1)
d['los_vs_prior']=np.where(d.prior_los_mean>0,d.time_in_hospital-d.prior_los_mean,0)
d['meds_vs_prior']=np.where(d.prior_meds_mean>0,d.num_medications-d.prior_meds_mean,0)
d['enc_id_gap']=gb.encounter_id.diff().fillna(-1)   # proxy for elapsed time between visits

HIST=['enc_seq','is_repeat']
TRAJ=['prior_readmits','prior_readmit_rate','prior_los_mean','prior_los_max','prior_meds_mean',
      'prior_dx_mean','prior_inpatient_max','los_vs_prior','meds_vs_prior','enc_id_gap']
y=d.y.values; g=d.patient_nbr.values
P=dict(n_estimators=600,learning_rate=0.03,num_leaves=31,min_child_samples=100,
       reg_lambda=5,colsample_bytree=0.7,subsample=0.8,subsample_freq=1,verbose=-1,random_state=0)

def ev(cols,label):
    X=d[cols].copy()
    for c in cols:
        if c in CAT: X[c]=X[c].astype('category')
    a,ap,br=[],[],[]
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=0).split(X,y,g):
        m=lgb.LGBMClassifier(**P); m.fit(X.iloc[tr],y[tr]); pr=m.predict_proba(X.iloc[te])[:,1]
        a.append(roc_auc_score(y[te],pr)); ap.append(average_precision_score(y[te],pr)); br.append(brier_score_loss(y[te],pr))
    print(f"  {label:<50}{np.mean(a):.4f} ±{np.std(a):.4f}   {np.mean(ap):.4f}   {np.mean(br):.4f}",flush=True)
    return np.mean(a)

print("PATIENT-TRAJECTORY FEATURES  (5-fold patient-grouped CV)")
print(f"  {'feature set':<50}{'AUC':>8}{'':>9}{'PR-AUC':>9}{'Brier':>9}")
b=ev(NUM+CAT,"A. Cross-sectional only (no history)")
h=ev(NUM+HIST+CAT,"B. + visit counter")
t=ev(NUM+HIST+TRAJ+CAT,"C. + full trajectory features")
print(f"\n  gain from trajectory modelling: {t-b:+.4f} AUC over cross-sectional")

# how much is the prior-outcome feature specifically?
t2=ev(NUM+HIST+[x for x in TRAJ if 'readmit' not in x]+CAT,"D. trajectory WITHOUT prior-outcome features")
print(f"  of which prior-outcome history contributes: {t-t2:+.4f}")
sub=d[d.is_repeat==1]
print(f"\n  repeat encounters: {len(sub):,} ({len(sub)/len(d)*100:.1f}%)  readmit rate {sub.y.mean():.4f} vs {d[d.is_repeat==0].y.mean():.4f} for first visits")
print(f"  patients with >=1 prior readmission: readmit rate {d[d.prior_readmits>0].y.mean():.4f} (n={int((d.prior_readmits>0).sum()):,})")

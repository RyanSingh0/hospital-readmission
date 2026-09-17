"""Experiment 04 — walking from this pipeline to the published one, one choice at a time.

Each step applies a single preprocessing decision taken by the prior work, so the cost of that
decision can be read off directly instead of being attributed to the model family.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
import preprocessing as diab_prep
from preprocessing import NUM, CAT

orig=diab_prep.DEAD; diab_prep.DEAD=set()
d=diab_prep.load(verbose=False); diab_prep.DEAD=orig
d=d.sort_values('encounter_id')
d['enc_seq']=d.groupby('patient_nbr').cumcount(); d['is_repeat']=(d.enc_seq>0).astype(int)
y=d.y.values; g=d.patient_nbr.values
FULLN=NUM+['enc_seq','is_repeat']; FULLC=list(CAT)
P=dict(n_estimators=600,learning_rate=0.03,num_leaves=31,min_child_samples=100,
       reg_lambda=5,colsample_bytree=0.7,subsample=0.8,subsample_freq=1,verbose=-1,random_state=0)

# the paper's diagnosis treatment: binary flags instead of chapter categories
for grp in ['Circulatory','Respiratory','Digestive','Diabetes','Injury','Neoplasms']:
    d['has_'+grp.lower()]=((d.diag_1_grp==grp)|(d.diag_2_grp==grp)|(d.diag_3_grp==grp)).astype(int)
FLAGS=['has_'+x for x in ['circulatory','respiratory','digestive','diabetes','injury','neoplasms']]
# the paper's specialty reduction: top 10 + Other
top10=d.medical_specialty.value_counts().head(10).index
d['spec_red']=np.where(d.medical_specialty.isin(top10),d.medical_specialty,'Other')

def evaluate(num,cat,label,onehot=False,weight=False):
    a,ap,br=[],[],[]
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=0).split(d,y,g):
        p=dict(P)
        if weight: p['scale_pos_weight']=(y[tr]==0).sum()/(y[tr]==1).sum()
        if onehot:
            pre=ColumnTransformer([('n',StandardScaler(),num),
                                   ('c',OneHotEncoder(handle_unknown='ignore',min_frequency=20),cat)])
            m=Pipeline([('p',pre),('m',lgb.LGBMClassifier(**p))])
            m.fit(d[num+cat].iloc[tr],y[tr]); pr=m.predict_proba(d[num+cat].iloc[te])[:,1]
        else:
            X=d[num+cat].copy()
            for c in cat: X[c]=X[c].astype('category')
            m=lgb.LGBMClassifier(**p); m.fit(X.iloc[tr],y[tr]); pr=m.predict_proba(X.iloc[te])[:,1]
        a.append(roc_auc_score(y[te],pr)); ap.append(average_precision_score(y[te],pr)); br.append(brier_score_loss(y[te],pr))
    print(f"  {label:<54}{np.mean(a):.4f}   {np.mean(ap):.4f}   {np.mean(br):.4f}",flush=True)
    return np.mean(a)

print("WALKING FROM OUR PIPELINE TO THE PAPER'S, ONE CHOICE AT A TIME")
print(f"  {'configuration':<54}{'AUC':>6}{'':>4}{'PR-AUC':>7}{'':>3}{'Brier':>6}")
base=evaluate(FULLN,FULLC,"0. Ours (native categoricals, all columns, no weighting)")
a1=evaluate(FULLN,[c for c in FULLC if c!='payer_code'],"1. - drop payer_code (as the paper does)")
c2=[c for c in FULLC if c!='medical_specialty']+['spec_red']
a2=evaluate(FULLN,c2,"2. - reduce medical_specialty to top-10 + Other")
c3=[c for c in c2 if c not in ('diag_1_grp','diag_2_grp','diag_3_grp')]
a3=evaluate(FULLN+FLAGS,c3,"3. - binary diagnosis flags instead of chapters")
c4=[c for c in c3 if c!='payer_code']
a4=evaluate(FULLN+FLAGS,c4,"4. - all three of the above together")
a5=evaluate(FULLN+FLAGS,c4,"5. + one-hot encoding instead of native categoricals",onehot=True)
a6=evaluate(FULLN+FLAGS,c4,"6. + cost-sensitive weighting  [= the paper's recipe]",onehot=True,weight=True)
n7=[c for c in FULLN if c not in ('enc_seq','is_repeat')]
a7=evaluate(n7+FLAGS,c4,"7. - remove visit history  [full paper replication]",onehot=True,weight=True)
print(f"\n  ours {base:.4f}  ->  paper's recipe {a7:.4f}   difference {base-a7:+.4f}")
print(f"  paper's own reported XGBoost figure: 0.664")
print("\n  cost of each individual choice, measured from our pipeline:")
for lab,v in [("drop payer_code",a1),("reduce medical_specialty",a2),("binary diagnosis flags",a3)]:
    print(f"    {lab:<34}{v-base:+.4f}")
print(f"    {'one-hot vs native categorical':<34}{a5-a4:+.4f}")
print(f"    {'cost-sensitive weighting':<34}{a6-a5:+.4f}")
print(f"    {'removing visit history':<34}{a7-a6:+.4f}")

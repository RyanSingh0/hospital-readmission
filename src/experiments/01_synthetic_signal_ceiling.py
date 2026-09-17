"""Experiment 01 — the signal ceiling of the synthetic 30k cohort.

Establishes that the Kaggle synthetic file was generated from a single two-branch rule, and
that no model can exceed an AUC of roughly 0.581 on it. Run this before drawing any
conclusion from that dataset, because the ceiling is the only meaningful benchmark it has.
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_rel
import lightgbm as lgb
d=pd.read_csv('../../data/hospital_readmissions_30k.csv'); d['y']=(d.readmitted_30_days=='Yes').astype(int)
d['nonhome']=(d.discharge_destination!='Home').astype(int)
d['comorb']=((d.diabetes=='Yes')|(d.hypertension=='Yes')).astype(int)
d['RULE']=d.nonhome*d.comorb
y=d.y.values
NUM=['age','cholesterol','bmi','medication_count','length_of_stay']
CAT_ALL=['gender','blood_pressure','diabetes','hypertension','discharge_destination']
CAT_SIG=['diabetes','hypertension','discharge_destination']
def prep(num,cat): return ColumnTransformer([('n',StandardScaler(),num),('c',OneHotEncoder(handle_unknown='ignore'),cat)])
RF=dict(n_estimators=150,max_depth=20,random_state=0,n_jobs=-1)
specs={
 'A. Original recipe (RF+SMOTE, BP one-hot)':(lambda: ImbPipeline([('p',prep(NUM,CAT_ALL)),('s',SMOTE(random_state=0)),('m',RandomForestClassifier(**RF))]), NUM+CAT_ALL),
 'B. A without SMOTE':(lambda: Pipeline([('p',prep(NUM,CAT_ALL)),('m',RandomForestClassifier(**RF))]), NUM+CAT_ALL),
 'C. RF, blood_pressure dropped':(lambda: Pipeline([('p',prep(NUM,['gender']+CAT_SIG)),('m',RandomForestClassifier(**RF))]), NUM+['gender']+CAT_SIG),
 'D. Logistic, 3 signal features only':(lambda: Pipeline([('p',prep([],CAT_SIG)),('m',LogisticRegression(max_iter=1000))]), CAT_SIG),
 'E. Logistic, 3 signal + interaction':(lambda: Pipeline([('p',prep(['RULE'],CAT_SIG)),('m',LogisticRegression(max_iter=1000))]), CAT_SIG+['RULE']),
 'G. ORACLE single binary rule':(lambda: Pipeline([('m',LogisticRegression())]), ['RULE']),
 'H. LightGBM regularised, all features':(None, NUM+CAT_ALL),
}
folds=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=7).split(d,y))
scores={}
for name,(mk,cols) in specs.items():
    s=[]
    for tr,te in folds:
        if mk is None:
            X=d[cols].copy()
            for c in CAT_ALL: X[c]=X[c].astype('category')
            m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.03,num_leaves=8,min_child_samples=200,
                reg_lambda=10,colsample_bytree=0.7,subsample=0.8,subsample_freq=1,verbose=-1,random_state=0)
            m.fit(X.iloc[tr],y[tr]); p=m.predict_proba(X.iloc[te])[:,1]
        else:
            m=mk(); m.fit(d[cols].iloc[tr],y[tr]); p=m.predict_proba(d[cols].iloc[te])[:,1]
        s.append(roc_auc_score(y[te],p))
    scores[name]=np.array(s); print(f"done {name}",flush=True)
np.save('scores.npy',scores,allow_pickle=True)
print("\n"+f"{'model':<46}{'AUC (15 folds)':>16}{'95% CI':>20}")
print("-"*84)
for n,s in sorted(scores.items(),key=lambda x:-x[1].mean()):
    ci=1.96*s.std(ddof=1)/np.sqrt(len(s))
    print(f"{n:<46}{s.mean():>11.4f} ±{s.std(ddof=1):.4f}  [{s.mean()-ci:.4f}, {s.mean()+ci:.4f}]")
print("-"*84)
print(f"{'ANALYTIC CEILING (Bayes AUC, true rule)':<46}{0.5814:>11.4f}")
print(f"{'Original project reported':<46}{0.5654:>11.4f}")
best=max(scores,key=lambda k:scores[k].mean())
print(f"\npaired t-tests vs {best}:")
for n,s in scores.items():
    if n==best: continue
    t,p=ttest_rel(scores[best],s)
    print(f"   vs {n:<44} diff {scores[best].mean()-s.mean():+.4f}  p={p:.1e}")

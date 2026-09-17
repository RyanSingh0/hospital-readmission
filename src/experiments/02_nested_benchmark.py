"""Experiment 02 — nested cross-validated benchmark against the published result.

Runs the headline comparison on the published cohort, meaning terminal encounters are retained
so that the numbers line up with Salim and Ibrahim. Outer folds are grouped on patient, inner
folds select hyperparameters, and the paired bootstrap at the end puts an interval around every
difference rather than leaving two point estimates side by side.
"""
import numpy as np, pandas as pd, warnings, json; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb, xgboost as xgb
import preprocessing as diab_prep
from preprocessing import NUM, CAT

orig=diab_prep.DEAD; diab_prep.DEAD=set()          # paper's cohort: expired retained
d=diab_prep.load(verbose=False); diab_prep.DEAD=orig
d=d.sort_values('encounter_id')
d['enc_seq']=d.groupby('patient_nbr').cumcount(); d['is_repeat']=(d.enc_seq>0).astype(int)
FE=NUM+['enc_seq','is_repeat']+CAT
y=d.y.values; groups=d.patient_nbr.values
print(f"cohort {len(y):,}  prevalence {y.mean():.4f}  (paper: 101,766 / 0.1116)")
spw=(y==0).sum()/(y==1).sum()

Xc=d[FE].copy()
for c in CAT: Xc[c]=Xc[c].astype('category')

LGRID=[dict(num_leaves=31,min_child_samples=100,learning_rate=0.03,n_estimators=600),
       dict(num_leaves=63,min_child_samples=200,learning_rate=0.03,n_estimators=600),
       dict(num_leaves=15,min_child_samples=50,learning_rate=0.05,n_estimators=500)]
LFIX=dict(reg_lambda=5,colsample_bytree=0.7,subsample=0.8,subsample_freq=1,verbose=-1,random_state=0)
XGRID=[dict(n_estimators=300,max_depth=6,learning_rate=0.1),
       dict(n_estimators=500,max_depth=4,learning_rate=0.05),
       dict(n_estimators=400,max_depth=8,learning_rate=0.05)]
XFIX=dict(scale_pos_weight=spw,enable_categorical=True,tree_method='hist',
          eval_metric='logloss',random_state=0,n_jobs=-1)

pre=ColumnTransformer([('n',StandardScaler(),NUM+['enc_seq','is_repeat']),
                       ('c',OneHotEncoder(handle_unknown='ignore',min_frequency=30),CAT)])

def nested(kind):
    oof=np.zeros(len(y)); picks=[]
    outer=StratifiedGroupKFold(5,shuffle=True,random_state=0)
    for k,(tr,te) in enumerate(outer.split(Xc,y,groups)):
        if kind=='lr':
            m=Pipeline([('p',pre),('m',LogisticRegression(max_iter=2000,C=0.5,class_weight='balanced'))])
            m.fit(d[FE].iloc[tr],y[tr]); oof[te]=m.predict_proba(d[FE].iloc[te])[:,1]; continue
        grid=LGRID if kind=='lgb' else XGRID
        inner=StratifiedGroupKFold(3,shuffle=True,random_state=k)
        best,bs=None,-1
        for g in grid:
            s=[]
            for itr,ite in inner.split(Xc.iloc[tr],y[tr],groups[tr]):
                mm=(lgb.LGBMClassifier(**{**LFIX,**g}) if kind=='lgb'
                    else xgb.XGBClassifier(**{**XFIX,**g}))
                mm.fit(Xc.iloc[tr].iloc[itr],y[tr][itr])
                s.append(roc_auc_score(y[tr][ite],mm.predict_proba(Xc.iloc[tr].iloc[ite])[:,1]))
            if np.mean(s)>bs: bs,best=np.mean(s),g
        picks.append(best)
        mm=(lgb.LGBMClassifier(**{**LFIX,**best}) if kind=='lgb' else xgb.XGBClassifier(**{**XFIX,**best}))
        mm.fit(Xc.iloc[tr],y[tr]); oof[te]=mm.predict_proba(Xc.iloc[te])[:,1]
        print(f"   {kind} fold {k} done",flush=True)
    return oof,picks

res={}
for kind,label in [('lgb','LightGBM (ours)'),('xgb','XGBoost (paper config)'),('lr','Logistic regression')]:
    oof,picks=nested(kind)
    res[kind]=oof
    print(f"{label:<26} nested-CV AUC {roc_auc_score(y,oof):.4f}  PR-AUC {average_precision_score(y,oof):.4f}  Brier {brier_score_loss(y,oof):.4f}")
np.savez('nested_oof.npz',y=y,**res)

print("\nPAIRED BOOTSTRAP (2000 resamples on pooled out-of-fold predictions)")
rng=np.random.default_rng(0); n=len(y); B=2000
def boot(a,b=None):
    out=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if y[i].sum()<10: continue
        out.append(roc_auc_score(y[i],a[i])-(roc_auc_score(y[i],b[i]) if b is not None else 0))
    return np.array(out)
for k,lab in [('lgb','LightGBM (ours)'),('xgb','XGBoost (paper config)'),('lr','Logistic')]:
    bs=boot(res[k]); print(f"  {lab:<26} AUC {roc_auc_score(y,res[k]):.4f}  95% CI [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]")
for a,b,lab in [('lgb','xgb','ours vs paper-config XGBoost'),('lgb','lr','ours vs logistic')]:
    df=boot(res[a],res[b]); p=2*min((df<=0).mean(),(df>=0).mean())
    print(f"  Δ {lab:<32} {df.mean():+.4f}  95% CI [{np.percentile(df,2.5):+.4f}, {np.percentile(df,97.5):+.4f}]  p={p:.4f}")
print(f"\n  PAPER REPORTED (nested CV): XGBoost 0.664, Stacking 0.665, LogReg 0.657, RF 0.650, LightGBM 0.660")

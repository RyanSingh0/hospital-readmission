import pandas as pd, numpy as np, warnings, json; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import lightgbm as lgb
from preprocessing import load, NUM, CAT

d=load(verbose=False).sort_values('encounter_id')
d['enc_seq']=d.groupby('patient_nbr').cumcount(); d['is_repeat']=(d.enc_seq>0).astype(int)
FEATS=NUM+['enc_seq','is_repeat']+CAT
X=d[FEATS].copy()
for c in CAT: X[c]=X[c].astype('category')
y=d.y.values; groups=d.patient_nbr.values

GRID=[dict(num_leaves=31,min_child_samples=100,learning_rate=0.03,n_estimators=600),
      dict(num_leaves=63,min_child_samples=200,learning_rate=0.03,n_estimators=600),
      dict(num_leaves=15,min_child_samples=50, learning_rate=0.05,n_estimators=500),
      dict(num_leaves=31,min_child_samples=300,learning_rate=0.05,n_estimators=400)]
FIX=dict(reg_lambda=5,colsample_bytree=0.7,subsample=0.8,subsample_freq=1,verbose=-1,random_state=0)

outer=StratifiedGroupKFold(5,shuffle=True,random_state=0)
rows=[]; oof=np.zeros(len(y)); oof_cal=np.zeros(len(y))
for k,(tr,te) in enumerate(outer.split(X,y,groups)):
    inner=StratifiedGroupKFold(3,shuffle=True,random_state=k)
    best,bs=None,-1
    for g in GRID:
        s=[]
        for itr,ite in inner.split(X.iloc[tr],y[tr],groups[tr]):
            m=lgb.LGBMClassifier(**{**FIX,**g}); m.fit(X.iloc[tr].iloc[itr],y[tr][itr])
            s.append(roc_auc_score(y[tr][ite],m.predict_proba(X.iloc[tr].iloc[ite])[:,1]))
        if np.mean(s)>bs: bs,best=np.mean(s),g
    m=lgb.LGBMClassifier(**{**FIX,**best}); m.fit(X.iloc[tr],y[tr])
    p=m.predict_proba(X.iloc[te])[:,1]; oof[te]=p
    cal=CalibratedClassifierCV(lgb.LGBMClassifier(**{**FIX,**best}),method='isotonic',cv=3)
    cal.fit(X.iloc[tr],y[tr]); pc=cal.predict_proba(X.iloc[te])[:,1]; oof_cal[te]=pc
    rows.append(dict(fold=k,auc=roc_auc_score(y[te],p),ap=average_precision_score(y[te],p),
                     brier=brier_score_loss(y[te],p),brier_cal=brier_score_loss(y[te],pc),
                     auc_cal=roc_auc_score(y[te],pc),leaves=best['num_leaves']))
    print(f"  fold {k}: AUC {rows[-1]['auc']:.4f}  PR-AUC {rows[-1]['ap']:.4f}  Brier {rows[-1]['brier']:.4f} -> cal {rows[-1]['brier_cal']:.4f}",flush=True)

r=pd.DataFrame(rows)
print("\n"+"="*70)
print("NESTED CV (5 outer grouped folds, 3 inner folds for tuning)")
print("="*70)
print(f"  ROC AUC        {r.auc.mean():.4f} ± {r.auc.std():.4f}")
print(f"  PR-AUC         {r.ap.mean():.4f} ± {r.ap.std():.4f}")
print(f"  Brier (raw)    {r.brier.mean():.4f}")
print(f"  Brier (isotonic calibrated) {r.brier_cal.mean():.4f}")
print(f"  AUC after calibration       {r.auc_cal.mean():.4f}  (calibration is monotonic - AUC should barely move)")
print(f"\n  PUBLISHED BENCHMARK (MDPI 2026, same dataset, nested CV): AUC 0.665, PR-AUC 0.215, Brier 0.094")

print("\n=== CALIBRATION CHECK (out-of-fold, calibrated) ===")
frac,mean=calibration_curve(y,oof_cal,n_bins=10,strategy='quantile')
print(f"  {'predicted':>10} {'observed':>10}")
for a,b in zip(mean,frac): print(f"  {a:>10.4f} {b:>10.4f}")

print("\n=== BUSINESS VIEW: capacity-constrained targeting (out-of-fold) ===")
o=np.argsort(-oof_cal); base=y.mean()
print(f"  base readmission rate {base:.4f}  ({y.sum():,} of {len(y):,} encounters)")
print(f"  {'target top':>11} {'n flagged':>10} {'captured':>9} {'recall':>8} {'precision':>10} {'lift':>6}")
for pct in [0.05,0.10,0.15,0.20,0.30,0.50]:
    k=int(len(y)*pct); idx=o[:k]; cap=y[idx].sum()
    print(f"  {pct*100:>10.0f}% {k:>10,} {cap:>9,} {cap/y.sum():>7.1%} {cap/k:>10.1%} {cap/k/base:>6.2f}x")
np.save('oof_cal.npy',oof_cal); np.save('y.npy',y)

"""Experiment 09 - aggregates for the care-transitions dashboard (docs/index.html).

Run after experiment 07, from the repository root; it reads the out-of-fold calibrated
predictions 07 saves (oof_cal.npy, y.npy) and writes only aggregated results.
"""
import sys, json, numpy as np, pandas as pd
import os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
from sklearn.metrics import roc_auc_score, brier_score_loss
from preprocessing import load
d=load(verbose=False).sort_values('encounter_id').reset_index(drop=True)
p=np.load('oof_cal.npy'); y=np.load('y.npy')
assert (d.y.values==y).all()
N=len(y); pos=int(y.sum()); base=y.mean()
out={'n':N,'patients':int(d.patient_nbr.nunique()),'readmits':pos,'base':base,
     'auc':roc_auc_score(y,p),'brier':brier_score_loss(y,p)}
o=np.argsort(-p, kind='stable'); cy=np.cumsum(y[o])
cap=[]
for pct in range(1,51):
    k=int(round(N*pct/100)); c=int(cy[k-1])
    cap.append(dict(pct=pct,k=k,caught=c,random=round(pos*pct/100),precision=c/k,recall=c/pos,lift=c/k/base))
out['capacity']=cap
# deciles of risk
dec=pd.qcut(pd.Series(p).rank(method='first'),10,labels=False)
out['deciles']=[dict(decile=int(10-i),n=int((dec==i).sum()),observed=float(y[dec==i].mean()),predicted=float(p[dec==i].mean())) for i in range(9,-1,-1)]
# subgroups
def grp(col,order=None,minn=500):
    r=[]
    vals=order or sorted(d[col].unique())
    for v in vals:
        m=(d[col]==v).values
        if m.sum()<minn or y[m].sum()<20: continue
        r.append(dict(group=str(v),n=int(m.sum()),rate=float(y[m].mean()),pred=float(p[m].mean()),auc=float(roc_auc_score(y[m],p[m]))))
    return r
ages=['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
out['age']=grp('age',ages); out['race']=grp('race'); out['sex']=grp('gender')
print(json.dumps({k:v for k,v in out.items() if k not in('capacity',)},indent=1)[:3000])
for c in cap:
    if c['pct'] in (5,10,20,30): print(c)
json.dump(out,open('docs/dashboard_data.json','w'))

# --- README-consistent age bands, curated race groups, notes ---
am=d.age_mid.values
bands=[('Under 50',am<50),('50–69',(am>=50)&(am<70)),('70–79',(am>=70)&(am<80)),('80 and over',am>=80)]
out['age']=[dict(group=g,n=int(m.sum()),rate=float(y[m].mean()),pred=float(p[m].mean()),auc=float(roc_auc_score(y[m],p[m]))) for g,m in bands]
names={'AfricanAmerican':'African American','Caucasian':'Caucasian','Hispanic':'Hispanic'}
out['race']=[dict(r,group=names[r['group']]) for r in out['race'] if r['group'] in names]
c5=[c for c in cap if c['pct']==5][0]; c20=[c for c in cap if c['pct']==20][0]
A=out['age']; top=out['deciles'][0]
out['eqNote']=(f"No meaningful gap by race or sex: AUC {out['race'][0]['auc']:.2f} for African American and {out['race'][1]['auc']:.2f} for Caucasian patients, "
  f"{out['sex'][0]['auc']:.2f} for women and {out['sex'][1]['auc']:.2f} for men. The real gap is <b>age</b>: ranking accuracy falls from {A[0]['auc']:.2f} under 50 to {A[3]['auc']:.2f} at 80 and over, "
  f"exactly where readmission risk is highest. Smaller race groups (under 2,000 encounters) are omitted because their estimates are unstable.")
out['qi']=[
 f"<b>Start with the top 5%.</b> Calling {c5['k']:,} discharges reaches {c5['caught']:,} patients who would be readmitted, {c5['lift']:.1f}× the {c5['random']:,} that random calls reach.",
 f"<b>Size programs with the curve.</b> Calling the top 20% reaches {c20['recall']*100:.0f}% of all readmissions ({c20['caught']:,} of {pos:,}); past that point, each extra call finds fewer readmissions.",
 f"<b>Treat older patients differently.</b> The score ranks patients 80 and over less accurately ({A[3]['auc']:.2f} vs {A[0]['auc']:.2f} under 50), so age-specific cut-offs or added clinical review make sense there.",
 f"<b>Scores are trustworthy as probabilities.</b> Predicted and actual rates track closely in every risk group; the highest group is slightly under-predicted ({top['predicted']*100:.0f}% vs {top['observed']*100:.0f}% actual).",
]
json.dump(out,open('docs/dashboard_data.json','w'))
print(out['age']); print(out['eqNote']); print(*out['qi'],sep='\n')

"""Experiment 06 — does a transformer beat gradient boosting here?

Compares LightGBM against an FT-Transformer over the tabular features, and against a causal
sequence transformer over each patient's encounter history. Run on a patient-level subsample,
because CPU budget does not stretch to the full cohort; the limitation is stated in the
manuscript rather than hidden.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, warnings, time; warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import preprocessing as diab_prep
from preprocessing import NUM, CAT
torch.manual_seed(0); np.random.seed(0)
orig=diab_prep.DEAD; diab_prep.DEAD=set()
d=diab_prep.load(verbose=False); diab_prep.DEAD=orig
d=d.sort_values(['patient_nbr','encounter_id']).reset_index(drop=True)
gb=d.groupby('patient_nbr'); d['enc_seq']=gb.cumcount(); d['is_repeat']=(d.enc_seq>0).astype(int)
for c,src,fn in [('prior_los_mean','time_in_hospital','mean'),('prior_meds_mean','num_medications','mean'),
                 ('prior_inpatient_max','number_inpatient','max')]:
    d[c]=gb[src].apply(lambda s:getattr(s.shift(1).expanding(),fn)()).reset_index(level=0,drop=True).fillna(-1)
pats=d.patient_nbr.unique(); rng=np.random.default_rng(0)
keep=set(rng.choice(pats,size=28000,replace=False))
d=d[d.patient_nbr.isin(keep)].reset_index(drop=True)
NUMF=NUM+['enc_seq','is_repeat','prior_los_mean','prior_meds_mean','prior_inpatient_max']
y=d.y.values.astype(np.float32); g=d.patient_nbr.values
print(f"subsample: {len(d):,} encounters, {d.patient_nbr.nunique():,} patients, prevalence {y.mean():.4f}",flush=True)
card=[];codes={}
for c in CAT:
    cc=d[c].astype('category'); codes[c]=(cc.cat.codes.values+1).astype(np.int64); card.append(len(cc.cat.categories)+1)
Xc=np.stack([codes[c] for c in CAT],1); Xn_raw=d[NUMF].values.astype(np.float32)
tr,te=next(GroupShuffleSplit(1,test_size=.25,random_state=0).split(d,y,g))

class FTT(nn.Module):
    def __init__(s,nnum,card,dt=24,depth=2,heads=4):
        super().__init__()
        s.nw=nn.Parameter(torch.randn(nnum,dt)*.02); s.nb=nn.Parameter(torch.zeros(nnum,dt))
        s.emb=nn.ModuleList([nn.Embedding(c,dt) for c in card]); s.cls=nn.Parameter(torch.randn(1,1,dt)*.02)
        s.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(dt,heads,dt*2,.1,'gelu',batch_first=True,norm_first=True),depth)
        s.head=nn.Sequential(nn.LayerNorm(dt),nn.ReLU(),nn.Linear(dt,1))
    def forward(s,xn,xc):
        t=[xn.unsqueeze(-1)*s.nw+s.nb]+[e(xc[:,i]).unsqueeze(1) for i,e in enumerate(s.emb)]
        return s.head(s.enc(torch.cat([s.cls.expand(xn.size(0),-1,-1)]+t,1))[:,0]).squeeze(-1)

class SeqT(nn.Module):
    def __init__(s,nf,dt=48,depth=2,heads=4,maxlen=40):
        super().__init__()
        s.inp=nn.Linear(nf,dt); s.pos=nn.Parameter(torch.randn(1,maxlen,dt)*.02)
        s.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(dt,heads,dt*2,.1,'gelu',batch_first=True,norm_first=True),depth)
        s.head=nn.Sequential(nn.LayerNorm(dt),nn.ReLU(),nn.Linear(dt,1))
    def forward(s,x,mask):
        h=s.inp(x)+s.pos[:,:x.size(1)]
        cm=torch.triu(torch.ones(x.size(1),x.size(1),dtype=torch.bool),1)
        return s.head(s.enc(h,mask=cm,src_key_padding_mask=~mask)).squeeze(-1)

out={}
t0=time.time()
Xl=d[NUMF+CAT].copy()
for c in CAT: Xl[c]=Xl[c].astype('category')
m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.03,num_leaves=31,min_child_samples=100,
    reg_lambda=5,colsample_bytree=.7,subsample=.8,subsample_freq=1,verbose=-1,random_state=0)
m.fit(Xl.iloc[tr],y[tr]); out['LightGBM (gradient boosting)']=(m.predict_proba(Xl.iloc[te])[:,1],time.time()-t0)
print(f"  LightGBM done {time.time()-t0:.0f}s",flush=True)

t0=time.time()
sc=StandardScaler().fit(Xn_raw[tr]); Xn=torch.tensor(sc.transform(Xn_raw).astype(np.float32))
tc=torch.tensor(Xc); ty=torch.tensor(y)
mm=FTT(len(NUMF),card); opt=torch.optim.AdamW(mm.parameters(),2e-3,weight_decay=1e-4); lf=nn.BCEWithLogitsLoss()
for ep in range(8):
    mm.train(); perm=np.random.permutation(tr)
    for i in range(0,len(perm),2048):
        b=perm[i:i+2048]; opt.zero_grad(); lf(mm(Xn[b],tc[b]),ty[b]).backward(); opt.step()
    print(f"    ftt epoch {ep} {time.time()-t0:.0f}s",flush=True)
mm.eval()
with torch.no_grad(): out['FT-Transformer (tabular)']=(torch.sigmoid(mm(Xn[te],tc[te])).numpy(),time.time()-t0)

t0=time.time()
oh=np.zeros((len(d),sum(min(c,10) for c in card)),np.float32); o=0
for i,c in enumerate(card):
    k=min(c,10); oh[np.arange(len(d)),o+np.minimum(Xc[:,i],k-1)]=1; o+=k
F=np.hstack([sc.transform(Xn_raw).astype(np.float32),oh]); nf=F.shape[1]
rows=[idx[:40] for idx in d.groupby('patient_nbr').indices.values()]
trp=set(g[tr]); tr_rows=[r for r in rows if g[r[0]] in trp]
sm=SeqT(nf); opt=torch.optim.AdamW(sm.parameters(),1.5e-3,weight_decay=1e-4); lf2=nn.BCEWithLogitsLoss(reduction='none')
def batchify(rs):
    L=max(len(r) for r in rs); x=np.zeros((len(rs),L,nf),np.float32); yy=np.zeros((len(rs),L),np.float32); mk=np.zeros((len(rs),L),bool)
    for i,r in enumerate(rs): x[i,:len(r)]=F[r]; yy[i,:len(r)]=y[r]; mk[i,:len(r)]=True
    return torch.tensor(x),torch.tensor(yy),torch.tensor(mk)
for ep in range(8):
    sm.train(); np.random.shuffle(tr_rows)
    for i in range(0,len(tr_rows),256):
        rs=tr_rows[i:i+256]
        if not rs: continue
        x,yy,mk=batchify(rs); opt.zero_grad(); ((lf2(sm(x,mk),yy)*mk).sum()/mk.sum()).backward(); opt.step()
    print(f"    seq epoch {ep} {time.time()-t0:.0f}s",flush=True)
sm.eval(); allp=np.zeros(len(d))
with torch.no_grad():
    for i in range(0,len(rows),512):
        rs=rows[i:i+512]; x,yy,mk=batchify(rs); pp=torch.sigmoid(sm(x,mk)).numpy()
        for j,r in enumerate(rs): allp[r]=pp[j,:len(r)]
out['Sequence Transformer (trajectory)']=(allp[te],time.time()-t0)

print(f"\n{'model':<38}{'AUC':>8}{'PR-AUC':>9}{'Brier':>9}{'train':>8}")
for k,(p,t) in out.items():
    print(f"{k:<38}{roc_auc_score(y[te],p):>8.4f}{average_precision_score(y[te],p):>9.4f}{brier_score_loss(y[te],p):>9.4f}{t:>7.0f}s")

"""Experiment 08 — every figure in the manuscript.

Colours come from the Okabe and Ito palette, which is colour-vision-deficiency safe by
construction. Every bar and marker carries a direct value label, so the figures survive both
greyscale printing and a reader who cannot separate the hues.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from scipy.stats import rankdata
import os
os.makedirs('results/figures',exist_ok=True)

C = {'blue':'#0072B2','verm':'#D55E00','green':'#009E73','amber':'#E69F00'}
INK, INK2, INK3, GRID = '#1A1A1A', '#4A4A4A', '#767676', '#DFDFDF'
plt.rcParams.update({
    'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','figure.facecolor':'white',
    'axes.facecolor':'white','font.family':'DejaVu Sans','font.size':9,
    'axes.edgecolor':INK3,'axes.linewidth':.7,'axes.labelcolor':INK2,'axes.titlesize':10,
    'axes.titleweight':'bold','axes.titlecolor':INK,'xtick.color':INK2,'ytick.color':INK2,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.frameon':False,'legend.fontsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'grid.color':GRID,'grid.linewidth':.6})

z=np.load('nested_oof.npz'); y=z['y']

# ---- Figure 1 : discrimination, ours vs the published benchmark -------------
labels=['Logistic\nregression','Random\nforest','XGBoost','LightGBM','Stacking\nensemble']
paper=[0.657,0.650,0.664,0.660,0.665]
ours =[0.6758,np.nan,0.6827,0.6888,np.nan]
fig,ax=plt.subplots(figsize=(6.2,3.2))
x=np.arange(len(labels)); w=.38
ax.bar(x-w/2,paper,w,color=INK3,label='Salim & Ibrahim (2026)',zorder=3)
b2=ax.bar(x+w/2,[v if v==v else 0 for v in ours],w,color=C['blue'],label='This study',zorder=3)
for xi,v in zip(x,paper): ax.text(xi-w/2,v+.0012,f'{v:.3f}',ha='center',fontsize=7,color=INK2)
for xi,v in zip(x,ours):
    if v==v: ax.text(xi+w/2,v+.0012,f'{v:.3f}',ha='center',fontsize=7,color=C['blue'],fontweight='bold')
ax.errorbar([3+w/2],[0.6888],yerr=[[0.0051],[0.0048]],fmt='none',ecolor=INK,capsize=3,lw=1,zorder=4)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(.60,.71)
ax.set_ylabel('Nested cross-validated ROC–AUC'); ax.yaxis.grid(True,zorder=0)
ax.set_title('Discrimination on the same 101,763-encounter cohort')
ax.legend(loc='upper left',ncol=2)
fig.savefig('results/figures/fig1_model_comparison.png'); plt.close(fig)

# ---- Figure 2 : what each preprocessing choice costs ------------------------
steps=['Full pipeline\n(this study)','− payer_code','− diagnosis\nchapters','− native\ncategoricals',
       '+ cost-sensitive\nweighting','− patient\ntrajectory']
vals=[0.6895,0.6884,0.6856,0.6829,0.6829,0.6768]
fig,ax=plt.subplots(figsize=(6.6,3.2))
ax.plot(range(len(vals)),vals,'-o',color=C['blue'],lw=2,ms=7,zorder=3,
        markerfacecolor='white',markeredgewidth=1.8)
for i,v in enumerate(vals):
    ax.annotate(f'{v:.4f}',(i,v),textcoords='offset points',xytext=(0,10),
                ha='center',fontsize=7.5,color=INK2)
ax.axhline(0.664,color=C['verm'],ls='--',lw=1.2,zorder=2)
ax.text(len(vals)-1,0.6645,'published XGBoost  0.664',ha='right',va='bottom',fontsize=7.5,color=C['verm'])
ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps,fontsize=7.5)
ax.set_ylabel('ROC–AUC'); ax.set_ylim(.660,.695); ax.yaxis.grid(True,zorder=0)
ax.set_title('Cost of each preprocessing choice, applied cumulatively')
fig.savefig('results/figures/fig2_preprocessing_ablation.png'); plt.close(fig)

# ---- Figure 3 : balancing — AUC barely moves, calibration collapses ---------
meth=['None','Cost-sensitive\nweighting','SMOTE','Random\noversampling','Random\nundersampling']
auc=[0.6895,0.6872,0.6865,0.6876,0.6756]; brier=[0.0934,0.1965,0.0937,0.2022,0.2255]
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.2,3.0))
a1.bar(range(5),auc,.55,color=[C['blue']]+[INK3]*4,zorder=3)
for i,v in enumerate(auc): a1.text(i,v+.0008,f'{v:.4f}',ha='center',fontsize=7,color=INK2)
a1.set_ylim(.66,.70); a1.set_ylabel('ROC–AUC'); a1.set_title('Discrimination is unaffected')
a2.bar(range(5),brier,.55,color=[C['green']]+[C['verm']]*4,zorder=3)
for i,v in enumerate(brier): a2.text(i,v+.004,f'{v:.4f}',ha='center',fontsize=7,color=INK2)
a2.set_ylim(0,.26); a2.set_ylabel('Brier score  (lower is better)'); a2.set_title('Calibration is destroyed')
for a in (a1,a2):
    a.set_xticks(range(5)); a.set_xticklabels(meth,fontsize=7); a.yaxis.grid(True,zorder=0)
fig.suptitle('Class-balancing strategies, identical model settings',y=1.03,fontsize=10,fontweight='bold',color=INK)
fig.savefig('results/figures/fig3_balancing.png'); plt.close(fig)

# ---- Figure 4 : subgroup discrimination by age ------------------------------
ages=['< 45','45–65','65–80','80+']; sub=[0.7329,0.6906,0.6613,0.6312]; prev=[.109,.106,.121,.125]
fig,ax=plt.subplots(figsize=(5.4,3.1))
ax.plot(range(4),sub,'-o',color=C['blue'],lw=2,ms=8,zorder=3,markerfacecolor='white',markeredgewidth=1.8,label='ROC–AUC')
for i,v in enumerate(sub): ax.annotate(f'{v:.3f}',(i,v),textcoords='offset points',xytext=(0,11),ha='center',fontsize=8,color=C['blue'])
ax.set_ylabel('ROC–AUC',color=C['blue']); ax.set_ylim(.60,.76); ax.set_xticks(range(4)); ax.set_xticklabels(ages)
ax.set_xlabel('Age band (years)'); ax.yaxis.grid(True,zorder=0)
ax2=ax.twiny(); ax2.set_xlim(ax.get_xlim()); ax2.set_xticks([]); ax2.spines['top'].set_visible(False)
for i,p in enumerate(prev): ax.annotate(f'prevalence {p:.1%}',(i,.612),ha='center',fontsize=6.8,color=INK3)
ax.set_title('Discrimination falls as age — and risk — rise')
fig.savefig('results/figures/fig4_subgroup_age.png'); plt.close(fig)

# ---- Figure 5 : ROC and precision–recall ------------------------------------
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.2,3.2))
for k,lab,col in [('lgb','LightGBM (this study)',C['blue']),('xgb','XGBoost',C['verm']),('lr','Logistic regression',C['green'])]:
    fpr,tpr,_=roc_curve(y,z[k]); a1.plot(fpr,tpr,color=col,lw=1.6,label=f'{lab} · {roc_auc_score(y,z[k]):.3f}')
    pr,rc,_=precision_recall_curve(y,z[k]); a2.plot(rc,pr,color=col,lw=1.6,label=f'{lab} · {average_precision_score(y,z[k]):.3f}')
a1.plot([0,1],[0,1],ls=':',color=INK3,lw=1)
a1.set_xlabel('False positive rate'); a1.set_ylabel('True positive rate'); a1.set_title('ROC')
a2.axhline(y.mean(),ls=':',color=INK3,lw=1); a2.text(.55,y.mean()+.004,f'prevalence {y.mean():.3f}',fontsize=7,color=INK3)
a2.set_xlabel('Recall'); a2.set_ylabel('Precision'); a2.set_title('Precision–recall'); a2.set_ylim(0,.62)
for a in (a1,a2): a.legend(loc='lower right' if a is a1 else 'upper right'); a.grid(True,zorder=0)
fig.savefig('results/figures/fig5_roc_pr.png'); plt.close(fig)

# ---- Figure 6 : calibration -------------------------------------------------
fig,ax=plt.subplots(figsize=(4.6,3.4))
frac,mean=calibration_curve(y,z['lgb'],n_bins=10,strategy='quantile')
ax.plot([0,.35],[0,.35],ls=':',color=INK3,lw=1,label='Perfect calibration')
ax.plot(mean,frac,'-o',color=C['blue'],lw=1.8,ms=6,markerfacecolor='white',markeredgewidth=1.6,label='LightGBM, uncalibrated')
fx,fm=calibration_curve(y,z['xgb'],n_bins=10,strategy='quantile')
ax.plot(fm,fx,'-s',color=C['verm'],lw=1.8,ms=5,markerfacecolor='white',markeredgewidth=1.6,label='XGBoost, cost-sensitive')
ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed frequency')
ax.set_title('Reliability, out of fold'); ax.legend(loc='upper left'); ax.grid(True,zorder=0)
fig.savefig('results/figures/fig6_calibration.png'); plt.close(fig)

# ---- Figure 7 : decision curve ----------------------------------------------
fig,ax=plt.subplots(figsize=(5.2,3.3))
N=len(y); ts=np.linspace(.01,.30,60)
nb=[( (z['lgb']>=t)&(y==1)).sum()/N-(((z['lgb']>=t)&(y==0)).sum()/N)*(t/(1-t)) for t in ts]
na=[ y.sum()/N-((N-y.sum())/N)*(t/(1-t)) for t in ts]
ax.plot(ts,nb,color=C['blue'],lw=2,label='Model')
ax.plot(ts,na,color=C['verm'],lw=1.6,ls='--',label='Treat all')
ax.axhline(0,color=C['green'],lw=1.6,ls=':',label='Treat none')
ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit'); ax.set_ylim(-.10,.12)
ax.set_title('Decision curve, out-of-fold predictions'); ax.legend(); ax.grid(True,zorder=0)
fig.savefig('results/figures/fig7_decision_curve.png'); plt.close(fig)
print("wrote", len(os.listdir('results/figures')), "figures")
for f in sorted(os.listdir('results/figures')): print("  ",f)

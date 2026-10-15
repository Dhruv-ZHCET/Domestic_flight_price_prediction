import numpy as np,pandas as pd,json,os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from pathlib import Path
D='/mnt/data/kaggle_flight_project/v3_work/Flight_Price_Project';A='/mnt/data/kaggle_flight_project/v3_work';tr=pd.read_csv('/mnt/data/train.csv');te=pd.read_csv('/mnt/data/test.csv');y=tr.price.to_numpy()
with open(D+'/weights_blend_class_specialist.json') as f: weights=json.load(f)
po=np.zeros(len(tr));pt=np.zeros(len(te))
for n in weights['Economy']:
 with np.load(D+f'/oof_{n}.npz') as z: m=z['oof'];p=z['test']
 for cls in ['Economy','Business']:
  sel=tr.flight_class.eq(cls).to_numpy();st=te.flight_class.eq(cls).to_numpy();po[sel]+=weights[cls][n]*m[sel];pt[st]+=weights[cls][n]*p[st]
strata=[]
for cls in tr.flight_class.unique():
 sel=tr.flight_class==cls;strata.append(pd.Series(cls+'_'+pd.qcut(tr.loc[sel,'price'],5,labels=False,duplicates='drop').astype(str),index=tr.index[sel]))
strata=pd.concat(strata).sort_index().to_numpy()
folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=2026).split(np.zeros(len(y)),strata))
keys=[c for c in te.columns if c not in ['id','airline','duration']]
means=np.full(len(y),np.nan);stds=np.full(len(y),np.nan);counts=np.zeros(len(y));foldid=np.zeros(len(y),int)
for k,(a,b) in enumerate(folds,1):
 tab=tr.iloc[a].groupby(keys,dropna=False).price.agg(['mean','std','count'])
 j=tr.iloc[b][keys].merge(tab,left_on=keys,right_index=True,how='left')
 means[b]=j['mean'].to_numpy();stds[b]=j['std'].fillna(0).to_numpy();counts[b]=j['count'].fillna(0).to_numpy();foldid[b]=k
m=tr.groupby(keys,dropna=False).price.agg(['mean','std','count']);jt=te[keys].merge(m,left_on=keys,right_index=True,how='left')
print('LOOKUP count distribution OOF',pd.Series(counts[counts>0]).value_counts().sort_index().to_dict());print('LOOKUP count distribution TEST',jt['count'].dropna().value_counts().sort_index().to_dict());print('matching OOF',np.sum(counts>0),'matching test',jt['count'].notna().sum())
for n in [1,2,3]:
 for lam in [.0,.25,.5,.75,1.]:
  mask=(counts>=n)&(stds<=2000)&np.isfinite(means)
  pp=po.copy();pp[mask]=po[mask]*(1-lam)+means[mask]*lam
  vals=[np.sqrt(np.mean((pp[(foldid==i)]-y[(foldid==i)])**2)) for i in [1,2,3,4,5]]
  print('lookup n',n,'blendweight',lam,'nrows',mask.sum(),'CV',round(np.sqrt(np.mean((pp-y)**2)),2),'folds',*[round(x,1) for x in vals],flush=True)
np.savez_compressed(A+'/baseline_with_lookup.npz',baseline_oof=po,baseline_test=pt,lookup_mean_oof=means,lookup_std_oof=stds,lookup_count_oof=counts,foldid=foldid,lookup_mean_test=jt['mean'].to_numpy(),lookup_std_test=jt['std'].fillna(0).to_numpy(),lookup_count_test=jt['count'].fillna(0).to_numpy())

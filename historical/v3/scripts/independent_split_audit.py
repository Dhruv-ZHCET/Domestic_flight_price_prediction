"""Fresh, separately seeded holdouts not used to choose the local smoother.
This check uses a standalone Extra Trees regressor, not the saved previous blended predictions.
"""
import os,sys
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
sys.path.insert(0,'/mnt/data/kaggle_flight_project/v3_work/Flight_Price_Project')
import modeling as m
from pathlib import Path
D=Path('/mnt/data/kaggle_flight_project/v3_work')
code=(D/'build_v3_final.py').read_text()
fn=code[code.index('def fit_group_curve('):code.index('\nmetrics=lambda')]
exec(fn)
tr=m.tr;te=m.te;y=m.y
KEY=[c for c in te.columns if c not in ['id','airline','duration','days_left']]
strata=m.strata_labels(y,m.X.flight_class)
result=[]
for k,seed in enumerate([113,399,901,1028]):
 a,b=next(StratifiedShuffleSplit(n_splits=1,test_size=.20,random_state=seed).split(np.zeros(len(y)),strata))
 model=m.make_est('extra_trees_tuned');model.fit(m.X.iloc[a],y[a]);ptrain=np.clip(model.predict(m.X.iloc[a]),1,None);p=np.clip(model.predict(m.X.iloc[b]),1,None)
 train_rmse=float(np.sqrt(np.mean((ptrain-y[a])**2)))
 initial=float(np.sqrt(np.mean((p-y[b])**2)))
 exkey=[c for c in te.columns if c not in ['id','airline','duration']]
 tab=tr.iloc[a].groupby(exkey,dropna=False).price.agg(['mean','std','count'])
 j=tr.iloc[b][exkey].merge(tab,left_on=exkey,right_index=True,how='left')
 ex=j['mean'].notna().to_numpy()&(j['std'].fillna(0).to_numpy()<=2000)
 p[ex]=j['mean'].to_numpy()[ex]
 post_exact=float(np.sqrt(np.mean((p-y[b])**2)))
 est,n,std,dist=fit_group_curve(tr.iloc[a],tr.iloc[b],penalty=150)
 mk=(~ex)&(n>=3)&(std<=800)&(dist<=0)&np.isfinite(est)
 for w in [.15,.25]:
  q=p.copy();q[mk]=(1-w)*q[mk]+w*est[mk]
  result.append(dict(seed=seed,weight=w,n_train=len(a),n_val=len(b),train_rmse=train_rmse,base_holdout_rmse=initial,exact_holdout_rmse=post_exact,smooth_holdout_rmse=float(np.sqrt(np.mean((q-y[b])**2))),modified=int(mk.sum()),improvement=float(post_exact-np.sqrt(np.mean((q-y[b])**2)))))
 print('AUDIT seed',seed,'train rmse',round(train_rmse,1),'val baseline',round(initial,1),'exact',round(post_exact,1),'smooth_15',round(result[-2]['smooth_holdout_rmse'],1),'smooth_25',round(result[-1]['smooth_holdout_rmse'],1),flush=True)
pd.DataFrame(result).to_csv(D/'independent_split_audit.csv',index=False)

import sys,os,json,time
sys.path.insert(0,'/mnt/data/kaggle_flight_project')
import compact_under500 as m
from sklearn.model_selection import GroupShuffleSplit
import numpy as np,pandas as pd
rows=[]
for fold,(tr,va) in enumerate(GroupShuffleSplit(n_splits=2,test_size=.20,random_state=18).split(m.X,m.y,groups=m.train.flight),1):
 pp={}
 for n in ['et180','lgb260','cat420','rf220']:
  p,_,_=m.fit_pred(n,tr,va);pp[n]=p
 for name,p in {**pp,'blend_et_lgb_440':.6*pp['et180']+.4*pp['lgb260'],'blend_rf_lgb_480':.5*pp['rf220']+.5*pp['lgb260']}.items():
  r={'split':fold,'model':name,'trees':440 if name=='blend_et_lgb_440' else 480 if name=='blend_rf_lgb_480' else m.CFG[name]['trees'],**m.metric(m.y[va],p)}
  rows.append(r);print(r,flush=True)
 pd.DataFrame(rows).to_csv(m.OUT+'/group_blend_checks.csv',index=False)

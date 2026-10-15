import sys, numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data/kaggle_flight_project')
import compact_under500 as m
from sklearn.model_selection import train_test_split
rows=[]
for seed in [113,399,901,1028]:
 a,b=train_test_split(np.arange(len(m.y)),test_size=.2,random_state=seed,stratify=m.strata())
 pp={}
 for name in ['et180','lgb260','cat420']:
  pp[name],_,_=m.fit_pred(name,a,b)
 pp['blend_et_lgb_440']=.6*pp['et180']+.4*pp['lgb260']
 for name,p in pp.items():
  r={'seed':seed,'model':name,**m.metric(m.y[b],p)}
  rows.append(r)
  print('FRESH',r,flush=True)
 pd.DataFrame(rows).to_csv(m.OUT+'/independent_holdouts.csv',index=False)

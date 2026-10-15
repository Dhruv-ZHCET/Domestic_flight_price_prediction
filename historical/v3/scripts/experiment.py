import os, sys, time, json
os.environ['OMP_NUM_THREADS']='3';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
sys.path.insert(0,'/mnt/data/kaggle_flight_project/v3_work/Flight_Price_Project')
import modeling as bas
FOLD=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=2026).split(np.zeros(len(bas.y)),bas.strata_labels(bas.y,bas.X.flight_class)))
BASEDIR='/mnt/data/kaggle_flight_project';OUT=BASEDIR+'/v3_work'
TR=bas.tr;TE=bas.te;y=bas.y

def feature_frame(df):
 z=bas.feats(df).copy()
 # Engineer only feature-derived combinations; never use price or sample ID.
 z['flight_class_code']=z.flight.astype(str)+'_CLS_'+z.flight_class
 z['flight_route']=z.flight.astype(str)+'_R_'+z.route
 z['flight_stop']=z.flight.astype(str)+'_S_'+z.stops
 z['route_dep_class']=z.route_class+'__'+z.departure_time
 z['route_arr_class']=z.route_class+'__'+z.arrival_time
 z['route_stop_class']=z.route_class+'__'+z.stops
 z['stop_class']=z.stops+'__'+z.flight_class
 z['duration_bin']=pd.cut(z.duration, bins=[-1,1,2,3,4,6,9,15,30,100],labels=False).fillna(-1).astype(int).astype(str)
 z['flight_duration_bin']=z.flight.astype(str)+'_D_'+z.duration_bin
 z['flight_dep']=z.flight.astype(str)+'_T_'+z.departure_time
 z['lead_week']=((z.days_left-1)//7).astype(int)
 z['lead_bin']=pd.cut(z.days_left,bins=[-1,1,3,7,14,21,30,45,60,120],labels=False).fillna(-1).astype(int).astype(str)
 z['class_lead_bin']=z.flight_class+'__'+z.lead_bin
 z['route_lead_bin']=z.route+'__'+z.lead_bin
 z['stop_duration']=z.stops.astype(str)+'__'+z.duration_bin
 z['duration_missing_class']=z.duration_missing.astype(str)+'__'+z.flight_class
 z['log_days_left']=np.log1p(z.days_left)
 z['log_duration']=np.log1p(z.duration)
 # avoid poor categorical predictions due to spurious synthetic ordering of nominal flight numbers
 return z
X=feature_frame(TR);XT=feature_frame(TE)
NEWCATS=[c for c in X.columns if X[c].dtype=='object'];NUMS=[c for c in X.columns if c not in NEWCATS]

def metric(pred,index):
 yt=y[index];return dict(rmse=round(float(np.sqrt(np.mean((pred-yt)**2))),2),mae=round(float(np.mean(abs(pred-yt))),2),r2=round(float(r2_score(yt,pred)),6),bus_rmse=round(float(np.sqrt(np.mean((pred[TR.flight_class.iloc[index].values=='Business']-yt[TR.flight_class.iloc[index].values=='Business'])**2))),2))

def get_model(name):
 if name=='cb_simple':return CatBoostRegressor(iterations=1050,depth=6,learning_rate=.055,l2_leaf_reg=12,random_strength=1.2,loss_function='RMSE',random_seed=90,thread_count=3,verbose=False,allow_writing_files=False)
 if name=='cb_features':return CatBoostRegressor(iterations=1250,depth=7,learning_rate=.043,l2_leaf_reg=9,random_strength=1.8,loss_function='RMSE',random_seed=90,thread_count=3,verbose=False,allow_writing_files=False)
 if name=='cb_deep':return CatBoostRegressor(iterations=950,depth=8,learning_rate=.045,l2_leaf_reg=12,random_strength=2,loss_function='RMSE',random_seed=90,thread_count=3,verbose=False,allow_writing_files=False)
 if name=='lgb_simple':return LGBMRegressor(n_estimators=700,learning_rate=.045,num_leaves=15,max_depth=7,min_child_samples=65,subsample=.84,subsample_freq=1,colsample_bytree=.81,reg_lambda=15,reg_alpha=1,max_bin=191,verbosity=-1,n_jobs=3,random_state=90)
 if name=='lgb_features':return LGBMRegressor(n_estimators=900,learning_rate=.035,num_leaves=22,max_depth=8,min_child_samples=45,subsample=.84,subsample_freq=1,colsample_bytree=.83,reg_lambda=15,reg_alpha=1,verbosity=-1,n_jobs=3,random_state=90)
 if name=='extra_reg':return ExtraTreesRegressor(n_estimators=280,max_depth=23,min_samples_leaf=5,min_samples_split=12,max_features=.76,n_jobs=3,random_state=90)
 if name=='extra_more':return ExtraTreesRegressor(n_estimators=340,max_depth=30,min_samples_leaf=3,min_samples_split=9,max_features=.78,n_jobs=3,random_state=90)
 raise KeyError(name)

def fit(name,train_indices,valid_indices=None, test=False, oldfeatures=False):
 a=(bas.X if oldfeatures else X).iloc[train_indices].copy()
 b=(bas.X if oldfeatures else X).iloc[valid_indices].copy() if valid_indices is not None else None
 t=(bas.XT if oldfeatures else XT).copy() if test else None
 cats=bas.CATS if oldfeatures else NEWCATS
 reg=get_model(name)
 if name.startswith('cb'):
  reg.fit(a,y[train_indices],cat_features=cats)
 elif name.startswith('lgb'):
  catlist=[]
  for c in cats:
   ctype=pd.CategoricalDtype(categories=pd.Index(a[c].unique()),ordered=False)
   for d in (a,b,t):
    if d is not None:d[c]=d[c].astype(ctype)
  reg.fit(a,y[train_indices],categorical_feature=cats)
 else:
  pre=ColumnTransformer([('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),cats),('num',SimpleImputer(strategy='median'),[col for col in a.columns if col not in cats])],sparse_threshold=0)
  reg=make_pipeline(pre,reg);reg.fit(a,y[train_indices])
 p=np.clip(reg.predict(b),1,None) if b is not None else None
 q=np.clip(reg.predict(t),1,None) if t is not None else None
 return p,q,reg

def phase1():
 i,j=FOLD[0]; names=['cb_simple','cb_features','cb_deep','lgb_simple','lgb_features','extra_reg','extra_more']
 results=[]
 for n in names:
  st=time.time()
  try:
   p,_,m=fit(n,i,j)
   mm=metric(p,j)
   for k,v in mm.items():pass
   np.savez_compressed(OUT+f'/onefold_{n}.npz',val_idx=j,pred=p)
   result=dict(model=n,**mm,seconds=round(time.time()-st,1))
   print(result,flush=True);results.append(result)
  except Exception as e: print('ERROR',n,str(e),flush=True)
 pd.DataFrame(results).sort_values('rmse').to_csv(OUT+'/onefold_benchmarks.csv',index=False)

def five(names):
 for n in names:
  po=np.empty(len(y));pr=[];foldrows=[]
  for k,(i,j) in enumerate(FOLD,1):
   st=time.time();p,q,_=fit(n,i,j,test=True)
   po[j]=p;pr.append(q);row={'fold':k,'model':n,**metric(p,j),'seconds':round(time.time()-st,1)}
   print(row,flush=True);foldrows.append(row)
  t=np.mean(pr,axis=0);np.savez_compressed(OUT+f'/oof_{n}.npz',oof=po,test=t)
  pd.DataFrame(foldrows).to_csv(OUT+f'/fivefold_{n}.csv',index=False)
  print('CV FINISHED',n, metric(po,np.arange(len(y))),flush=True)

def group_test(names):
 a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=18).split(X,y,groups=TR.flight))
 for n in names:
  st=time.time();p,_,mdl=fit(n,a,b)
  p_train=np.clip(mdl.predict(X.iloc[a]),1,None)
  print('GROUP HOLDOUT',n,'train',metric(p_train,a),'valid',metric(p,b),'seconds',round(time.time()-st,1),flush=True)

if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='onefold':phase1()
 elif mode=='five':five(sys.argv[2:])
 elif mode=='group':group_test(sys.argv[2:])

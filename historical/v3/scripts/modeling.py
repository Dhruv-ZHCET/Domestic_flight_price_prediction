"""Leakage-safe Kaggle flight-price baseline, validation, tuning, OOF ensemble, and submissions.
Usage: python modeling.py benchmark ; python modeling.py final
All target-informed fitting is performed inside the training partition. Test is prediction-only.
"""
import os, sys, json, time, warnings, copy
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_log_error
from sklearn.inspection import permutation_importance
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from joblib import dump
ROOT='/mnt/data'; OUT=ROOT+'/kaggle_flight_project'; os.makedirs(OUT,exist_ok=True)
CATS=['flight','flight_prefix','source_city','destination_city','departure_time','arrival_time','stops','flight_class','route','route_class','time_pair','prefix_class','stops_route']
NUM=['duration','days_left','flight_number','duration_missing','airline_missing','days_left_sq','days_left_inv','duration_per_stop','stops_num']
tr=pd.read_csv(ROOT+'/train.csv'); te=pd.read_csv(ROOT+'/test.csv'); sub=pd.read_csv(ROOT+'/sample_submission.csv')
assert tr.price.notna().all() and te.columns.tolist()==tr.columns.drop('price').tolist()
assert sub.id.equals(te.id) and not te.id.duplicated().any()
y=tr.price.to_numpy(float)

def feats(x):
 x=x.copy()
 x['flight_prefix']=x.flight.str.extract(r'^([^ ]+)')[0].fillna('UNK')
 x['flight_number']=pd.to_numeric(x.flight.str.extract(r'(\d+)\s*$')[0],errors='coerce')
 x['duration_missing']=x.duration.isna().astype(int)
 x['airline_missing']=x.airline.isna().astype(int)
 x['route']=x.source_city.astype(str)+'__'+x.destination_city.astype(str)
 x['route_class']=x['route']+'__'+x.flight_class.astype(str)
 x['time_pair']=x.departure_time.astype(str)+'__'+x.arrival_time.astype(str)
 x['prefix_class']=x.flight_prefix+'__'+x.flight_class.astype(str)
 x['stops_route']=x.stops.astype(str)+'__'+x.route
 x['stops_num']=x.stops.map({'zero':0,'one':1,'two_or_more':2}).astype(int)
 x['days_left_sq']=x.days_left.astype(float)**2
 x['days_left_inv']=1.0/(x.days_left.astype(float)+1)
 x['duration_per_stop']=x.duration/(x.stops_num+1)
 for c in CATS:x[c]=x[c].fillna('__UNKNOWN__').astype(str)
 return x[CATS+NUM]
X=feats(tr);XT=feats(te)

def regress_metrics(y_true,p):
 p=np.clip(np.asarray(p,dtype=float),1,None)
 return {'rmse':float(np.sqrt(mean_squared_error(y_true,p))),
         'mae':float(mean_absolute_error(y_true,p)),
         'r2':float(r2_score(y_true,p)),
         'rmsle':float(root_mean_squared_log_error(y_true,p))}

def linear_pipe(poly=False):
 num_pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())])
 if poly: num_pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('poly',PolynomialFeatures(degree=2,include_bias=False)),('scale',StandardScaler())])
 enc=ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore',min_frequency=3),CATS),('numeric',num_pipe,NUM)])
 return Pipeline([('prep',enc),('regressor',Ridge(alpha=200 if not poly else 450,solver='lsqr'))])

def ordinal_pipe(reg):
 ct=ColumnTransformer([('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),CATS),('num',SimpleImputer(strategy='median'),NUM)],sparse_threshold=0)
 return Pipeline([('prep',ct),('regressor',reg)])

def cat_frame(train,valid):
 a=train.copy();b=valid.copy()
 # categorical types for LightGBM; categories are derived solely from fold train.
 for c in CATS:
  category=pd.CategoricalDtype(categories=pd.Index(a[c].unique()),ordered=False)
  a[c]=a[c].astype(category); b[c]=b[c].astype(category)
 return a,b

def make_est(name):
 if name=='ridge':return linear_pipe()
 if name=='polynomial_ridge':return linear_pipe(poly=True)
 if name=='decision_tree':return ordinal_pipe(DecisionTreeRegressor(max_depth=18,min_samples_leaf=6,random_state=42))
 if name=='bagged_trees':return ordinal_pipe(BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=22,min_samples_leaf=4,max_features=.90,random_state=42),n_estimators=95,max_samples=.85,bootstrap=True,n_jobs=4,random_state=42))
 if name=='random_forest':return ordinal_pipe(RandomForestRegressor(n_estimators=180,max_depth=24,min_samples_leaf=3,max_features=.9,n_jobs=4,random_state=42))
 if name=='extra_trees':return ordinal_pipe(ExtraTreesRegressor(n_estimators=230,max_depth=28,min_samples_leaf=2,max_features=.94,n_jobs=4,random_state=42))
 if name=='extra_trees_tuned':return ordinal_pipe(ExtraTreesRegressor(n_estimators=260,max_depth=None,min_samples_leaf=3,max_features=.85,n_jobs=4,random_state=42))
 if name=='hist_gradient':return ordinal_pipe(HistGradientBoostingRegressor(max_iter=370,learning_rate=.06,max_leaf_nodes=27,l2_regularization=5,min_samples_leaf=25,random_state=42))
 if name=='lightgbm':return LGBMRegressor(n_estimators=850,learning_rate=.045,num_leaves=23,max_depth=-1,min_child_samples=35,subsample=.85,subsample_freq=1,colsample_bytree=.88,reg_lambda=7,reg_alpha=.1,verbosity=-1,n_jobs=4,random_state=42)
 if name=='lightgbm_deep':return LGBMRegressor(n_estimators=1050,learning_rate=.035,num_leaves=40,min_child_samples=24,subsample=.82,subsample_freq=1,colsample_bytree=.9,reg_lambda=12,verbosity=-1,n_jobs=4,random_state=42)
 if name=='xgboost':return ordinal_pipe(XGBRegressor(n_estimators=900,max_depth=6,learning_rate=.045,subsample=.84,colsample_bytree=.92,min_child_weight=10,reg_lambda=12,objective='reg:squarederror',tree_method='hist',n_jobs=4,random_state=42))
 if name=='catboost':return CatBoostRegressor(iterations=1150,depth=7,learning_rate=.045,l2_leaf_reg=6,loss_function='RMSE',random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
 if name=='catboost_deep':return CatBoostRegressor(iterations=1650,depth=8,learning_rate=.035,l2_leaf_reg=8,loss_function='RMSE',random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
 raise ValueError(name)

def fit_predict(name,train_i,valid_i,full_test=None,return_fit=False,extras=None):
 a=X.iloc[train_i];b=X.iloc[valid_i] if valid_i is not None else None
 ya=y[train_i]
 if name=='class_mean':
  mapping=pd.Series(ya).groupby(a.flight_class.to_numpy()).mean().to_dict()
  p=b.flight_class.map(mapping).fillna(np.mean(ya)).to_numpy() if b is not None else None
  q=full_test.flight_class.map(mapping).fillna(np.mean(ya)).to_numpy() if full_test is not None else None
  return (p,q,None) if return_fit else (p,q)
 mdl=make_est(name)
 if name.startswith('catboost'):
  mdl.fit(a,ya,cat_features=CATS)
  p=mdl.predict(b) if b is not None else None
  q=mdl.predict(full_test) if full_test is not None else None
 elif name.startswith('lightgbm'):
  if b is not None:a,b=cat_frame(a,b)
  else:a,_=cat_frame(a,a)
  mdl.fit(a,ya,categorical_feature=CATS)
  p=mdl.predict(b) if b is not None else None
  if full_test is not None:
   qf=full_test.copy()
   for col in CATS:qf[col]=qf[col].astype(a[col].dtype)
   q=mdl.predict(qf)
  else:q=None
 else:
  mdl.fit(a,ya)
  p=mdl.predict(b) if b is not None else None
  q=mdl.predict(full_test) if full_test is not None else None
 if p is not None:p=np.clip(p,1,None)
 if q is not None:q=np.clip(q,1,None)
 return (p,q,mdl) if return_fit else (p,q)

def strata_labels(y,klass,bins=5):
 data=pd.DataFrame({'y':y,'klass':klass})
 groups=[]
 for k in data.klass.unique():
  rows=data.index[data.klass==k]; q=pd.qcut(data.loc[rows,'y'],bins,labels=False,duplicates='drop')
  groups.append(pd.Series(k+'_'+q.astype(str).to_numpy(),index=rows))
 return pd.concat(groups).sort_index().to_numpy()

if __name__=='__main__':
 mode=sys.argv[1] if len(sys.argv)>1 else 'benchmark'
 if mode=='benchmark':
  groups=strata_labels(y,X.flight_class)
  train_idx,val_idx=train_test_split(np.arange(len(tr)),test_size=.24,random_state=42,stratify=groups)
  candidates=['class_mean','ridge','polynomial_ridge','decision_tree','bagged_trees','random_forest','extra_trees','hist_gradient','lightgbm','lightgbm_deep','xgboost','catboost']
  rows=[];pv={}
  for n in candidates:
   t=time.time()
   try:
    p,_=fit_predict(n,train_idx,val_idx)
    mt=regress_metrics(y[val_idx],p)
    for cls in ['Economy','Business']:
     mask=(X.iloc[val_idx].flight_class==cls).to_numpy()
     mt['rmse_'+cls.lower()]=float(np.sqrt(mean_squared_error(y[val_idx][mask],p[mask])))
    rows.append({'model':n,**mt,'train_seconds':round(time.time()-t,1)})
    pv[n]=p
    pd.DataFrame(rows).sort_values('rmse').to_csv(OUT+'/holdout_model_benchmarks.csv',index=False)
    print(f"{n:20s} RMSE {mt['rmse']:12,.2f} MAE {mt['mae']:10,.2f} R2 {mt['r2']:.5f} RMSLE {mt['rmsle']:.5f} [{time.time()-t:.1f}s]",flush=True)
   except Exception as ex:
    print(f'FAILED {n}: {ex}',flush=True)
  np.savez_compressed(OUT+'/holdout_predictions.npz',indices=val_idx,labels=y[val_idx],**pv)
  print('BENCHMARK DONE',flush=True)
 elif mode=='final':
  # Tune choices based on 24% holdout; report unbiased 5-fold training-only OOF.
  scores=pd.read_csv(OUT+'/holdout_model_benchmarks.csv').sort_values('rmse')
  finalists=['extra_trees','catboost','lightgbm','xgboost','random_forest','bagged_trees']
  print('FIVE FOLD FINALISTS:',finalists,flush=True)
  cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=2026)
  folds=list(cv.split(np.zeros(len(y)),strata_labels(y,X.flight_class)))
  oof={};test_predict={};detailed=[]
  for n in finalists:
   preds=np.zeros(len(y));testfold=[]
   for i,(tri,vai) in enumerate(folds):
    t=time.time();p,q=fit_predict(n,tri,vai,XT)
    preds[vai]=p;testfold.append(q)
    m=regress_metrics(y[vai],p)
    detailed.append({'model':n,'fold':i+1,**m})
    print(f'{n} fold {i+1}/5 RMSE {m["rmse"]:.2f} duration {time.time()-t:.1f}s',flush=True)
   oof[n]=preds;test_predict[n]=np.mean(testfold,axis=0)
   report=regress_metrics(y,preds)
   print('FIVE FOLD SCORE',n,report,flush=True)
   pd.DataFrame(detailed).to_csv(OUT+'/cv_by_fold.csv',index=False)
   pd.DataFrame([{'model':name,**regress_metrics(y,values)} for name,values in oof.items()]).sort_values('rmse').to_csv(OUT+'/cv_summary.csv',index=False)
   np.savez_compressed(OUT+'/fivefold_oof.npz',labels=y,**oof)
   np.savez_compressed(OUT+'/fivefold_test_predictions.npz',**test_predict)
  print('FINALISTS DONE',flush=True)
 elif mode=='single':
  # Build a single model's 5-fold OOF and averaged test submission without using test labels.
  name=sys.argv[2]
  folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=2026).split(np.zeros(len(y)),strata_labels(y,X.flight_class)))
  oof=np.zeros(len(y));foldpred=[];stats=[]
  for i,(a,b) in enumerate(folds):
   tm=time.time(); p,qt=fit_predict(name,a,b,XT)
   oof[b]=p;foldpred.append(qt)
   info={'model':name,'fold':i+1,**regress_metrics(y[b],p)}
   stats.append(info);print(name,'fold',i+1,info,'sec',round(time.time()-tm,2),flush=True)
  q=np.mean(foldpred,axis=0)
  report={'model':name,**regress_metrics(y,oof)}
  print('COMPLETE',report,flush=True)
  pd.DataFrame(stats).to_csv(f'{OUT}/cv_folds_{name}.csv',index=False)
  pd.DataFrame({'id':te.id,'price':q}).to_csv(f'{OUT}/submission_{name}.csv',index=False)
  np.savez_compressed(f'{OUT}/oof_{name}.npz',oof=oof,test=q)
 elif mode=='groupstress':
  # Unseen flight stress test, not used for hyperparameter tuning.
  split=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=18)
  a,b=next(split.split(X,y,groups=tr.flight))
  for n in ['extra_trees','lightgbm','catboost']:
   tm=time.time();p,_=fit_predict(n,a,b)
   print('GROUPED FLIGHT HOLDOUT',n,regress_metrics(y[b],p),'sec',round(time.time()-tm,2),flush=True)
 elif mode=='onefold':
  name=sys.argv[2];idx=int(sys.argv[3])-1
  folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=2026).split(np.zeros(len(y)),strata_labels(y,X.flight_class)))
  a,b=folds[idx];tm=time.time();p,q=fit_predict(name,a,b,XT)
  np.savez_compressed(f'{OUT}/{name}_fold{idx+1}.npz',valid_idx=b,pred=p,test=q)
  print(name,'fold',idx+1,regress_metrics(y[b],p),'sec',round(time.time()-tm,2),flush=True)
 elif mode=='collect':
  name=sys.argv[2];oof=np.zeros(len(y));count=np.zeros(len(y));testfold=[]
  for i in range(1,6):
   with np.load(f'{OUT}/{name}_fold{i}.npz') as z:
    oof[z['valid_idx']]=z['pred'];count[z['valid_idx']]+=1;testfold.append(z['test'])
  assert (count==1).all()
  q=np.mean(testfold,axis=0)
  pd.DataFrame({'id':te.id,'price':q}).to_csv(f'{OUT}/submission_{name}.csv',index=False)
  np.savez_compressed(f'{OUT}/oof_{name}.npz',oof=oof,test=q)
  print('COMPLETED',name,regress_metrics(y,oof),flush=True)

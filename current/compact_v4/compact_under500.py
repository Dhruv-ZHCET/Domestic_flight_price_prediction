"""Reproducible tree-budget (<500 trees per deployed model) flight-price pipeline.
No price lookups, no target-derived aggregations/interpolation, no test-label use.
"""
import os,sys,json,time,warnings
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split,GroupShuffleSplit
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

ROOT='/mnt/data'; OUT=ROOT+'/kaggle_flight_project/compact_v4';os.makedirs(OUT,exist_ok=True)
train=pd.read_csv(ROOT+'/train.csv');test=pd.read_csv(ROOT+'/test.csv');sub=pd.read_csv(ROOT+'/sample_submission.csv')
assert sub.id.equals(test.id)
y=train.price.to_numpy(dtype=float)
CATS=['flight','flight_prefix','source_city','departure_time','stops','arrival_time','destination_city','flight_class', 'route','route_class','time_pair','prefix_class','stops_route']
NUM=['duration','days_left','flight_number','duration_missing','airline_missing','days_left_sq','days_left_inv','duration_per_stop','stops_num']

def feature_frame(df):
 x=df.copy()
 x['flight_prefix']=x.flight.str.extract(r'^(\S+)')[0].fillna('UNK')
 x['flight_number']=pd.to_numeric(x.flight.str.extract(r'(\d+)\s*$')[0],errors='coerce')
 x['duration_missing']=x.duration.isna().astype('int8');x['airline_missing']=x.airline.isna().astype('int8')
 x['route']=x.source_city+'__'+x.destination_city
 x['route_class']=x.route+'__'+x.flight_class
 x['time_pair']=x.departure_time+'__'+x.arrival_time
 x['prefix_class']=x.flight_prefix+'__'+x.flight_class
 x['stops_route']=x.stops+'__'+x.route
 x['stops_num']=x.stops.map({'zero':0,'one':1,'two_or_more':2})
 x['days_left_sq']=x.days_left**2
 x['days_left_inv']=1/(x.days_left+1)
 x['duration_per_stop']=x.duration/(x.stops_num+1)
 for c in CATS: x[c]=x[c].fillna('UNKNOWN').astype(str)
 return x[CATS+NUM]
X=feature_frame(train);XT=feature_frame(test)

# Candidate names specify tree counts and bounded depths (no unlimited max-depth anywhere).
CFG={
 'rf220':dict(algo='rf',trees=220,depth=15,leaf=3,features=.8),
 'rf300':dict(algo='rf',trees=300,depth=19,leaf=3,features=.85),
 'et220':dict(algo='et',trees=220,depth=20,leaf=4,features=.85),
 'et300':dict(algo='et',trees=300,depth=24,leaf=3,features=.85),
 'et360':dict(algo='et',trees=360,depth=28,leaf=3,features=.85),
 'cat300':dict(algo='cat',trees=300,depth=6,lr=.075,l2=8),
 'cat420':dict(algo='cat',trees=420,depth=7,lr=.075,l2=10),
 'cat480':dict(algo='cat',trees=480,depth=8,lr=.065,l2=10),
 'lgb320':dict(algo='lgb',trees=320,depth=6,lr=.07,leaves=20,leaf=35,l2=10),
 'lgb440':dict(algo='lgb',trees=440,depth=8,lr=.055,leaves=23,leaf=35,l2=9),
 'lgb480':dict(algo='lgb',trees=480,depth=7,lr=.055,leaves=31,leaf=40,l2=15),
 'xgb360':dict(algo='xgb',trees=360,depth=5,lr=.07,child=12,l2=15),
 # For combined ensembles, two independently fit components sum <500
 'et180':dict(algo='et',trees=180,depth=24,leaf=3,features=.85),
 'cat280':dict(algo='cat',trees=280,depth=7,lr=.09,l2=10),
 'lgb260':dict(algo='lgb',trees=260,depth=7,lr=.085,leaves=25,leaf=35,l2=10),
 'rf180':dict(algo='rf',trees=180,depth=18,leaf=3,features=.85),
 'cat_class440':dict(algo='cat_class',trees=440,depth=7),
 'rf_class440':dict(algo='rf_class',trees=440,depth=18),
 }

def model(cfg):
 algo=cfg['algo']; seed=42
 if algo in ['rf','et']:
  kw=dict(n_estimators=cfg['trees'],max_depth=cfg['depth'],min_samples_leaf=cfg['leaf'],max_features=cfg['features'],random_state=seed,n_jobs=4)
  reg=RandomForestRegressor(**kw) if algo=='rf' else ExtraTreesRegressor(**kw)
  prep=ColumnTransformer([('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),CATS),('num',SimpleImputer(strategy='median'),NUM)],sparse_threshold=0)
  return Pipeline([('preprocess',prep),('model',reg)])
 if algo=='cat':
  return CatBoostRegressor(iterations=cfg['trees'],depth=cfg['depth'],learning_rate=cfg['lr'],l2_leaf_reg=cfg['l2'],loss_function='RMSE',random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,random_strength=1.2)
 if algo=='lgb':
  return LGBMRegressor(n_estimators=cfg['trees'],max_depth=cfg['depth'],num_leaves=cfg['leaves'],min_child_samples=cfg['leaf'],learning_rate=cfg['lr'],reg_lambda=cfg['l2'],reg_alpha=.2,subsample=.85,subsample_freq=1,colsample_bytree=.85,n_jobs=4,random_state=seed,verbosity=-1)
 if algo=='xgb':
  prep=ColumnTransformer([('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),CATS),('num',SimpleImputer(strategy='median'),NUM)],sparse_threshold=0)
  reg=XGBRegressor(n_estimators=cfg['trees'],max_depth=cfg['depth'],learning_rate=cfg['lr'],min_child_weight=cfg['child'],reg_lambda=cfg['l2'],reg_alpha=.2,subsample=.85,colsample_bytree=.85,objective='reg:squarederror',n_jobs=4,random_state=seed,tree_method='hist')
  return Pipeline([('preprocess',prep),('model',reg)])
 raise ValueError(cfg)

def prep_cats(a,b,q=None):
 a=a.copy();b=b.copy();q=q.copy() if q is not None else None
 for c in CATS:
  typ=pd.CategoricalDtype(categories=pd.Index(a[c].unique()),ordered=False)
  a[c]=a[c].astype(typ);b[c]=b[c].astype(typ)
  if q is not None:q[c]=q[c].astype(typ)
 return a,b,q

def fit_pred(name,idx_tr,idx_va,with_test=False):
 cfg=CFG[name]; assert 0<cfg['trees']<500 and cfg['depth']>0
 if cfg['algo'] in ['cat_class','rf_class']:
  p=np.empty(len(idx_va));ptr=np.empty(len(idx_tr));q=np.empty(len(XT)) if with_test else None
  for cl in ['Business','Economy']:
   ai=np.flatnonzero(train.flight_class.to_numpy()[idx_tr]==cl);bi=np.flatnonzero(train.flight_class.to_numpy()[idx_va]==cl)
   qa=np.flatnonzero(test.flight_class.to_numpy()==cl) if with_test else np.array([],dtype=int)
   if cfg['algo']=='cat_class':
    cc=dict(algo='cat',trees=220,depth=7 if cl=='Business' else 6,lr=.09,l2=10)
   else:cc=dict(algo='rf',trees=220,depth=18,leaf=3,features=.85)
   mdl=model(cc)
   aa=X.iloc[idx_tr[ai]];bb=X.iloc[idx_va[bi]]
   if cc['algo']=='cat':mdl.fit(aa,y[idx_tr[ai]],cat_features=CATS)
   else:mdl.fit(aa,y[idx_tr[ai]])
   p[bi]=np.clip(mdl.predict(bb),1,None);ptr[ai]=np.clip(mdl.predict(aa),1,None)
   if with_test and len(qa):q[qa]=np.clip(mdl.predict(XT.iloc[qa]),1,None)
  return p,ptr,q
 mdl=model(cfg)
 a=X.iloc[idx_tr];b=X.iloc[idx_va];q=XT if with_test else None
 if cfg['algo']=='lgb':a,b,q=prep_cats(a,b,q)
 if cfg['algo']=='cat':mdl.fit(a,y[idx_tr],cat_features=CATS)
 else:mdl.fit(a,y[idx_tr])
 pv=np.clip(mdl.predict(b),1,None)
 ptr=np.clip(mdl.predict(a),1,None)
 pq=np.clip(mdl.predict(q),1,None) if with_test else None
 return pv,ptr,pq

def metric(yv,p):
 return {'rmse':round(float(np.sqrt(mean_squared_error(yv,p))),3),'mae':round(float(mean_absolute_error(yv,p)),3),'r2':round(float(r2_score(yv,p)),6)}

def strata():
 lbl=np.empty(len(y),dtype=object)
 for cl in train.flight_class.unique():
  ix=np.where(train.flight_class.to_numpy()==cl)[0]
  lbl[ix]=cl+'_'+pd.qcut(y[ix],5,labels=False,duplicates='drop').astype(str)
 return lbl

def run_screen():
 a,b=train_test_split(np.arange(len(y)),test_size=.24,random_state=42,stratify=strata())
 rows=pd.read_csv(OUT+'/screen.csv').to_dict('records') if os.path.exists(OUT+'/screen.csv') else []
 for name in ['rf220','rf300','et220','et300','et360','cat300','cat420','cat480','lgb320','lgb440','lgb480','xgb360','et180','cat280','lgb260','rf180']:
  if any(r['name']==name for r in rows): continue
  t=time.time(); p,pt,_=fit_pred(name,a,b)
  res=dict(name=name,trees=CFG[name]['trees'],depth=CFG[name]['depth'],**metric(y[b],p),train_rmse=metric(y[a],pt)['rmse'],seconds=round(time.time()-t,1))
  rows.append(res);print('SCREEN',res,flush=True);pd.DataFrame(rows).sort_values('rmse').to_csv(OUT+'/screen.csv',index=False)

def run_cv(names,kind='stratified',splits=5):
 if kind=='stratified': folds=list(StratifiedKFold(n_splits=splits,shuffle=True,random_state=2026).split(X,strata()))
 elif kind=='group':folds=list(GroupShuffleSplit(n_splits=2,test_size=.20,random_state=18).split(X,y,groups=train.flight))
 else:raise ValueError(kind)
 rows=[]
 for name in names:
  oof=np.full(len(y),np.nan);qfold=[]; fold_rows=[]
  for f,(a,b) in enumerate(folds,1):
   tm=time.time(); p,pt,pq=fit_pred(name,a,b,with_test=kind=='stratified')
   oof[b]=p
   if pq is not None:qfold.append(pq)
   row=dict(model=name,fold=f,kind=kind,n_train=len(a),n_val=len(b),trees=CFG[name]['trees'],depth=CFG[name]['depth'],train_rmse=metric(y[a],pt)['rmse'],**metric(y[b],p),seconds=round(time.time()-tm,1))
   fold_rows.append(row);rows.append(row)
   print('CV',row,flush=True)
   pd.DataFrame(rows).to_csv(OUT+f'/folds_{kind}.csv',index=False)
  if kind=='stratified':
   assert np.isfinite(oof).all()
   q=np.mean(qfold,axis=0)
   np.savez_compressed(OUT+f'/pred_{name}.npz',oof=oof,test=q)
   print('OOF',name,metric(y,oof),flush=True)

if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='screen':run_screen()
 elif mode=='cv':run_cv(sys.argv[2:])
 elif mode=='group':run_cv(sys.argv[2:],'group')

"""Train-only model ablations, ExtraTrees tuning, and permutation importance."""
import sys, time, json
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from modeling import X,y,OUT,CATS,NUM,ordinal_pipe,regress_metrics,strata_labels,tr
labels=strata_labels(y,X.flight_class)
a,b=train_test_split(np.arange(len(y)),test_size=.24,random_state=42,stratify=labels)
settings=[
 ('reference',230,28,2,.94),
 ('leaf_1_all',260,None,1,1.0),
 ('leaf_1_feature_85',260,None,1,.85),
 ('leaf_1_feature_7',260,None,1,.7),
 ('leaf_2_no_depth',260,None,2,.95),
 ('leaf_2_feature_75',260,None,2,.75),
 ('leaf_3_all',260,None,3,1.0),
 ('leaf_3_feature_85',260,None,3,.85),
 ('leaf_4_all',260,None,4,1.0),
 ('leaf_2_depth20',260,20,2,1.0),
 ('leaf_1_depth22',260,22,1,1.0),
 ('leaf_2_all',260,None,2,1.0),
 ('leaf_1_depth28',260,28,1,.95),
]
rows=[];best=(1e20,None)
for name,trees,depth,leaf,feat in settings:
 start=time.time()
 model=ordinal_pipe(ExtraTreesRegressor(n_estimators=trees,max_depth=depth,min_samples_leaf=leaf,max_features=feat,n_jobs=4,random_state=42))
 model.fit(X.iloc[a],y[a]);pred=np.clip(model.predict(X.iloc[b]),1,None);metric=regress_metrics(y[b],pred)
 rows.append({'label':name,'n_estimators':trees,'max_depth':depth if depth else 'unrestricted','min_samples_leaf':leaf,'max_features':feat,**metric})
 if metric['rmse']<best[0]:best=(metric['rmse'],name)
 pd.DataFrame(rows).sort_values('rmse').to_csv(f'{OUT}/extratrees_hyperparam_tuning.csv',index=False)
 print(name,round(metric['rmse'],1),'sec',round(time.time()-start,1),flush=True)
print('TUNING BEST',best)
# Ablate feature families to see what improves generalization, instead of ranking noisy high-cardinality feature importance blindly.
sets={
 'all_features':CATS+NUM,
 'no_exact_flight': [c for c in CATS+NUM if c!='flight'],
 'no_flight_identifiers': [c for c in CATS+NUM if c not in ['flight','flight_prefix','flight_number','prefix_class']],
 'no_route_interactions': [c for c in CATS+NUM if c not in ['route','route_class','stops_route']],
 'no_missingness_flags': [c for c in CATS+NUM if c not in ['duration_missing','airline_missing']],
 'basic_11_features': ['flight','flight_prefix','source_city','destination_city','departure_time','arrival_time','stops','flight_class','duration','days_left'],
 'without_time_pair': [c for c in CATS+NUM if c!='time_pair'],
 'without_lead_time_derived': [c for c in CATS+NUM if c not in ['days_left_sq','days_left_inv']],
}
abl=[]
for name,cols in sets.items():
 newcat=[c for c in CATS if c in cols];newnum=[c for c in NUM if c in cols]
 from sklearn.compose import ColumnTransformer
 from sklearn.preprocessing import OrdinalEncoder
 from sklearn.impute import SimpleImputer
 from sklearn.pipeline import Pipeline
 pre=ColumnTransformer([('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),newcat),('num',SimpleImputer(strategy='median'),newnum)],sparse_threshold=0)
 model=Pipeline([('prep',pre),('model',ExtraTreesRegressor(n_estimators=230,max_depth=28,min_samples_leaf=2,max_features=.94,n_jobs=4,random_state=42))])
 model.fit(X.loc[X.index[a],cols],y[a]);pred=np.clip(model.predict(X.loc[X.index[b],cols]),1,None)
 r={'feature_set':name,'num_features':len(cols),**regress_metrics(y[b],pred)};abl.append(r);print('ABLATION',name,r['rmse'],flush=True)
pd.DataFrame(abl).sort_values('rmse').to_csv(f'{OUT}/feature_ablation.csv',index=False)
# Permute original feature columns within the 24%-holdout, never train. Scoring = negative MSE => square root for report.
model=ordinal_pipe(ExtraTreesRegressor(n_estimators=230,max_depth=28,min_samples_leaf=2,max_features=.94,n_jobs=4,random_state=42))
model.fit(X.iloc[a],y[a]);base=np.sqrt(mean_squared_error(y[b],model.predict(X.iloc[b])))
rng=np.random.default_rng(2026);imps=[];valid=X.iloc[b].copy()
for c in X.columns:
 score=[]
 for rep in range(3):
  vv=valid.copy();vv[c]=rng.permutation(valid[c].to_numpy())
  score.append(np.sqrt(mean_squared_error(y[b],model.predict(vv)))-base)
 imps.append({'feature':c,'rmse_increase_mean':float(np.mean(score)),'rmse_increase_std':float(np.std(score))})
print('TOP PERMUTATION IMPORTANCE')
perm=pd.DataFrame(imps).sort_values('rmse_increase_mean',ascending=False);print(perm.head(20).to_string(index=False))
perm.to_csv(f'{OUT}/holdout_permutation_importance.csv',index=False)

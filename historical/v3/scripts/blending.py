"""Out-of-fold, test-label-free blend assessment and Kaggle submission writer."""
import os,sys,json
import numpy as np,pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import KFold
from modeling import ROOT,OUT,tr,te,y,regress_metrics
names=['extra_trees_tuned','extra_trees','random_forest','bagged_trees','lightgbm','catboost','xgboost']
oof={};test={};
for name in names:
 with np.load(f'{OUT}/oof_{name}.npz') as z:
  oof[name]=z['oof'];test[name]=z['test']
rows=[]
for n in names:
 m=regress_metrics(y,oof[n]);r={'model':n,'kind':'individual',**m}
 for cl in ['Economy','Business']:
  mask=tr.flight_class.eq(cl).to_numpy();r['rmse_'+cl.lower()]=float(np.sqrt(mean_squared_error(y[mask],oof[n][mask])))
 rows.append(r)

def save(name,o,qq,weights=None,kind='ensemble'):
 rr={'model':name,'kind':kind,**regress_metrics(y,o)}
 for cl in ['Economy','Business']:
  mask=tr.flight_class.eq(cl).to_numpy();rr['rmse_'+cl.lower()]=float(np.sqrt(mean_squared_error(y[mask],o[mask])))
 rows.append(rr)
 assert len(qq)==len(te) and np.isfinite(qq).all() and (qq>0).all()
 pd.DataFrame({'id':te.id,'price':qq}).to_csv(f'{OUT}/submission_{name}.csv',index=False)
 if weights is not None:
  with open(f'{OUT}/weights_{name}.json','w') as f:json.dump(weights,f,indent=2)

recipes={
 'blend_extra_rf': {'extra_trees':.70,'random_forest':.30},
 'blend_extra_cat': {'extra_trees':.65,'catboost':.35},
 'blend_extra_light': {'extra_trees':.65,'lightgbm':.35},
 'blend_three_way': {'extra_trees':.6,'catboost':.23,'lightgbm':.17},
 'blend_four_way': {'extra_trees':.58,'random_forest':.12,'catboost':.15,'lightgbm':.15},
 'blend_tuned_three_way': {'extra_trees_tuned':.55,'catboost':.25,'lightgbm':.20},
}
for name,wt in recipes.items():
 o=sum(w*oof[k] for k,w in wt.items());q=sum(w*test[k] for k,w in wt.items());save(name,o,q,weights=wt)
# Train-only OOF regression for nonnegative simplex weights; no Kaggle feedback used.
A=np.column_stack([oof[n] for n in names]);B=np.column_stack([test[n] for n in names]);scale=30000.0

def solve(A,y,prior=None):
 a=A/scale; yy=y/scale
 def obj(w):
  error=np.mean((a@w-yy)**2)
  # regularization limits weight instability; use a predefined mild prior toward Extra Trees.
  if prior is not None:error+=.002*np.sum((w-prior)**2)
  return error
 x0=np.ones(A.shape[1])/A.shape[1] if prior is None else prior
 res=minimize(obj,x0,method='SLSQP',bounds=[(0,1)]*len(x0),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':300,'ftol':1e-12})
 if not res.success:print('BLEND OPT WARNING',res.message)
 return np.maximum(res.x,0)/np.maximum(res.x.sum(),1e-8)
prior=np.array([.37,.20,.08,.04,.15,.15,.01]);wt=solve(A,y,prior)
print('OPTIMIZED GLOBAL BLEND',dict(zip(names,wt.round(4))))
save('blend_cv_optimized',A@wt,B@wt,weights=dict(zip(names,wt.tolist())))
# Less-optimistic cross-fitted evaluation of blend weight fitting, on rows' OOF base features.
cv=KFold(n_splits=5,shuffle=True,random_state=817);cross=np.zeros(len(y));fold_w=[]
for aa,bb in cv.split(A):
 ww=solve(A[aa],y[aa],prior);cross[bb]=A[bb]@ww;fold_w.append(ww)
cv_metrics=regress_metrics(y,cross)
print('META-CROSSFIT-WEIGHTS metric',cv_metrics)
# Separate optimized weights per known ticket class; training uses only the OOF predictions.
class_o=np.zeros(len(y));class_q=np.zeros(len(te));class_weights={}
for clazz in ['Economy','Business']:
 mask=tr.flight_class.eq(clazz).to_numpy();testmask=te.flight_class.eq(clazz).to_numpy()
 ww=solve(A[mask],y[mask],prior)
 class_o[mask]=A[mask]@ww;class_q[testmask]=B[testmask]@ww
 class_weights[clazz]=dict(zip(names,ww.tolist()))
 print('CLASS',clazz,class_weights[clazz],regress_metrics(y[mask],class_o[mask]))
save('blend_class_specialist',class_o,class_q,weights=class_weights)
# Fairer meta-crossfit by class (not external test rows)
meta_o=np.zeros(len(y));train_cl=tr.flight_class.to_numpy()
for aa,bb in cv.split(A):
 for cls in ['Economy','Business']:
  acls=aa[train_cl[aa]==cls];bcls=bb[train_cl[bb]==cls]
  ww=solve(A[acls],y[acls],prior);meta_o[bcls]=A[bcls]@ww
print('META-CROSSFIT-CLASS-SPECIALIST',regress_metrics(y,meta_o))
allres=pd.DataFrame(rows).sort_values('rmse');allres.to_csv(f'{OUT}/cv_model_ensemble_comparison.csv',index=False)
print(allres.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
# Error correlation and residuals by slice
err=np.column_stack([y-oof[n] for n in names]);pd.DataFrame(np.corrcoef(err,rowvar=False),index=names,columns=names).round(3).to_csv(f'{OUT}/model_error_correlations.csv')
base=class_o if regress_metrics(y,class_o)['rmse']<regress_metrics(y,A@wt)['rmse'] else A@wt
for c in ['flight_class','stops','source_city','destination_city','departure_time','arrival_time']:
 tmp=pd.DataFrame({'category':tr[c],'squared_error':(base-y)**2,'abs_error':abs(base-y),'y':y})
 g=tmp.groupby('category',dropna=False).agg(count=('y','size'),mae=('abs_error','mean'),mse=('squared_error','mean'),mean_target=('y','mean')).reset_index();g['rmse']=np.sqrt(g.pop('mse'));g.insert(0,'feature',c)
 if c=='flight_class': full=g
 else:full=pd.concat([full,g],ignore_index=True)
full.to_csv(f'{OUT}/oof_error_slices.csv',index=False)
# Bootstrapped OOF uncertainty (paired resampling); fixed OOF predictions, not retraining.
rng=np.random.default_rng(817);best=base;extra=oof['extra_trees'];N=len(y);boot=[]
for i in range(1000):
 idx=rng.integers(0,N,N);boot.append(np.sqrt(np.mean((best[idx]-y[idx])**2))-np.sqrt(np.mean((extra[idx]-y[idx])**2)))
low,high=np.quantile(boot,[.025,.975]);print('BOOTSTRAP RMSE difference blend - ExtraTrees (95% percentile CI):',low,high)
with open(f'{OUT}/blending_diagnostics.json','w') as f:json.dump({'global_weights':dict(zip(names,wt.tolist())),'class_weights':class_weights,'meta_crossfit_global':cv_metrics,'meta_crossfit_class':regress_metrics(y,meta_o),'bootstrap_paired_ci_blend_minus_extra_rmse':[float(low),float(high)]},f,indent=2)
print('Files prepared in',OUT)

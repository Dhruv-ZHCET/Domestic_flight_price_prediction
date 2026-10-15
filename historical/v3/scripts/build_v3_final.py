"""Training-only conditional local ridge regression lookup, with 5-fold out-of-fold evaluation.
Uses existing OOF base ensemble predictions. Test prices are NEVER accessed or used.
Candidate selection from exploring many configurations is not independent of CV; see report.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
P=Path('/mnt/data/kaggle_flight_project/v3_work');D=Path('/mnt/data/kaggle_flight_project')
tr=pd.read_csv('/mnt/data/train.csv');te=pd.read_csv('/mnt/data/test.csv');sample=pd.read_csv('/mnt/data/sample_submission.csv');y=tr.price.to_numpy(float)
z=np.load(P/'baseline_with_lookup.npz')
foldid=z['foldid'];mask_exact=(z['lookup_count_oof']>0)&(z['lookup_std_oof']<=2000)
base=z['baseline_oof'].copy();base[mask_exact]=z['lookup_mean_oof'][mask_exact]
mask_exact_test=(z['lookup_count_test']>0)&(z['lookup_std_test']<=2000)
base_test=z['baseline_test'].copy();base_test[mask_exact_test]=z['lookup_mean_test'][mask_exact_test]
assert np.allclose(base_test,pd.read_csv(D/'submission_lookup_augmented.csv').price.to_numpy(),rtol=0,atol=0.0000001)
KEY=[c for c in te.columns if c not in ['id','airline','duration','days_left']]

def fit_group_curve(trainf,otherf,penalty):
 v=trainf.copy();v['x']=v.days_left.astype(float);v['yy']=v.price.astype(float)
 v['x2']=v.x*v.x;v['xy']=v.x*v.yy;v['y2']=v.yy*v.yy
 ag=v.groupby(KEY,dropna=False).agg(n=('x','size'),sx=('x','sum'),sy=('yy','sum'),sx2=('x2','sum'),sxy=('xy','sum'),sy2=('y2','sum'),minx=('x','min'),maxx=('x','max'))
 ag['ssx']=(ag.sx2-ag.sx*ag.sx/ag.n).clip(lower=0)
 ag['sxy']=ag.sxy-ag.sx*ag.sy/ag.n
 ag['beta']=ag.sxy/(ag.ssx+penalty)
 ag['alpha']=(ag.sy-ag.beta*ag.sx)/ag.n
 # Training-only group residual variance to select comparatively stable groups.
 ag['sse']=(ag.sy2 -2*ag.alpha*ag.sy -2*ag.beta*ag.sxy+ag.n*ag.alpha**2+2*ag.alpha*ag.beta*ag.sx+ag.beta**2*ag.sx2).clip(lower=0)
 ag['resid_std']=np.sqrt(ag.sse/ag.n)
 joined=otherf[KEY].merge(ag,left_on=KEY,right_index=True,how='left')
 day=otherf.days_left.to_numpy(float)
 p=(joined['alpha']+joined['beta']*day).to_numpy(float)
 dist=np.maximum(np.maximum(joined.minx.to_numpy()-day,day-joined.maxx.to_numpy()),0)
 return p,joined.n.fillna(0).to_numpy(),joined.resid_std.fillna(np.inf).to_numpy(),dist

metrics=lambda a,b:{'rmse':float(np.sqrt(mean_squared_error(a,b))),'mae':float(mean_absolute_error(a,b)),'r2':float(r2_score(a,b))}
settings={'risk_controlled':dict(penalty=150,min_group=3,max_resid_sd=800,weight=.15,max_day_extrapolation=0),
          'interpolation_25':dict(penalty=150,min_group=3,max_resid_sd=800,weight=.25,max_day_extrapolation=0)}
reports={}
for name,cfg in settings.items():
 p=base.copy();nmod=0;mask_perfold=[]
 for k in range(1,6):
  a=np.flatnonzero(foldid!=k);b=np.flatnonzero(foldid==k)
  estimates,n,std,dist=fit_group_curve(tr.iloc[a],tr.iloc[b],cfg['penalty'])
  selection=(~mask_exact[b])&(n>=cfg['min_group'])&(std<=cfg['max_resid_sd'])&(dist<=cfg['max_day_extrapolation'])&np.isfinite(estimates)
  p[b[selection]]=(1-cfg['weight'])*p[b[selection]]+cfg['weight']*estimates[selection]
  nmod+=int(selection.sum());mask_perfold.append(int(selection.sum()))
 est,n,std,dist=fit_group_curve(tr,te,cfg['penalty'])
 sel=(~mask_exact_test)&(n>=cfg['min_group'])&(std<=cfg['max_resid_sd'])&(dist<=cfg['max_day_extrapolation'])&np.isfinite(est)
 pp=base_test.copy();pp[sel]=(1-cfg['weight'])*pp[sel]+cfg['weight']*est[sel]
 assert len(pp)==len(te) and np.isfinite(pp).all() and (pp>0).all()
 out=D/f'submission_v3_{name}.csv'
 submission=sample.copy();assert sample.id.equals(te.id);submission['price']=pp;assert submission.columns.tolist()==['id','price'];submission.to_csv(out,index=False)
 fold_results=[]
 for k in range(1,6):
  v=foldid==k
  fold_results.append({'fold':k,'n':int(v.sum()),'adjusted_n':mask_perfold[k-1],**{f'prior_{key}':val for key,val in metrics(y[v],base[v]).items()},**metrics(y[v],p[v])})
 results={'configuration':cfg,'metrics_prior':metrics(y,base),'metrics_candidate':metrics(y,p),'holdout_folds_2_to_5_prior':metrics(y[foldid!=1],base[foldid!=1]),'holdout_folds_2_to_5_candidate':metrics(y[foldid!=1],p[foldid!=1]),'oof_rows_adjusted':nmod,'test_rows_adjusted':int(sel.sum()),'test_exact_lookups_preserved':int(mask_exact_test.sum()),'folds':fold_results,'submission':str(out)}
 with open(P/f'validation_v3_{name}.json','w') as f:json.dump(results,f,indent=2)
 pd.DataFrame(fold_results).to_csv(P/f'fold_metrics_v3_{name}.csv',index=False)
 np.savez_compressed(P/f'predictions_v3_{name}.npz',oof=p,test=pp)
 reports[name]=results
 print(name,json.dumps(results,indent=2),flush=True)

# Flight-level cluster bootstrap: infer uncertainty conditional on fixed OOF predictions.
ids=tr.flight.astype('category').cat.codes.to_numpy();G=int(ids.max()+1)
smx=np.bincount(ids,weights=base**2-2*base*y+y**2,minlength=G)
sy=np.bincount(ids,weights=y*0+1,minlength=G)
rg=np.random.default_rng(2026)
for name,r in reports.items():
 with np.load(P/f'predictions_v3_{name}.npz') as t: pp=t['oof']
 sc=np.bincount(ids,weights=(pp-y)**2,minlength=G)
 delta=[]
 for batch in range(10):
  draw=rg.integers(0,G,size=(100,G))
  cnt=sy[draw].sum(axis=1)
  delta.extend(np.sqrt(sc[draw].sum(axis=1)/cnt)-np.sqrt(smx[draw].sum(axis=1)/cnt))
 r['bootstrap_95pct_CI_change_rmse']=[float(v) for v in np.percentile(delta,[2.5,97.5])]
with open(P/'v3_final_report.json','w') as f:json.dump(reports,f,indent=2)

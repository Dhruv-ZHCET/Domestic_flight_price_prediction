"""Holdout permutation importance and residual diagnostic charts (training labels only)."""
import os,io,base64
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from modeling import tr,y,X,OUT,make_est,strata_labels
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
labels=strata_labels(y,X.flight_class)
a,b=train_test_split(np.arange(len(y)),test_size=.24,random_state=42,stratify=labels)
model=make_est('extra_trees_tuned');model.fit(X.iloc[a],y[a]);va=X.iloc[b].copy();yb=y[b];pred=model.predict(va)
base=np.sqrt(mean_squared_error(yb,pred))
# Each permutation is done on validation input only.
rng=np.random.default_rng(42);imp=[]
for c in X.columns:
 xx=va.copy();xx[c]=rng.permutation(xx[c].to_numpy())
 p=model.predict(xx)
 inc=np.sqrt(mean_squared_error(yb,p))-base
 imp.append({'feature':c,'delta_holdout_rmse':float(inc)})
 print('IMPORTANCE',c,round(inc,2),flush=True)
impdf=pd.DataFrame(imp).sort_values('delta_holdout_rmse',ascending=False)
impdf.to_csv(f'{OUT}/holdout_permutation_importance.csv',index=False)
# Save reusable figures for report
plt.figure(figsize=(10,7));top=impdf.head(19).iloc[::-1];sns.barplot(data=top,x='delta_holdout_rmse',y='feature',color='#2a9d8f');plt.xlabel('Increase in validation RMSE if the feature is shuffled');plt.title('Permutation importance on unseen validation records');plt.tight_layout();plt.savefig(f'{OUT}/permutation_importance.png',dpi=155);plt.close()
# Residual analysis uses strongest class specialist OOF predictions (recalculate from documented stored weights)
import json
with open(f'{OUT}/weights_blend_class_specialist.json') as f:ww=json.load(f)
po=np.zeros(len(y))
for cls in ['Economy','Business']:
 mask=tr.flight_class.eq(cls).to_numpy()
 for name,w in ww[cls].items():
  if w:
   with np.load(f'{OUT}/oof_{name}.npz') as z:po[mask]+=w*z['oof'][mask]
frame=pd.DataFrame({'actual':y,'predicted':po,'residual':po-y,'absolute_error':abs(po-y),'class':tr.flight_class,'days_left':tr.days_left,'duration':tr.duration})
plt.figure(figsize=(9,5));sample=frame.sample(min(len(frame),4000),random_state=42)
sns.scatterplot(data=sample,x='actual',y='predicted',hue='class',alpha=.30,s=11,linewidth=0)
plt.plot([0,320000],[0,320000],color='#e76f51',linestyle='--',linewidth=1.3);plt.xlabel('Actual training price (OOF)');plt.ylabel('OOF predicted price');plt.title('Cross-validated predictions vs actual (no training-row predictions)');plt.tight_layout();plt.savefig(f'{OUT}/oof_actual_vs_predicted.png',dpi=165);plt.close()
plt.figure(figsize=(9,5));sns.histplot(data=sample,x='residual',hue='class',bins=60,element='step');plt.xlabel('Prediction − actual');plt.title('OOF residual distribution');plt.tight_layout();plt.savefig(f'{OUT}/oof_residuals.png',dpi=165);plt.close()
frame['absolute_percentage_error']=frame.absolute_error/np.maximum(frame.actual,1)
frame.sort_values('absolute_error',ascending=False).head(150).to_csv(f'{OUT}/largest_oof_errors.csv',index=False)
print('DONE permutation/residuals',round(base,2))

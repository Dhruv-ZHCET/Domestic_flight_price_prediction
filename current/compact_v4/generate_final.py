"""Fit ONE copy of each bounded-depth model on the labeled training CSV.
All final submissions contain <500 total trees PER PREDICTION, unlike CV-averaged ensembles.
This is deliberately free from target lookups, grouping by price, and interpolation.
"""
import sys,os,json,time,zipfile
sys.path.insert(0,'/mnt/data/kaggle_flight_project')
import numpy as np,pandas as pd
import compact_under500 as m
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
OUT=m.OUT
prices={}
for name in ['et180','lgb260','rf220','cat420']:
 cfg=m.CFG[name];t=time.time()
 model=m.model(cfg)
 if cfg['algo']=='lgb':
  a,_,b=m.prep_cats(m.X,m.X,m.XT)
  model.fit(a,m.y,categorical_feature=m.CATS)
 elif cfg['algo']=='cat':
  model.fit(m.X,m.y,cat_features=m.CATS);b=m.XT
 else:
  model.fit(m.X,m.y);b=m.XT
 prices[name]=np.clip(model.predict(b),1,None)
 print('FIT',name,cfg['trees'],'trees',round(time.time()-t,2),'secs',flush=True)

P={
 'submission_compact_recommended_440.csv':(.6*prices['et180']+.4*prices['lgb260'],440,'ExtraTrees180(60%) + LightGBM260(40%)'),
 'submission_catboost_420.csv':(prices['cat420'],420,'Standalone CatBoost'),
 'submission_randomforest_220.csv':(prices['rf220'],220,'Standalone RandomForest'),
 'submission_rf_lightgbm_480.csv':(.5*prices['rf220']+.5*prices['lgb260'],480,'RandomForest220(50%) + LightGBM260(50%)'),
 }
for fname,(p,n,desc) in P.items():
 assert n<500 and np.isfinite(p).all() and (p>0).all()
 f=OUT+'/'+fname
 pd.DataFrame({'id':m.test.id,'price':p}).to_csv(f,index=False)
 val=pd.read_csv(f)
 assert len(val)==len(m.test) and list(val.columns)==['id','price'] and val.id.equals(m.sub.id)
 print('SAVED',fname,desc,'total_trees',n,'pred_mean',round(float(p.mean()),1),flush=True)

bases=['et180','et220','et300','cat280','cat420','lgb260','lgb440','rf220','cat_class440','rf_class440']
D={n:np.load(OUT+f'/pred_{n}.npz')['oof'] for n in bases if os.path.exists(OUT+f'/pred_{n}.npz')}
OOF={
 'Standalone Random Forest 220':D['rf220'],
 'Standalone CatBoost 420':D['cat420'],
 'Standalone LightGBM 260':D['lgb260'],
 'Standalone Extra Trees 180':D['et180'],
 'Standalone Extra Trees 300':D['et300'],
 'Class-specific CatBoost 440':D['cat_class440'],
 'Class-specific Random Forest 440':D['rf_class440'],
 'RandomForest220 + LightGBM260':.5*D['rf220']+.5*D['lgb260'],
 'ExtraTrees180 + LightGBM260':.6*D['et180']+.4*D['lgb260'],
}
B={'Standalone Random Forest 220':220,'Standalone CatBoost 420':420,'Standalone LightGBM 260':260,'Standalone Extra Trees 180':180,'Standalone Extra Trees 300':300,'Class-specific CatBoost 440':440,'Class-specific Random Forest 440':440,'RandomForest220 + LightGBM260':480,'ExtraTrees180 + LightGBM260':440}
rows=[]
for name,p in OOF.items():
 row={'model':name,'total_trees':B[name],**m.metric(m.y,p)}
 for klass in ['Economy','Business']:
  mask=m.train.flight_class.eq(klass).to_numpy()
  row[klass.lower()+'_rmse']=round(float(np.mean((p[mask]-m.y[mask])**2)**.5),3)
 rows.append(row)
results=pd.DataFrame(rows).sort_values('rmse')
results.to_csv(OUT+'/final_cv_comparison.csv',index=False)
with open(OUT+'/exact_hyperparameters.json','w') as f:
 json.dump({'model_configurations':{name:m.CFG[name] for name in ['et180','lgb260','rf220','cat420']},'featured_blend':{'extra_trees_weight':.60,'lightgbm_weight':.40,'total_trees':440,'single_full_train_fit':True,'no_price_lookups':True,'no_interpolation':True},'randomforest_blend':{'rf_weight':.50,'lightgbm_weight':.50,'total_trees':480}},f,indent=2)
print('RESULTS\n',results.to_string(index=False),flush=True)
with open(OUT+'/README.md','w') as f:
 f.write('''# Kaggle Flight Price Prediction — Compact V4 (<500 Trees Total)\n\n## Objective\nBuild bounded-depth tree regressors **without price lookups, exact matching, local interpolation, neural networks, price-based target encoding or test-label training**.\n\n## Recommended submission\n`submission_compact_recommended_440.csv`: one ExtraTrees ensemble of **180 trees, max_depth=24, min_samples_leaf=3, max_features=0.85**, weighted 0.60; one LightGBM booster of **260 trees, max_depth=7, num_leaves=25, min_child_samples=35, learning_rate=0.085, reg_lambda=10, reg_alpha=0.2, subsample=0.85, subsample_freq=1, colsample_bytree=0.85**, weighted 0.40. **Total trained trees used for inference = 440**.\n\n## Additional submissions\n- `submission_catboost_420.csv`: CatBoost 420 trees, depth 7, lr=0.075, l2_leaf_reg=10, random_strength=1.2. Stronger on two grouped-flight stress comparisons against our chosen blend, but weaker on ordinary random CV.\n- `submission_randomforest_220.csv`: RandomForest 220 trees, max_depth=15, min_samples_leaf=3, max_features=0.8, bootstrap=True.\n- `submission_rf_lightgbm_480.csv`: RandomForest220 + LightGBM260, 50/50 blend = 480 trees total.\n\n## Leakage prevention\n- All engineered features come from flight details, never the target `price`.\n- The `price` column is used only to fit models within train partitions.\n- The supplied test.csv does NOT have price labels and is not used for fitting or tuning.\n- Five-fold stratified CV gives OOF metrics in final_cv_comparison.csv; validation splits use the target only for stratification.\n- Other evidence in independent_holdouts.csv (four fresh train/valid splits) and group_blend_checks.csv (two held-out flight-code groups).\n- The final prediction uses **one full-training fit per component**, NOT averaging all five CV fits. Thus the total deployed tree count remains below 500.\n- `id` is never an input feature. Missing duration is handled within estimators and represented by a missing-value flag. Airline name is not used as input; flight prefix is used.\n\n## Caveats\n- Bounded depth / fewer trees do NOT mathematically eliminate overfitting. Some training-validation gaps remain.\n- Grouped-flight validation is a harsher shift than the provided test, because most test flight codes appear in train.\n- Earlier lookup-based model achieved Kaggle 0.95729; none of these new files has a confirmed Kaggle score. Better model simplicity may require accepting a leaderboard trade-off.\n- Competition score metric still unverified; OOF RMSE used as evaluation.\n\n## Reproduction\n1. Place train.csv, test.csv, sample_submission.csv in /mnt/data, and compact_under500.py at /mnt/data/kaggle_flight_project/.\n2. Run `python compact_under500.py cv et180 lgb260 rf220 cat420` for five-fold CV.\n3. Run `python compact_v4/generate_final.py` to fit once on full train and create submissions.\n4. See exact_hyperparameters.json for all configuration values.\n''')
zipname=OUT+'/Kaggle_Compact_Under500_Models.zip'
paths=[OUT+'/'+n for n in P]+[OUT+'/final_cv_comparison.csv',OUT+'/exact_hyperparameters.json',OUT+'/README.md',OUT+'/screen.csv',OUT+'/folds_stratified.csv',OUT+'/folds_group.csv',OUT+'/group_blend_checks.csv',OUT+'/independent_holdouts.csv',m.ROOT+'/kaggle_flight_project/compact_under500.py',OUT+'/generate_final.py',OUT+'/group_blend_check.py',OUT+'/independent_check.py']
with zipfile.ZipFile(zipname,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in paths:
  if os.path.exists(p):z.write(p,os.path.basename(p))
print('PACKAGE',zipname,os.path.getsize(zipname))

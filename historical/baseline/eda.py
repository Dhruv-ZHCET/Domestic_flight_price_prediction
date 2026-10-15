import os, sys, warnings, json, html, base64, io
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, spearmanr, pearsonr, chi2_contingency
from scipy.spatial.distance import jensenshannon
from sklearn.feature_selection import mutual_info_regression
ROOT='/mnt/data'; OUT=f'{ROOT}/kaggle_flight_project';os.makedirs(OUT,exist_ok=True)
tr=pd.read_csv(f'{ROOT}/train.csv');te=pd.read_csv(f'{ROOT}/test.csv');sub=pd.read_csv(f'{ROOT}/sample_submission.csv')
colors=['#264653','#2a9d8f','#e9c46a','#e76f51','#6c5ce7','#4c78a8']; sns.set_theme(style='whitegrid',font_scale=.85)
tr['flight_prefix']=tr.flight.str.split().str[0]
te['flight_prefix']=te.flight.str.split().str[0]
for x in (tr,te):
 x['route']=x.source_city+' → '+x.destination_city
 x['time_pair']=x.departure_time+' → '+x.arrival_time
 x['airline_missing']=x.airline.isna()
 x['duration_missing']=x.duration.isna()
tr['log_price']=np.log1p(tr.price)
figs=[]
def draw(title, render, explanation):
 fig=plt.figure(figsize=(10.5,4.3),dpi=125)
 try:
  render(fig)
  fig.tight_layout(pad=1.3)
  b=io.BytesIO();fig.savefig(b,format='png',dpi=115,bbox_inches='tight',facecolor='white');b.seek(0)
  figs.append({'title':title,'img':base64.b64encode(b.getvalue()).decode(),'explanation':explanation})
 finally:plt.close(fig)
def ax():return plt.gca()

def bar_count(col):
 def r(fig):
  order=tr[col].fillna('(missing)').value_counts().head(20).index
  cnt=tr[col].fillna('(missing)').value_counts().reindex(order)
  plt.barh(range(len(cnt)),cnt.values,color=colors[1]);plt.yticks(range(len(cnt)),[str(t) for t in cnt.index]);plt.gca().invert_yaxis();plt.xlabel('Count')
 return r

def grouped_price(col,min_count=1,limit=20):
 def r(fig):
  g=tr.groupby(col,dropna=False).price.agg(['size','median','mean']).loc[lambda x:x['size']>=min_count].sort_values('median',ascending=False).head(limit).iloc[::-1]; v=g['median'];
  ax=plt.gca();ax.barh(range(len(g)),v,color=colors[0]);ax.set_yticks(range(len(g)));ax.set_yticklabels([str(s) for s in g.index]);ax.set_xlabel('Median ticket price');
  for i,(_,t) in enumerate(g.iterrows()):ax.text(t['median']*1.008,i,f"n={int(t['size'])}",va='center',fontsize=8)
  ax.set_xlim(0,g['median'].max()*1.26)
 return r

draw('Target price: raw distribution',lambda f:(sns.histplot(tr.price,bins=95,color=colors[0]),plt.xlabel('Price')), 'The raw price distribution is right-skewed and contains expensive business-class flights; removing these indiscriminately would be unsafe.')
draw('Target price: log(1 + price)',lambda f:(sns.histplot(np.log1p(tr.price),bins=80,color=colors[1]),plt.xlabel('log(1 + price)')), 'A log transformation compresses the high-price tail, but using it for training depends on the Kaggle metric.')
draw('Flight class: frequency',bar_count('flight_class'),'Class balance matters for validation: stratify folds so both Economy and Business are represented consistently.')
draw('Class-conditional price distribution',lambda f:(sns.boxenplot(data=tr,x='flight_class',y='price',color=colors[1]),plt.xlabel('Flight class'),plt.ylabel('Price')), 'Prices differ dramatically between classes; consider class interactions or specialist models.')
for col in ['airline','source_city','destination_city','departure_time','arrival_time','stops','flight_prefix']:
 draw(f'Univariate distribution: {col}',bar_count(col),f'Assess frequency imbalance and rare categories in {col}; rare levels must be handled safely in validation.')
for col in ['duration','days_left']:
 def fn(f,col=col):
  sns.histplot(tr[col],bins=55,color=colors[2],edgecolor='none');plt.xlabel(col)
 draw(f'Univariate distribution: {col}',fn,f'Inspect distribution and missingness in {col}; do not fit imputers using the test targets.')
draw('Missingness relationships',lambda f:(sns.heatmap(pd.crosstab(tr.airline.isna(),tr.duration.isna(),normalize='all')*100,annot=True,fmt='.1f',cmap='GnBu',cbar=False),plt.xlabel('Duration missing'),plt.ylabel('Airline missing')), 'Airline missingness and duration missingness are largely independent; capture both indicators.')
for col in ['airline','source_city','destination_city','departure_time','arrival_time','stops','flight_prefix']:
 draw(f'Median price by {col}',grouped_price(col,25,18),f'Target relationship by {col} (displayed groups have at least 25 observations); avoid drawing conclusions from tiny subgroups.')
# numeric price relationship via scatter, medians
for col in ['duration','days_left']:
 def fn(f,col=col):
  d=tr.dropna(subset=[col]);sns.scatterplot(data=d.sample(min(3500,len(d)),random_state=42),x=col,y='price',hue='flight_class',alpha=.28,s=10,palette=[colors[1],colors[3]],linewidth=0);plt.legend(title='Class')
 draw(f'Price vs {col}: class-colored scatter',fn,'Flight class mediates the relation; fit conditional patterns rather than relying on global correlations.')
def price_days(f):
 d=tr.groupby(['days_left','flight_class']).price.agg(['median','mean']).reset_index()
 sns.lineplot(data=d,x='days_left',y='median',hue='flight_class',palette=[colors[1],colors[3]],linewidth=2);plt.ylabel('Median price');plt.xlabel('Days until departure')
draw('Days until departure vs median price, separated by class',price_days,'Investigate nonlinear and interacting lead-time relationships; tree ensembles can capture non-monotonicity.')
def price_duration(f):
 d=tr.copy();d['duration_bin']=pd.cut(d.duration,[0,2,3,5,7,10,15,25,50]);g=d.groupby(['duration_bin','flight_class'],observed=True).price.median().reset_index();g['duration_bin']=g.duration_bin.astype(str);sns.lineplot(data=g,x='duration_bin',y='price',hue='flight_class',marker='o',palette=[colors[1],colors[3]]);plt.xticks(rotation=45,ha='right');plt.ylabel('Median price')
draw('Median price by duration bucket and class',price_duration,'Duration effects may vary by class and are conditional on route/stops.')
def pair(f):
 s=tr.pivot_table(index='departure_time',columns='arrival_time',values='price',aggfunc='median');sns.heatmap(s,annot=True,fmt='.0f',cmap='YlGnBu');plt.ylabel('Departure time');plt.xlabel('Arrival time')
draw('Departure × arrival time: median price',pair,'Time interactions can matter more than individual main effects, although class/route confounding needs consideration.')
for clazz in ['Economy','Business']:
 def route(f,clazz=clazz):
  v=tr[tr.flight_class.eq(clazz)].pivot_table(index='source_city',columns='destination_city',values='price',aggfunc='median');sns.heatmap(v,annot=True,fmt='.0f',cmap='YlGnBu');plt.xlabel('Destination');plt.ylabel('Source')
 draw(f'{clazz}: route median price matrix',route,f'The route can explain pricing differences inside {clazz} even after flight class is controlled.')
def routes_volume(f):
 mat=pd.crosstab(tr.source_city,tr.destination_city);sns.heatmap(mat,annot=True,fmt='.0f',cmap='Blues');plt.xlabel('Destination');plt.ylabel('Source')
draw('Origin–destination frequency matrix',routes_volume,'Small and empty routes can be hard to predict; include route and component city features.')
def class_stops(f):
 sns.barplot(data=tr,x='stops',y='price',hue='flight_class',estimator=np.median,errorbar=None,palette=[colors[1],colors[3]]);plt.ylabel('Median price')
draw('Stops × flight class vs price',class_stops,'Stop counts affect fares differently within classes; beware extremely rare multi-stop records.')
def miss_class(f):
 g=tr.groupby(['flight_class','airline_missing','duration_missing']).price.agg(['median','size']).reset_index();g['group']=g.airline_missing.map({True:'Airline NA',False:'Airline present'})+' / '+g.duration_missing.map({True:'Duration NA',False:'Duration present'});sns.barplot(data=g,x='group',y='median',hue='flight_class',palette=[colors[1],colors[3]]);plt.xticks(rotation=20,ha='right');plt.ylabel('Median price')
draw('Target by missingness pattern and class',miss_class,'Evaluate whether missingness indicators carry target signal within each ticket class.')
def corr(f):
 x=tr[['price','log_price','duration','days_left']].copy();x['class_business']=tr.flight_class.eq('Business').astype(int);x['stops_num']=tr.stops.map({'zero':0,'one':1,'two_or_more':2});sns.heatmap(x.corr(method='spearman'),vmin=-1,vmax=1,cmap='vlag',center=0,annot=True,fmt='.2f')
draw('Numeric + ordinal Spearman correlation matrix',corr,'Class is likely dominant in unconditional correlation; conditional effects and nonlinear patterns require multivariate models.')
# train/test distribution shift
cat_columns=['airline','flight_prefix','source_city','destination_city','departure_time','arrival_time','stops','flight_class','route']
cat_shift=[]
for c in cat_columns:
 a=tr[c].fillna('(missing)').value_counts(normalize=True);b=te[c].fillna('(missing)').value_counts(normalize=True);allc=a.index.union(b.index);p=a.reindex(allc,fill_value=0);q=b.reindex(allc,fill_value=0)
 cat_shift.append({'feature':c,'Jensen_Shannon_distance':float(jensenshannon(p,q,base=2)),'unseen_test_categories':int(len(set(b.index)-set(a.index))), 'train_distinct':int(len(a)), 'test_distinct':int(len(b))})
num_shift=[]
for c in ['days_left','duration']:
 a=tr[c].dropna();b=te[c].dropna();ks=ks_2samp(a,b)
 num_shift.append({'feature':c,'train_median':float(a.median()),'test_median':float(b.median()),'KS_statistic':round(float(ks.statistic),4),'KS_pvalue':round(float(ks.pvalue),5),'train_missing_pct':round(100*tr[c].isna().mean(),2),'test_missing_pct':round(100*te[c].isna().mean(),2)})
sh=pd.DataFrame(cat_shift).sort_values('Jensen_Shannon_distance',ascending=False)
draw('Train/test categorical shift (JS distance)',lambda f:(sns.barplot(data=sh,x='Jensen_Shannon_distance',y='feature',color=colors[3]),plt.xlabel('0 = identical frequencies; higher = more shift')), 'Compare train and test feature distributions without using test labels; drift can weaken a good CV result.')
# mutual information log target, ordinal mixed
mi_features=['airline','flight_prefix','source_city','destination_city','departure_time','arrival_time','stops','flight_class','duration','days_left','route','time_pair','airline_missing','duration_missing']
md=tr[mi_features].copy();discrete=[]
for c in md.columns:
 if md[c].dtype=='object' or str(md[c].dtype)=='bool':
  md[c]=pd.factorize(md[c].fillna('__MISSING__'),sort=True)[0]; discrete.append(True)
 else:
  md[c]=md[c].fillna(md[c].median());discrete.append(False)
mi=mutual_info_regression(md,np.log1p(tr.price),discrete_features=discrete,random_state=42,n_neighbors=5)
mig=pd.DataFrame({'feature':mi_features,'MI_log_price':mi}).sort_values('MI_log_price',ascending=False)
draw('Univariate mutual information ranking (descriptive only)',lambda f:(sns.barplot(data=mig,x='MI_log_price',y='feature',color=colors[0])), 'Mutual information can detect nonlinear associations but is univariate and sensitive to high-cardinality features. It is not a validated feature-selection score.')
# granular stats
numeric=tr[['duration','days_left','price']].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).round(2)
missing=pd.DataFrame({'train_count':tr[['airline','duration']].isna().sum(),'train_pct':100*tr[['airline','duration']].isna().mean(),'test_count':te[['airline','duration']].isna().sum(),'test_pct':100*te[['airline','duration']].isna().mean()}).round(2)
class_stats=tr.groupby('flight_class').price.agg(['size','min','median','mean','std','max']).round(1)
summary_cat=[]
for c in cat_columns:
 g=tr.groupby(c,dropna=False).price.agg(['size','mean','median','std']).reset_index().sort_values('size',ascending=False)
 g.insert(0,'grouping_feature',c);g=g.rename(columns={c:'group_value'});summary_cat.append(g.head(100))
pd.concat(summary_cat,ignore_index=True).to_csv(f'{OUT}/eda_grouped_target_statistics.csv',index=False)
sh.to_csv(f'{OUT}/train_test_categorical_shift.csv',index=False)
pd.DataFrame(num_shift).to_csv(f'{OUT}/train_test_numeric_shift.csv',index=False)
mig.to_csv(f'{OUT}/mutual_information_exploratory.csv',index=False)
train_unknown_flight=~te.flight.isin(tr.flight); mapping=tr.dropna(subset=['airline']).groupby('flight_prefix').airline.agg(lambda s:s.mode().iloc[0]).to_dict()
recovered=tr.loc[tr.airline.isna(),'flight_prefix'].isin(mapping).sum()
issues={
 'train_rows':len(tr),'test_rows':len(te),'price_min':int(tr.price.min()),'price_mean':float(tr.price.mean()),'price_median':float(tr.price.median()),'price_max':int(tr.price.max()),'price_p99':float(tr.price.quantile(.99)),
 'train_duplicate_records_excluding_id':int(tr.drop(columns=['id','log_price','flight_prefix','route','time_pair','airline_missing','duration_missing']).duplicated().sum()),
 'test_new_flight_record_pct':round(float(train_unknown_flight.mean())*100,2),
 'test_new_flight_record_count':int(train_unknown_flight.sum()),
 'airline_missing_prefix_mappable':int(recovered),'airline_missing_total':int(tr.airline.isna().sum()),
 'test_airline_prefix_unseen': sorted(set(te.flight_prefix)-set(tr.flight_prefix)),
 'train_test_price_unavailable':True,
 'test_id_matches_submission':bool(te.id.equals(sub.id)),
 'class_summary':class_stats.to_dict(), 'num_shift':num_shift,'cat_shift':cat_shift,'mi':mig.to_dict(orient='records')}
with open(f'{OUT}/eda_summary.json','w') as f:json.dump(issues,f,indent=2)
# include hand-picked comprehensive tables
snips=[]
def tab(title,df):
 snips.append(f'<section><h2>{html.escape(title)}</h2>{df.to_html(border=0,classes="table",escape=True)}</section>')
tab('Dataset structure',pd.DataFrame({'Train':[tr.shape[0],12,'price'],'Test':[te.shape[0],11,'—']},index=['Rows','Original columns','Target']))
tab('Numerical descriptive statistics',numeric)
tab('Missingness comparison',missing)
tab('Class-conditional price statistics',class_stats)
tab('Univariate mutual information',mig.round(3).set_index('feature'))
tab('Categorical distribution shift (JS)',sh.round(4).set_index('feature'))
tab('Numeric train/test KS comparison',pd.DataFrame(num_shift).set_index('feature'))
for c in ['flight_class','stops','source_city','destination_city','departure_time','arrival_time','flight_prefix','airline']:
 tab('Price summary — '+c,tr.groupby(c,dropna=False).price.agg(['size','mean','median','std']).sort_values('size',ascending=False).head(24).round(1))
# diagnostic leakage style
warning_notes=[
'Only train.csv contains labels. Never fit on test.csv labels, pseudo-label test observations, or select models using Kaggle public leaderboard alone.',
'ID restarts at 1 in both files. Treat ID only as an output key; it is not a meaningful explanatory feature.',
'Flight-code prefix exactly identifies observed airline names when airline is not missing. Impute airline from this per-fold mapping; avoid treating missingness as a random category without testing.',
'Duration and airline are each missing in ~20% of observations. Test feature-missing indicators; impute duration within route/stops if robust, and compare with tree models that allow NaN.',
'Duplicate input feature combinations sometimes have different prices; not all residual error is removable from available predictors.',
'Many ticket classes/routes show different price dynamics. Try a global model with interactions and separate Economy/Business experts.',
'Consider a group-based flight-ID holdout as a stress test, alongside ordinary randomized K-fold; unseen flight codes appear in the test set.',
'Do not clip high prices simply for being outliers: business-class high fares are legitimate for this target.',
'RMSE, MAE and R² are reported pending official competition scoring metric. RMSE penalizes missing expensive fares sharply.',
'CatBoost/LightGBM/XGBoost are gradient-boosted decision-tree models, not neural networks.'
]
css='''<style>body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.48;margin:0;background:#f6f8fa;color:#152631}header{background:#173248;color:#fff;padding:36px max(30px,calc((100% - 1170px)/2))}h1{font-size:35px;margin:0 0 8px}header p{color:#d6e5ee}main{max-width:1170px;margin:25px auto;padding:0 22px 40px}h2{color:#173248;font-size:20px}.lead{font-size:15px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:18px}.card,section{background:white;border:1px solid #dfe6eb;border-radius:13px;padding:19px 22px;margin-bottom:14px;box-shadow:0 2px 8px #1232}.card h3{font-size:18px;margin:0}.card p{font-size:13px;color:#53616b}.card img{display:block;max-width:100%;height:auto;margin:7px auto}table{border-collapse:collapse;font-size:12px;width:100%;margin:10px 0;display:block;overflow-x:auto}th,td{padding:7px 11px;border-bottom:1px solid #eee;text-align:right}th:first-child,td:first-child{text-align:left}tr:nth-child(even){background:#f6f8f9}li{margin:9px 0}.small{font-size:12px;color:#668}</style>'''
header=f'<header><h1>Flight Price Prediction · Exploratory Data Analysis</h1><p>Training records: {len(tr):,} · Holdout records: {len(te):,} · Target: price · No test-target training</p></header>'
insights=f'''<section><h2>Key findings & modeling decisions</h2><ul>{''.join('<li>'+html.escape(i)+'</li>' for i in warning_notes)}</ul><p class="lead"><b>Coverage diagnostics:</b> {recovered:,} of {tr.airline.isna().sum():,} missing training airlines can be reconstructed from observed flight prefixes; {train_unknown_flight.sum():,}/{len(te):,} test rows ({train_unknown_flight.mean()*100:.2f}%) contain a flight identifier not present in training. Submission IDs match the test IDs: {te.id.equals(sub.id)}.</p></section>'''
plots='<h2>Visual EDA · univariate, bivariate and multivariate</h2><div class="grid">'+''.join(f'<div class="card"><h3>{html.escape(p["title"])}</h3><img src="data:image/png;base64,{p["img"]}" alt="{html.escape(p["title"])}"><p>{html.escape(p["explanation"])}</p></div>' for p in figs)+'</div>'
body='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Flight Price · EDA</title>'+css+'</head><body>'+header+'<main>'+insights+'<h2>Complete statistical tables</h2>'+''.join(snips)+plots+'<p class="small">Reproducible from eda.py; all price statistics use train.csv only. Categorical and numeric shift uses test.csv features only. Diagnostic MI is exploratory, not cross-validated selection.</p></main></body></html>'
with open(f'{OUT}/EDA_report.html','w',encoding='utf-8') as f:f.write(body)
print('EDA figures:',len(figs),'HTML size MB',len(body)/1e6)
print('summary',json.dumps({k:v for k,v in issues.items() if k not in ['class_summary','num_shift','cat_shift','mi']},indent=2))
print('MI ranking:\n',mig.to_string(index=False))
print('shift:\n',sh.to_string(index=False))

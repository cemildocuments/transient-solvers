"""Summarize paired experiments and produce machine-readable numerical results."""
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent

def interval(x,rng):
 a=np.asarray(x);idx=rng.integers(0,len(a),size=(10000,len(a)))
 return [float(v) for v in np.quantile(np.median(a[idx],axis=1),[.025,.975])]

def main():
 rows=[]
 for p in (ROOT/'results/cases').glob('*.json'):
  q=json.loads(p.read_text())
  for r in q:r.pop('t_1e6',None)
  rows+=q
 df=pd.DataFrame(rows).sort_values(['group','family','n','spread','seed','method']);df.to_csv(ROOT/'results/benchmark.csv',index=False)
 rng=np.random.default_rng(6502028);out={};table=[]
 for group in ['main','stress','scale']:
  v=df[df.group==group]
  if v.empty:continue
  p=v.pivot(index='case',columns='method',values=['rate','gain','t_1e3','alpha'])
  speed=p['t_1e3']['loaded']/p['t_1e3']['contractive'];retain=p['rate']['contractive']/p['rate']['spectral']
  out[group]={'cases':len(p),'speed_median':float(np.median(speed)),'speed_ci':interval(speed,rng),'speed_min':float(speed.min()),'speed_max':float(speed.max()),'retention_median':float(np.median(retain)),'spectral_amplifying':int((p['gain']['spectral']>1.00001).sum()),'jacobi_amplifying':int((p['gain']['jacobi']>1.00001).sum()),'spectral_gain_max':float(p['gain']['spectral'].max()),'jacobi_gain_max':float(p['gain']['jacobi'].max()),'contractive_gain_max':float(p['gain']['contractive'].max()),'alpha_min':float(p['alpha']['contractive'].min())}
  for family in ['gram','covariance','network']:
   t=v[v.family==family].pivot(index='case',columns='method',values=['rate','gain','t_1e3'])
   for method in ['loaded','uniform','jacobi','row_norm','blend','spectral','contractive']:
    table.append({'group':group,'family':family,'method':method,'cases':len(t),'gain_median':float(t['gain'][method].median()),'gain_max':float(t['gain'][method].max()),'speed_vs_loaded':float((t['t_1e3']['loaded']/t['t_1e3'][method]).median()),'spectral_rate_retained':float((t['rate'][method]/t['rate']['spectral']).median())})
 out['strata']=[]
 for (fam,sp),g in df[df.group=='main'].groupby(['family','spread']):
  p=g.pivot(index='case',columns='method',values=['rate','gain','t_1e3']);out['strata'].append({'family':fam,'spread':sp,'cases':len(p),'speed':float(np.median(p['t_1e3']['loaded']/p['t_1e3']['contractive'])),'retention':float(np.median(p['rate']['contractive']/p['rate']['spectral'])),'spectral_gain_max':float(p['gain']['spectral'].max()),'jacobi_gain_max':float(p['gain']['jacobi'].max())})
 out['optimization']={'median_seconds':float(df[df.method=='contractive'].seconds.median()),'max_seconds':float(df[df.method=='contractive'].seconds.max()),'status_counts':df[df.method=='contractive'].status.value_counts().to_dict(),'repaired':int((df[df.method=='contractive'].repair.fillna(0)>0).sum())}
 (ROOT/'results/summary.json').write_text(json.dumps(out,indent=2));pd.DataFrame(table).to_csv(ROOT/'results/group_summary.csv',index=False)
 details={}
 for group in ['main','stress','scale']:
  v=df[df.group==group];p=v.pivot(index='case',columns='method',values=['rate','gain','t_1e3','noise_output','noise_input']);active=p['gain']['spectral']>1.00001
  details[group]={'cases':len(p),'active_retention':float((p['rate']['contractive']/p['rate']['spectral'])[active].median()),'active_speed':float((p['t_1e3']['loaded']/p['t_1e3']['contractive'])[active].median()),'methods':{}}
  for method in ['loaded','uniform','jacobi','row_norm','blend','spectral','contractive']:
   details[group]['methods'][method]={'median_speed_vs_this':float((p['t_1e3'][method]/p['t_1e3']['contractive']).median()),'faster_count':int((p['t_1e3']['contractive']<p['t_1e3'][method]*(1-1e-5)).sum()),'output_noise_ratio':float((p['noise_output']['contractive']/p['noise_output'][method]).median()),'input_noise_ratio':float((p['noise_input']['contractive']/p['noise_input'][method]).median())}
 robpath=ROOT/'results/robustness.csv'
 if robpath.exists():
  rob=pd.read_csv(robpath);details['quantization']=rob[rob.kind=='gain_quantization'].groupby('level').agg(passes=('passes','sum'),count=('passes','size'),mean_retention=('relative_rate','mean'),min_retention=('relative_rate','min')).reset_index().to_dict('records')
 details['optimization_by_n']=df[df.method=='contractive'].groupby('n').seconds.agg(['median','max']).reset_index().to_dict('records')
 (ROOT/'results/detailed_summary.json').write_text(json.dumps(details,indent=2))
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()

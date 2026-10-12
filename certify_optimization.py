"""Dual upper bounds for rate optimization, independently checked against stored designs."""
from pathlib import Path
import os,sys,json,time,argparse
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[v]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent/'src'))
from core import allocation,decomposition
ROOT=Path(__file__).resolve().parent

def one(task):
 file,recompute=task
 dest=ROOT/'results/dual'/file.name
 if dest.exists() and not recompute:return json.loads(dest.read_text())
 z=np.load(ROOT/'data'/(file.stem+'.npz'));A=z['A'];out=[]
 for method in ['spectral','contractive']:
  d,info=allocation(A,method);stored=decomposition(A,z[method])[0][0];upper=info['dual_upper_bound']
  assert upper>=stored*(1-1e-7)
  out.append({'case':file.stem,'method':method,'stored_rate':stored,'upper_bound':upper,'relative_gap':max(0,upper/stored-1),'status':info['status']})
 dest.write_text(json.dumps(out,indent=2));return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--recompute',action='store_true');p.add_argument('--workers',type=int,default=6);args=p.parse_args()
 (ROOT/'results/dual').mkdir(exist_ok=True);files=sorted((ROOT/'results/cases').glob('*.json'));rows=[]
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  for i,r in enumerate(pool.map(one,[(f,args.recompute) for f in files])):
   rows+=r
   if (i+1)%30==0:print(i+1,len(files),flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'results/optimization_certificates.csv',index=False)
 print(pd.DataFrame(rows).groupby('method').relative_gap.agg(['median','max']).to_string())
if __name__=='__main__':main()

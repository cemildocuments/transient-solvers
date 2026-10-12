"""Run all fixed-protocol experiments, resuming completed cases."""
import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[v]='1'
from pathlib import Path
import sys,json,argparse,itertools,time,platform
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).parent/'src'))
from core import generate,allocation,measure
ROOT=Path(__file__).resolve().parent

def trial(task):
 group,n,spread,family,seed,methods,recompute=task
 key=f'{group}_{family}_n{n}_s{spread:g}_seed{seed}'
 path=ROOT/'results/cases'/f'{key}.json'
 if path.exists() and not recompute:return key,json.loads(path.read_text())
 A=generate(n,spread,seed,family);rows=[];arrays={'A':A}
 for method in methods:
  d,info=allocation(A,method);met=measure(A,d)
  rows.append(dict(case=key,group=group,n=n,spread=spread,family=family,seed=seed,method=method,condition=float(np.linalg.cond(A)),**info,**met));arrays[method]=d
 np.savez_compressed(ROOT/'data'/f'{key}.npz',**arrays)
 path.write_text(json.dumps(rows,indent=2,allow_nan=False))
 return key,rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--group',default='all');p.add_argument('--recompute',action='store_true');a=p.parse_args()
 (ROOT/'results/cases').mkdir(parents=True,exist_ok=True);(ROOT/'data').mkdir(exist_ok=True)
 cfg=json.loads((ROOT/'config/study.json').read_text());tasks=[]
 for group in ['main','stress','scale']:
  if a.group not in ['all',group]:continue
  v=cfg[group]
  tasks += [(group,n,s,f,k,cfg['methods'],a.recompute) for n,s,f,k in itertools.product(v['sizes'],v['spreads'],v['families'],v['seeds'])]
 start=time.perf_counter();rows=[];failed=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  fs={pool.submit(trial,t):t for t in tasks}
  for i,f in enumerate(as_completed(fs)):
   try:k,r=f.result();rows.extend(r)
   except Exception as e:failed.append({'task':fs[f],'error':str(e)});print('FAILED',fs[f],repr(e),flush=True)
   if (i+1)%12==0:print(f'{i+1}/{len(tasks)} cases; elapsed {time.perf_counter()-start:.1f} s',flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'results'/f'benchmark_{a.group}.csv',index=False)
 (ROOT/'results'/f'run_{a.group}.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-start,'workers':a.workers,'cases':len(tasks),'completed':len(rows)//len(cfg['methods']),'failed':failed,'python':sys.version,'platform':platform.system()},indent=2))
 if failed:raise SystemExit(1)
if __name__=='__main__':main()

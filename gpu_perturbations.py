"""Batched sphere-perturbation experiment; CPU double precision is the reference."""
from pathlib import Path
import sys,os,json,time
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[v]='1'
import numpy as np,torch,pandas as pd
sys.path.insert(0,str(Path(__file__).parent/'src'))
from core import decomposition,propagator
ROOT=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2)
 device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
 rng=np.random.default_rng(6502026);rows=[];N=32768;start=time.perf_counter();maxerr=0.
 cases=[f'main_{family}_n32_s1_seed{seed}' for family in ['gram','covariance','network'] for seed in [1000,1001,1002,1003]]
 cases += [f'stress_{family}_n32_s2_seed2000' for family in ['gram','covariance','network']]
 for case in cases:
  data=np.load(ROOT/'data'/f'{case}.npz');A=data['A'];n=len(A)
  E=rng.normal(size=(n,N));E/=np.linalg.norm(E,axis=0,keepdims=True)
  et=torch.tensor(E.astype('float32'),device=device)
  for method in ['loaded','spectral','contractive']:
   dec=decomposition(A,data[method]);ts=np.r_[0,np.geomspace(.0001/dec[0][-1],12/dec[0][0],79)]
   Ps=np.stack([propagator(dec,t) for t in ts]);T=torch.tensor(Ps.astype('float32'),device=device)
   # Maximum signed component measures excursion toward a common positive rail.
   peak=torch.full((N,),-float('inf'),device=device);normpeak=torch.zeros(N,device=device)
   for j in range(0,len(ts),8):
    Y=torch.matmul(T[j:j+8],et)
    peak=torch.maximum(peak,Y.amax(dim=(0,1)))
    normpeak=torch.maximum(normpeak,torch.linalg.vector_norm(Y,dim=1).amax(dim=0))
   peak=peak.cpu().numpy();normpeak=normpeak.cpu().numpy()
   # Identical inputs checked independently in double precision.
   yc=Ps@E[:,:128];pc=yc.max(axis=(0,1));maxerr=max(maxerr,float(np.max(np.abs(pc-peak[:128]))))
   rows.append({'case':case,'method':method,'trajectories':N,'rail_exceedances':int(np.sum(peak>1+1e-5)),'norm_growth':int(np.sum(normpeak>1+1e-5)),'max_component':float(peak.max()),'p99_component':float(np.quantile(peak,.99)),'device':device})
  print(case,flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'results/gpu_perturbations.csv',index=False)
 (ROOT/'results/gpu_validation.json').write_text(json.dumps({'device':device,'trajectories_per_method_case':N,'total_trajectories':len(rows)*N,'sample_times':80,'float32_vs_float64_peak_max_abs_error':maxerr,'seconds':time.perf_counter()-start,'seed':6502026},indent=2))
if __name__=='__main__':main()

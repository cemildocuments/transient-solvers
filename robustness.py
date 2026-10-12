"""Quantization, mismatch, output-noise and two-pole sensitivity checks."""
from pathlib import Path
import sys,os,json
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[v]='1'
import numpy as np,pandas as pd
from scipy.linalg import eigh,expm,svdvals
sys.path.insert(0,str(Path(__file__).parent/'src'))
from core import load_of,decomposition,propagator
ROOT=Path(__file__).resolve().parent

def main():
 rows=[];rng=np.random.default_rng(6502027)
 files=sorted((ROOT/'data').glob('main_*_n32_*.npz'))
 for file in files:
  z=np.load(file);A=z['A'];load=load_of(A);d=z['contractive'];n=len(A)
  for bits in [4,6,8,10,12]:
   w=load*d;levels=2**bits-1;loglo,loghi=np.log(.02),np.log(8.)
   # Round each channel down on the hardware grid: every local bound
   # and the total bandwidth cap are preserved without an extra clock.
   codes=np.clip(np.floor((np.log(w)-loglo)/(loghi-loglo)*levels+1e-10),0,levels)
   wq=np.exp(loglo+codes*(loghi-loglo)/levels);dq=wq/load;M=dq[:,None]*A
   alpha=eigh((M+M.T)/2,eigvals_only=True)[0];r=eigh(np.sqrt(dq)[:,None]*A*np.sqrt(dq)[None,:],eigvals_only=True)[0]
   rows.append({'case':file.stem,'kind':'gain_quantization','level':bits,'margin':alpha,'rate':r,'passes':bool(alpha>=-1e-10),'relative_rate':r/eigh(np.sqrt(d)[:,None]*A*np.sqrt(d)[None,:],eigvals_only=True)[0]})
  # Independent relative coefficient errors, including symmetry breaking.
  for sigma in [.0001,.001,.005,.01]:
   for rep in range(16):
    E=rng.normal(size=A.shape)*np.abs(A)*sigma;B=A+E;M=d[:,None]*B
    alpha=eigh((M+M.T)/2,eigvals_only=True)[0];er=eigh((d[:,None]*A+A*d[None,:])/2,eigvals_only=True)[0]-np.linalg.norm(d[:,None]*E,2)
    rows.append({'case':file.stem,'kind':'coefficient_mismatch','level':sigma,'replicate':rep,'margin':alpha,'rate':float(np.min(np.linalg.eigvals(M).real)),'passes':bool(alpha>=0),'bound_margin':er,'bound_passes':bool(er>=0)})
 pd.DataFrame(rows).to_csv(ROOT/'results/robustness.csv',index=False)
 # Higher-order model: tau z'= -z -Ax, x'=D z. Start with zero derivative.
 out=[]
 for fam in ['gram','covariance','network']:
  z=np.load(ROOT/'data'/f'main_{fam}_n16_s1_seed1000.npz');A=z['A'];n=len(A)
  for meth in ['loaded','spectral','contractive']:
   d=z[meth];vals=eigh(np.sqrt(d)[:,None]*A*np.sqrt(d)[None,:],eigvals_only=True)
   ts=np.r_[0,np.geomspace(.0001/vals[-1],15/vals[0],130)]
   for lag in [.001,.01,.1,.3,1.]:
    tau=lag/vals[-1];K=np.block([[np.zeros((n,n)),np.diag(d)],[-A/tau,-np.eye(n)/tau]])
    peak=1.
    for t in ts:peak=max(peak,svdvals(expm(K*t)[:n,:n])[0])
    out.append({'family':fam,'method':meth,'lag_ratio':lag,'peak':peak})
 pd.DataFrame(out).to_csv(ROOT/'results/two_pole.csv',index=False)
if __name__=='__main__':main()

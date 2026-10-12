"""Independent numerical checks of identities, certificates and saved results."""
from pathlib import Path
import sys,os,json
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[v]='1'
import numpy as np,pandas as pd
from scipy.linalg import expm,svdvals,eigh,solve_continuous_lyapunov
sys.path.insert(0,str(Path(__file__).parent/'src'))
from core import generate,decomposition,propagator,measure,load_of,allocation
ROOT=Path(__file__).resolve().parent

def main():
 rows=[];worst={'propagator':0.,'lyapunov':0.,'gain_grid':0.,'settling_grid':0.,'budget':0.};cert=[]
 files=sorted((ROOT/'results/cases').glob('*.json'))
 for file in files:
  records=json.loads(file.read_text());z=np.load(ROOT/'data'/(file.stem+'.npz'));A=z['A'];q=records[0]
  assert np.array_equal(A,generate(q['n'],q['spread'],q['seed'],q['family']))
  for row in records:
   row.pop('t_1e6',None);d=z[row['method']];w=load_of(A)*d
   assert w.min()>=.02-1e-8 and w.max()<=8+1e-8
   assert abs(w.sum()-len(A))<1e-6
   worst['budget']=max(worst['budget'],abs(w.sum()-len(A)))
   if row['method']=='contractive':
    M=d[:,None]*A;alpha=eigh((M+M.T)/2,eigvals_only=True)[0];r=eigh(np.sqrt(d)[:,None]*A*np.sqrt(d)[None,:],eigvals_only=True)[0]
    assert alpha>0
    cert.append({'case':file.stem,'relative_margin':alpha/r,'alpha':alpha})
  file.write_text(json.dumps(records,indent=2,allow_nan=False))
 # Independent dense exponential and Lyapunov residual checks.
 selected=[f for f in files if 'n16' in f.stem and f.stem.endswith(('seed1000','seed2000'))]
 for file in selected:
  z=np.load(ROOT/'data'/(file.stem+'.npz'));A=z['A']
  for meth in ['loaded','jacobi','spectral','contractive']:
   d=z[meth];M=d[:,None]*A;dec=decomposition(A,d)
   for t in [.001/dec[0][-1],1/dec[0][-1],1/dec[0][0]]:
    err=np.linalg.norm(propagator(dec,t)-expm(-M*t),2);worst['propagator']=max(worst['propagator'],err)
   C=solve_continuous_lyapunov(M,np.eye(len(A)));res=np.linalg.norm(M@C+C@M.T-np.eye(len(A)),2);worst['lyapunov']=max(worst['lyapunov'],res)
   coarse=measure(A,d,120);fine=measure(A,d,720)
   worst['gain_grid']=max(worst['gain_grid'],abs(coarse['gain']-fine['gain']))
   worst['settling_grid']=max(worst['settling_grid'],abs(coarse['t_1e3']/fine['t_1e3']-1))
 # Check exact two-channel boundary and equilibrium invariance.
 rng=np.random.default_rng(6502029);bad=0
 for _ in range(2000):
  rho=rng.uniform(.01,.99);s=10**rng.uniform(-2,2);q=10**rng.uniform(-3,3)
  A=np.array([[1,rho*s],[rho*s,s*s]]);d=np.array([1,q]);M=d[:,None]*A
  direct=eigh((M+M.T)/2,eigvals_only=True)[0]>=0
  bound=rho*rho*(1+q)**2<=4*q
  bad+=int(direct!=bound)
  x=rng.normal(size=2);b=A@x;sol=np.linalg.solve(M,d*b);assert np.linalg.norm(sol-x)<1e-8
 assert bad==0 and worst['propagator']<1e-8 and worst['gain_grid']<1e-7 and worst['settling_grid']<1e-8
 # Re-solve one independently regenerated problem and compare the objective.
 key='main_gram_n8_s1_seed1000';z=np.load(ROOT/'data'/(key+'.npz'));d,_=allocation(z['A'],'contractive')
 r1=decomposition(z['A'],d)[0][0];r0=decomposition(z['A'],z['contractive'])[0][0]
 assert abs(r1/r0-1)<1e-5
 out={'cases_checked':len(files),'independent_cases':len(selected),'checks_passed':True,'max_errors':{k:float(v) for k,v in worst.items()},'two_channel_trials':2000,'two_channel_disagreements':bad,'reoptimized_relative_rate_error':float(abs(r1/r0-1)),'minimum_certificate_to_rate':float(min(c['relative_margin'] for c in cert))}
 (ROOT/'results/verification.json').write_text(json.dumps(out,indent=2));pd.DataFrame(cert).to_csv(ROOT/'results/certificates.csv',index=False);print(json.dumps(out,indent=2))
if __name__=='__main__':main()

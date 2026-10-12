"""Generate and run an independent modified-nodal circuit check."""
from pathlib import Path
import argparse,sys,os,json,subprocess,shutil
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[v]='1'
import numpy as np,pandas as pd
from scipy.linalg import svdvals
from scipy.optimize import minimize_scalar
sys.path.insert(0,str(Path(__file__).parent/'src'))
from core import decomposition,propagator
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--ngspice',default='ngspice');args=p.parse_args()
 case='main_gram_n16_s1_seed1000';z=np.load(ROOT/'data'/f'{case}.npz');A=z['A'];n=len(A)
 dec=decomposition(A,z['spectral']);ts=np.geomspace(.0001/dec[0][-1],10/dec[0][0],250)
 vals=np.array([np.linalg.norm(propagator(dec,t),axis=1).max() for t in ts]);k=vals.argmax();tpeak=ts[k]
 P=propagator(dec,tpeak);node=np.linalg.norm(P,axis=1).argmax();e0=.1*P[node]/np.linalg.norm(P[node]);star=np.full(n,.9);b=A@star
 # C_i = 1/(omega_ref*d_i), G=1 S in normalized units; current and C scaled by 1e-6.
 omega_ref=2*np.pi*1e7;Gref=1e-6;end=12/dec[0][0]/omega_ref;step=end/8000
 folder=ROOT/'results/spice';folder.mkdir(exist_ok=True);summ=[]
 for method in ['loaded','spectral','contractive']:
  d=z[method];lines=['* Signed residual integrator, single-pole behavioural peripheral model','.options reltol=1e-8 abstol=1e-13 vntol=1e-10 method=gear']
  for i in range(n):
   lines.append(f'C{i} x{i} 0 {Gref/(omega_ref*d[i]):.16g} IC={star[i]+e0[i]:.16g}')
   lines.append(f'I{i} 0 x{i} {Gref*b[i]:.16g}')
   for j in range(n):
    if A[i,j]!=0:lines.append(f'G{i}_{j} x{i} 0 x{j} 0 {Gref*A[i,j]:.16g}')
  lines += ['.control','set wr_vecnames','set wr_singlescale',f'tran {step:.16g} {end:.16g} 0 {step:.16g} uic','wrdata '+method+'.dat '+' '.join(f'v(x{i})' for i in range(n)),'quit','.endc','.end']
  file=folder/(method+'.cir');file.write_text('\n'.join(lines)+'\n')
  r=subprocess.run([args.ngspice,'-b',str(file.name)],cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=True)
  (folder/(method+'.log')).write_text(r.stdout)
  data=np.loadtxt(folder/(method+'.dat'),skiprows=1);t=data[:,0];v=data[:,1:]
  pd.DataFrame(np.c_[t,v],columns=['time_s']+[f'v{i}' for i in range(n)]).to_csv(folder/(method+'.csv'),index=False)
  dc=decomposition(A,d);pred=np.array([star+propagator(dc,tt*omega_ref)@e0 for tt in t])
  summ.append({'method':method,'max_abs_error_v':float(abs(v-pred).max()),'peak_v':float(v.max()),'initial_error_norm_v':float(np.linalg.norm(e0)),'rail_v':1.,'node':int(node),'omega_ref_rad_s':omega_ref,'case':case,'points':len(t)})
 (folder/'summary.json').write_text(json.dumps(summ,indent=2));np.savez_compressed(folder/'initial_conditions.npz',A=A,e0=e0,star=star,tpeak=tpeak,node=node)
 print(json.dumps(summ,indent=2))
if __name__=='__main__':main()

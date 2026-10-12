"""Bandwidth-constrained diagonal allocation for analogue residual dynamics."""
import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.setdefault(v,'1')
import numpy as np
from scipy.linalg import eigh,svdvals,expm,solve_continuous_lyapunov
from scipy.optimize import brentq,minimize_scalar
import cvxpy as cp
import time,warnings

LOW,HIGH,ETA=.02,8.,.1

def generate(n,spread,seed,family):
 rng=np.random.default_rng(seed)
 if family=='gram':
  H=rng.normal(size=(2*n,n))/np.sqrt(2*n)
  B=H.T@H+.1*np.eye(n)
 elif family=='covariance':
  rho=rng.uniform(.55,.9)
  B=rho**abs(np.arange(n)[:,None]-np.arange(n)[None,:])+.1*np.eye(n)
 elif family=='network':
  w=rng.uniform(.5,1.5,n+1)
  B=np.diag(w[:-1]+w[1:]+.1)-np.diag(w[1:n],1)-np.diag(w[1:n],-1)
 else:raise ValueError(family)
 s=np.geomspace(1.,10.**spread,n);rng.shuffle(s)
 A=s[:,None]*B*s[None,:];A/=np.max(np.abs(A))
 return A

def load_of(A):return 1+np.abs(A).sum(axis=1)

def project_budget(w):
 n=len(w)
 shift=brentq(lambda c:np.clip(w+c,LOW,HIGH).sum()-n,-max(abs(w))-HIGH,max(abs(w))+HIGH)
 return np.clip(w+shift,LOW,HIGH)

def allocation(A,mode):
 n=len(A);load=load_of(A);vals,V=eigh(A);scale=1/np.sqrt(vals[0]*vals[-1]);B=A*scale
 u=np.ones(n)*n/load.sum()
 if mode=='loaded':return 1/load,{'seconds':0.,'status':'closed_form'}
 if mode=='uniform':return u,{'seconds':0.,'status':'closed_form'}
 if mode in ('jacobi','row_norm'):
  raw=load/(np.diag(A) if mode=='jacobi' else np.linalg.norm(A,axis=1))
  c=brentq(lambda c:np.clip(c*raw,LOW,HIGH).sum()-n,0,n/raw.min())
  return np.clip(c*raw,LOW,HIGH)/load,{'seconds':0.,'status':'closed_form'}
 if mode=='blend':
  j,_=allocation(A,'jacobi')
  def margin(x):
   d=u+x*(j-u);M=d[:,None]*A
   return eigh((M+M.T)/2,eigvals_only=True)[0]-ETA*eigh(np.sqrt(d)[:,None]*A*np.sqrt(d)[None,:],eigvals_only=True)[0]
  theta=1. if margin(1)>=0 else brentq(margin,0,1)
  return u+theta*(j-u),{'seconds':0.,'status':'bisection','theta':theta}
 d=cp.Variable(n);rate=cp.Variable();root=(V*np.sqrt(vals*scale))@V.T
 S=(cp.diag(d)@B+B@cp.diag(d))/2
 constraints=[load@d==n,cp.multiply(load,d)>=LOW,cp.multiply(load,d)<=HIGH,root@cp.diag(d)@root >> rate*np.eye(n)]
 if mode=='contractive':constraints += [S >> ETA*rate*np.eye(n)]
 elif mode=='rate_bound':constraints += [S >> rate*np.eye(n)]
 elif mode!='spectral':raise ValueError(mode)
 prob=cp.Problem(cp.Maximize(rate),constraints);start=time.perf_counter();solver='CLARABEL'
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  try:prob.solve(solver=solver,tol_gap_abs=1e-7,tol_feas=1e-7,max_iter=250)
  except cp.error.SolverError:
   solver='SCS';prob.solve(solver=solver,eps=1e-7,max_iters=100000)
 if d.value is None:raise RuntimeError((mode,prob.status))
 dv=project_budget(load*np.asarray(d.value))/load
 # Verify the delivered vector rather than trusting the solver's status.
 repair=0.
 if mode in ('contractive','rate_bound'):
  theta=ETA if mode=='contractive' else 1.
  def check(dv):
   r=eigh(np.sqrt(dv)[:,None]*A*np.sqrt(dv)[None,:],eigvals_only=True)[0]
   M=dv[:,None]*A;return eigh((M+M.T)/2,eigvals_only=True)[0]-theta*r
  if check(dv)<-1e-8*vals[-1]:
   # Mix with a feasible uniform allocation; keep every repaired case.
   if mode=='rate_bound':theta=0.999999
   f=lambda z:check((1-z)*dv+z*u)
   if f(1)<-1e-10:raise RuntimeError('Uniform feasibility failure')
   repair=brentq(f,0,1) if f(1)>0 else 1.
   dv=(1-repair)*dv+repair*u
 # Construct an independently evaluable dual upper bound. Projection removes
 # numerical negative dual eigenvalues; residual stationarity is bounded
 # using d >= 0. This bound applies even to an 'optimal_inaccurate' status.
 def psd(X):
  v,Q=eigh((X+X.T)/2);return (Q*np.maximum(v,0))@Q.T
 U=psd(constraints[3].dual_value)
 V=psd(constraints[4].dual_value) if mode in ('contractive','rate_bound') else np.zeros((n,n))
 eta=ETA if mode=='contractive' else (1. if mode=='rate_bound' else 0.)
 norm=np.trace(U)+eta*np.trace(V);U/=norm;V/=norm
 mu=np.maximum(0,constraints[1].dual_value)/norm
 nu=np.maximum(0,constraints[2].dual_value)/norm
 coeff=np.diag(root@U@root)+np.diag(B@V)
 tau=np.max(coeff/load+mu-nu)
 upper=(tau*n-LOW*np.sum(mu)+HIGH*np.sum(nu))/scale
 return dv,{'seconds':time.perf_counter()-start,'status':prob.status,'solver':solver,'repair':repair,'objective':float(rate.value)/scale,'dual_upper_bound':float(upper)}

def decomposition(A,d):
 rt=np.sqrt(d);vals,Q=eigh(rt[:,None]*A*rt[None,:]);return vals,Q,rt

def propagator(dec,t):
 vals,Q,rt=dec
 return ((rt[:,None]*Q)*np.exp(-vals*t))@Q.T/rt[None,:]

def measure(A,d,grid=120):
 dec=decomposition(A,d);vals,Q,rt=dec;n=len(A);r=vals[0]
 bound=np.sqrt(d.max()/d.min());end=np.log(bound/1e-5)/r
 ts=np.geomspace(1e-6/vals[-1],end,grid)
 norms=np.array([svdvals(propagator(dec,t))[0] for t in ts])
 maxima=[i for i in range(1,len(ts)-1) if norms[i]>=norms[i-1] and norms[i]>=norms[i+1]]
 peak=1.;tpeak=0.
 for i in maxima:
  if norms[i]<=1+1e-9:continue
  opt=minimize_scalar(lambda x:-svdvals(propagator(dec,np.exp(x)))[0],bounds=(np.log(ts[i-1]),np.log(ts[i+1])),method='bounded',options={'xatol':1e-9})
  if -opt.fun>peak:peak=-opt.fun;tpeak=np.exp(opt.x)
 def settling(eps):
  ids=np.flatnonzero(norms>eps)
  if len(ids)==0:return 0.
  i=ids[-1]
  if i+1==len(ts):raise RuntimeError('Insufficient settling horizon')
  return brentq(lambda t:svdvals(propagator(dec,t))[0]-eps,ts[i],ts[i+1],xtol=1e-10)
 M=d[:,None]*A;S=(M+M.T)/2;alpha=eigh(S,eigvals_only=True)[0]
 cov=solve_continuous_lyapunov(M,np.eye(n))
 incov=solve_continuous_lyapunov(M,np.diag(d*d))
 return dict(rate=float(r),alpha=float(alpha),gain=float(peak),t_peak=float(tpeak),t_1e3=float(settling(1e-3)),noise_output=float(np.trace(cov)/n),noise_input=float(np.trace(incov)/n),gain_ratio=float(d.max()/d.min()),bandwidth_sum=float(load_of(A)@d),bandwidth_min=float(np.min(load_of(A)*d)),bandwidth_max=float(np.max(load_of(A)*d)))

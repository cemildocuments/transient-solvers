"""Recompute the full experiment suite and its numerical verification."""
from pathlib import Path
import sys,argparse,subprocess
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--ngspice',default='ngspice');args=p.parse_args()
 jobs=[['run_study.py','--workers',str(args.workers),'--recompute'],['theory.py'],['engineering.py'],['robustness.py'],['gpu_perturbations.py'],['spice_check.py','--ngspice',args.ngspice],['certify_optimization.py','--workers',str(args.workers),'--recompute'],['verify.py'],['analyse.py']]
 for job in jobs:
  print('Running',job[0],flush=True);subprocess.run([sys.executable,*job],cwd=ROOT,check=True)
 print('All experiment and verification stages completed.')
if __name__=='__main__':main()

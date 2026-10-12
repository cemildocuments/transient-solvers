"""External structural-matrix check with verified source downloads."""
from pathlib import Path
import sys,os,json,requests,tarfile,io,hashlib
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[v]='1'
import numpy as np,pandas as pd
from scipy.io import mmread
sys.path.insert(0,str(Path(__file__).parent/'src'))
from core import allocation,measure
ROOT=Path(__file__).resolve().parent

def main():
 cache=ROOT/'data/engineering';cache.mkdir(exist_ok=True);rows=[];manifest=[]
 for name in ['bcsstk01','bcsstk02']:
  url=f'https://sparse.tamu.edu/MM/HB/{name}.tar.gz';dest=cache/(name+'.tar.gz')
  if not dest.exists():
   response=requests.get(url,timeout=60);response.raise_for_status();dest.write_bytes(response.content)
  content=dest.read_bytes()
  expected={'bcsstk01':'fbc7d883fbb1048ae7e98b12fd5b7c49fa68e0ec070b60f45047db44f63ba984','bcsstk02':'ee388126002b33fdb084fbe32f4ffec3ab11e01f2c83e69ce695fba5626abed0'}
  assert hashlib.sha256(content).hexdigest()==expected[name], 'Source archive checksum mismatch'
  manifest.append({'name':name,'url':url,'sha256':hashlib.sha256(content).hexdigest()})
  with tarfile.open(fileobj=io.BytesIO(content)) as t:
   entry=next(x for x in t.getmembers() if x.name.endswith(name+'.mtx'))
   A=mmread(io.BytesIO(t.extractfile(entry).read())).toarray()
  A/=abs(A).max();arrays={'A':A}
  for method in ['loaded','uniform','jacobi','row_norm','blend','spectral','contractive']:
   d,info=allocation(A,method);arrays[method]=d;rows.append({'case':name,'method':method,'n':len(A),'condition':float(np.linalg.cond(A)),**info,**measure(A,d)})
   print(name,method,rows[-1]['gain'],rows[-1]['t_1e3'],flush=True)
  np.savez_compressed(cache/(name+'.npz'),**arrays)
 (ROOT/'results/engineering_sources.json').write_text(json.dumps(manifest,indent=2));pd.DataFrame(rows).to_csv(ROOT/'results/engineering.csv',index=False)
if __name__=='__main__':main()

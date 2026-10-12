"""Exact two-channel transient amplification surface and feasible-gain boundary."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
 rho=np.linspace(.05,.98,75);scales=np.geomspace(1,100,81);t=np.r_[0,np.geomspace(1e-5,300,3000)]
 gain=np.ones((len(rho),len(scales)));peak=np.zeros_like(gain)
 for i,r in enumerate(rho):
  u=np.exp(-(1-r)*t);v=np.exp(-(1+r)*t);c=(u+v)/2;h=(u-v)/2
  for j,s in enumerate(scales):
   f=2*c*c+(s*s+s**-2)*h*h
   g=np.sqrt((f+np.sqrt(np.maximum(0,f*f-4*(u*v)**2)))/2);k=g.argmax();gain[i,j]=g[k];peak[i,j]=t[k]
 R=((1+np.sqrt(1-rho*rho))/rho)**2
 np.savez_compressed(ROOT/'results/two_channel.npz',rho=rho,scales=scales,gain=gain,peak_time=peak,admissible_gain_ratio=R)
 print('Exact two-channel surface saved; largest sampled gain',gain.max())
if __name__=='__main__':main()

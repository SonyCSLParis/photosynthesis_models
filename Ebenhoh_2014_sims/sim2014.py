import numpy as np
import pylab as pl
import time
import os
import model2014 as mod
from scipy.integrate import odeint    

def odelibint(PFDamp, PFDTs, pars, X0, h=.01,maxstep=.001):
  sols=np.array([X0])
  ts=[0]
  for i in range(len(PFDamp)):
    t0=ts[-1]
    t=np.linspace(t0,t0+PFDTs[0],PFDTs[0]/h)
    sol=odeint(mod.get_sys, sols[-1],t, args=(PFDamp[i], pars,), hmax=maxstep,mxstep=5000)
    ts=np.hstack([ts,t])
    sols=np.vstack([sols,sol])
    t0=ts[-1]
    t=np.linspace(t0,t0+PFDTs[1],PFDTs[1]/h)
    sol=odeint(mod.get_sys, sols[-1],t, args=(pars['PFD']['flash'], pars,),hmax=maxstep,mxstep=5000)
    ts=np.hstack([ts,t])
    sols=np.vstack([sols,sol])
  return ts, sols

def one_run_flash(X0,pars, h=.01, hmax=0.001):
   PFDTs=[120, 1]
   PFDamp=[pars['PFD']['dark']]*5+[pars['PFD']['light']]*11+[pars['PFD']['dark']]*10
   t_res,sol_res=odelibint(PFDamp, PFDTs, pars, X0, h, hmax)
   t2,PFD=mod.PFD_from_amps(PFDamp, PFDTs, h, pars)
   F=mod.fluo_TS(PFD, sol_res, pars)
   return t2, PFD, F, sol_res
   
def one_run_step(pars, X0, h=.01,maxstep=.001):
  sols=np.array([X0])
  ts=[0]
  for i in range(len(pars['stim']['PFDamp'])):
    t0=ts[-1]
    t=np.linspace(t0,t0+pars['stim']['PFDTs'][i],pars['stim']['PFDTs'][i]/h)
    sol=odeint(mod.get_sys, sols[-1],t, args=(pars['stim']['PFDamp'][i], pars,), hmax=maxstep,mxstep=5000)
    ts=np.hstack([ts,t])
    sols=np.vstack([sols,sol])
  return ts, sols

def constant_traj(X0, pars, PFD, T=400, svg=None, h=.01, hmax=0.001, final=False):
   t=np.linspace(0,T,T/h)
   sol=odeint(mod.get_sys, X0,t, args=(PFD, pars,), hmax=hmax)
   F=mod.get_fluo(PFD, sol[-1], pars)
   if svg:
      np.save(svg+'_pars',pars)
      np.save(svg+'_t',t2)
      np.save(svg+'_X',sol)
   #pl.plot(sol)
   if final: return F, sol[-1] 
   else: return F, sol

def diag_ss_1D(pars, PFDs, X0, svg=None):
   Fs=np.zeros([len(PFDs)])
   sols=np.zeros([len(PFDs),8])
   #X0=np.load('ss_0.8.npy')
   for i,PFD in enumerate(PFDs):
      Fs[i],sols[i]=constant_traj(X0, pars, PFD, T=400, final=True)
   if svg:
      np.save(svg+'_pars',pars)
      np.save(svg+'_PFDs',PFDs)
      np.save(svg+'_Fs',Fs)
      np.save(svg+'_sols',sols)
   return Fs

def diag_ss_2D(pars, X0, p1name, p2name, svg=None):
   p1s=pars["diag"][p1name]
   p2s=pars["diag"][p2name]
   Fs=np.zeros([len(p1s),len(p2s)])
   sols=np.zeros([len(p1s), len(p2s), 8])
   #X0=np.load('ss_0.8.npy')
   for i,PFD in enumerate(p1s):
      for j,p2 in enumerate(p2s):
         #X0[-2]=p2
         pars["rates"]["kATPcons"]=10.*p2 #10
         pars["rates"]["kNADPHcons"]=15.*p2 #15
         Fs[i,j],sols[i,j]=constant_traj(X0, pars, PFD,T=400,final=True)
   if svg:
      np.save(svg+'_pars',pars)
      np.save(svg+'_p1',p1s)
      np.save(svg+'_p2',p2s)
      np.save(svg+'_Fs',Fs)
      np.save(svg+'_sols',sols)
   return Fs

def test_diag(pars):
   p1name='PFDs'
   p2name='out_rates'
   svg="data/2014_noQ_%s_%s"%(p1name, p2name)
   Fs=diag_ss_2D(pars, X0, p1name, p2name, svg)


t0=time.time()
pars=mod.get_params()
X0=mod.get_init(pars)
PFDs=np.linspace(0.8,5000,100)
pars['PFD']['flash']=200
pars['PFD']['light']=100
pars['PFD']['dark']=0.8
pars["rates"]["kdynH"]=5e9*0
pars['diag']={'PFDs' : np.linspace(0.001,5000,30),
              'out_rates': np.linspace(0.001, 2.,30),
              'qEs': np.linspace(0, 1,10),
              'As': np.linspace(0, 1,10)
              }
pars['stim']={'PFDTs': [100, 800, 100],
              'PFDamp': [pars['PFD']['dark'],pars['PFD']['light'],pars['PFD']['dark']]
             }
#test_diag(pars)
"""
t, sol=one_run_step(pars,X0)
t2,PFD=mod.PFD_from_amps(pars['stim']['PFDamp'],pars['stim']['PFDTs'], .01, pars)
np.save('sol_noQ', sol)
print time.time()-t0
pars["rates"]["kdynH"]=5e9*100
t, sol_Q=one_run_step(pars,X0)
t2,PFD=mod.PFD_from_amps(pars['stim']['PFDamp'],pars['stim']['PFDTs'], .01, pars)
np.save('sol_Q', sol_Q)
"""

fname="data/2014_flash/"
if not os.path.exists(fname):
   os.makedirs(fname)   

t, PFD, F, sol=one_run_flash(X0, pars, .01)

#t2,PFD=mod.PFD_from_amps(pars['stim']['PFDamp'],pars['stim']['PFDTs'], .01, pars)
#F=mod.fluo_TS(PFD, sol, pars)
#np.save(fname+"t",t)
#np.save(fname+"PFD",PFD)
#np.save(fname+"F",F)
#np.save(fname+"sol",sol)
#np.save(fname+"pars",pars)

pl.subplot(211)
pl.plot(t, F)
pl.subplot(212)
pl.plot(t, PFD)
pl.savefig("lala.png")
print(time.time()-t0)

#Fs=diag_ss(pars, PFDs, X0, "data/diag_ss_2014")
#F,sol=constant_traj(X0, pars, 800, T=400)
#sol=np.load("data/diag_ss_2014_sols.npy")

"""
ph=-np.log(sol[:,5]/4e3)/np.log(10)   
pl.plot(PFDs, sol[:,0], "y",label="[P]")
pl.plot(PFDs, sol[:,-1], "r",label="[N]")
pl.plot(PFDs, sol[:,3], "k", label="[T]")
pl.plot(PFDs, ph, "k--", label="pH in lumen")
pl.xlabel(r"Photon Flux Density [ $\mu Em^{-2}s^{-1}$]")
pl.title("With quenching (2014)")
pl.legend()
pl.savefig("figs/diag_1D_2014.pdf",bbox_inches="tight")
t, PFD, F, sol=one_run(X0,pars)
np.save("data/2014/t",t)
np.save("data/2014/PFD",PFD)
np.save("data/2014/F",F)
np.save("data/2014/sol",sol)
"""



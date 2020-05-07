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
   t=np.linspace(0,T,int(T/h))
   sol=odeint(mod.get_sys, X0,t, args=(PFD, pars,), hmax=hmax)
   F=mod.get_fluo(PFD, sol[-1], pars)
   if svg:
      np.save(svg+'_pars',pars)
      np.save(svg+'_t',t2)
      np.save(svg+'_X',sol)
   #pl.plot(sol)
   if final: return F, sol[-1] 
   else: return t, F, sol


t0=time.time()
pars=mod.get_params()
X0=mod.get_init(pars)
PFDs=np.linspace(0.8,5000,100)
pars['PFD']['flash']=200
pars['PFD']['light']=100
pars['PFD']['dark']=0.8
pars["rates"]["kdynH"]=5e9*0
pars['stim']={'PFDTs': [100, 800, 100],
              'PFDamp': [pars['PFD']['dark'],pars['PFD']['light'],pars['PFD']['dark']]
             }


#sol=mod.get_sys(X0, 0, 1, pars)

"""
t, PFD, F, sol=one_run_flash(X0, pars, .01)
"""


t, F, sol=constant_traj(X0, pars, 200)


"""
T=400
h=.01
PFD=1
hmax=0.001
t=np.linspace(0,T,int(T/h))
sol=odeint(mod.get_sys, X0,t, args=(PFD, pars,), hmax=hmax, full_output=0,mxstep=5000)
#F=mod.get_fluo(PFD, sol[-1], pars)
"""
pl.plot(sol)
pl.savefig("lala.png")
"""
pl.subplot(211)
pl.plot(t, F)
pl.subplot(212)
pl.plot(t, PFD)
pl.savefig("lala.png")
print(time.time()-t0)
"""
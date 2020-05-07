import numpy as np
import pylab as pl
import time
import mod2011 as mod
from scipy.integrate import odeint    


def odelibint(PFDamp, PFDTs, pars,h=.01,maxstep=.001):
  h0, t0, n0, p0=mod.get_stat(pars, PFDamp[0])
  sols=np.array([[h0, n0, t0]])
  ts=[0]
  for i in range(len(PFDamp)):
    t0=ts[-1]
    t=np.linspace(t0,t0+PFDTs[0],PFDTs[0]/h)
    sol=odeint(mod.get_sys, sols[-1],t, args=(PFDamp[i], pars,), hmax=maxstep)
    ts=np.hstack([ts,t])
    sols=np.vstack([sols,sol])
    t0=ts[-1]
    t=np.linspace(t0,t0+PFDTs[1],PFDTs[1]/h)
    sol=odeint(mod.get_sys, sols[-1],t, args=(pars['PFDflash'], pars,),hmax=maxstep)
    ts=np.hstack([ts,t])
    sols=np.vstack([sols,sol])
  return ts, sols

def one_run(pars, h=.01, hmax=0.001):
   PFDTs=[120, 1]
   PFDamp=[pars['PFDdark']]*5+[pars['PFDlight']]*11+[pars['PFDdark']]*10
   t_res,sol_res=odelibint(PFDamp, PFDTs, pars, h, hmax)
   t2,PFD=mod.PFD_from_amps(PFDamp, PFDTs, h, pars)
   F=mod.fluo_TS(pars, sol_res[:,1], PFD)
   return t2, PFD, F, sol_res

def run_osc(stim,ts, pars, X0, Ni=10,maxstep=.001):
  sols=np.array([X0])
  all_ts=[0]
  pfds=[stim[0]]
  for i,t in enumerate(ts):
    t0=t
    tis=np.linspace(t0,t0+1/len(ts), Ni)
    sol=odeint(mod.get_sys, sols[-1],tis, args=(stim[i], pars,), hmax=maxstep,mxstep=5000)
    all_ts=np.hstack([all_ts,tis])
    sols=np.vstack([sols,sol])
    pfds=np.hstack([pfds, stim[i]*np.ones(len(tis))])
  return all_ts, sols, pfds

def runsim(f):
   pars=mod.get_params()
   pars['f']=f
   pars['PFDlight_0']=1000

   tf=10/pars["f"]
   N=10000 
   ts=np.linspace(0,tf,N)
   #ts=np.linspace(0,1,10000)
   h0, t0, n0, p0=mod.get_stat(pars, pars["PFDlight_0"])
   X0=np.array([h0, n0, t0])

   sols=odeint(mod.get_sys, X0, ts, args=(pars,), hmax=0.001,mxstep=5000)

   pfds=np.array([mod.get_PFD_osc(pars, t) for t in ts])
   F=mod.fluo_TS(pars, sols[:,1], pfds)

   Lcos=np.cos(2*np.pi*pars["f"]*ts)
   Lsin=-np.sin(2*np.pi*pars["f"]*ts)

   Fn=F-F.mean()

   cr=Fn@Lcos
   ci=Fn@Lsin
   
   amp=np.sqrt(cr**2+ci**2)
   ph=np.arctan(ci/cr)

   return F, amp, ph


t0=time.time()
   
#fs=np.array([0.001,.01,.1,1,10,100,1000])
fs=np.logspace(-3,3,25)
print(fs)
amps=np.zeros(len(fs))
phs=np.zeros(len(fs))
for i,f in enumerate(fs):
   F,amps[i],phs[i]=runsim(f)
   #pl.plot(F,label="%s"%f)
   tf=10./f
   N=10000 
   ts=np.linspace(0,tf,N)
   pl.plot((F-F.min())/F.max()-F.min())
   pl.plot(np.cos(2*np.pi*f*ts))
   pl.savefig("s%s.png"%i)
   pl.clf()

#pl.legend()   
#pl.savefig("lala.png")
#pl.clf()
pl.subplot(211)
pl.plot(np.log(fs),np.log(amps),"o-")
pl.subplot(212)
pl.plot(np.log(fs),phs,"o-")
pl.savefig("bode.png")
print(time.time()-t0)
import numpy as np
import pylab as pl
import time
import os
import model2014 as mod
from scipy.integrate import odeint    

t0=time.time()
pars=mod.get_params()
X0=mod.get_init(pars)

ts=np.linspace(0,10,10000)
pars["f"]=0
sols=odeint(mod.get_sys, X0, ts, args=(pars,), hmax=0.001)
X0=sols[:,-1]

"""
pl.plot(sols)
pl.savefig("ss.png")
pl.clf()
pl.plot(sols[6500:7000,0])
pl.savefig("sols_0.png")
pl.clf()
pl.plot(sols[6500:7000,1])
pl.savefig("sols_1.png")
pl.clf()

pl.plot(sols[6500:7000,2])
pl.savefig("sols_2.png")
pl.clf()

pl.plot(sols[6500:7000,3])
pl.savefig("sols_3.png")
pl.clf()

pl.plot(sols[6500:7000,4])
pl.savefig("sols_4.png")
pl.clf()

pl.plot(sols[6500:7000,5])
pl.savefig("sols_5.png")
pl.clf()

pl.plot(sols[6500:7000,6])
pl.savefig("sols_6.png")
pl.clf()

pl.plot(sols[6500:7000,7])
pl.savefig("sols_7.png")
pl.clf()


pars["f"]=100
tf=40/pars["f"]
N=10000 
ts=np.linspace(0,tf,N)

sols=odeint(mod.get_sys, X0, ts, args=(pars,), hmax=0.001)
pfds=np.array([mod.get_PFD_osc(pars, t) for t in ts])
F=mod.fluo_TS(pfds, sols, pars)


pl.plot(ts,F)
pl.savefig("fluo.png")
pl.clf()
pl.plot(sols[:,0])
pl.savefig("sols_0.png")
pl.clf()
pl.plot(sols[:,1])
pl.savefig("sols_1.png")
pl.clf()

pl.plot(sols[:,2])
pl.savefig("sols_2.png")
pl.clf()

pl.plot(sols[:,3])
pl.savefig("sols_3.png")
pl.clf()

pl.plot(sols[:,4])
pl.savefig("sols_4.png")
pl.clf()

pl.plot(sols[:,5])
pl.savefig("sols_5.png")
pl.clf()

pl.plot(sols[:,6])
pl.savefig("sols_6.png")
pl.clf()
"""
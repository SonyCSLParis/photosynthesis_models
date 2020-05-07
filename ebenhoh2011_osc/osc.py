import numpy as np
import pylab as pl
import time
import mod2011 as mod
from scipy.integrate import odeint    
from numpy import fft

t0=time.time()
pars=mod.get_params()
pars["f"]=1000

tf=10/pars["f"]
N=10000
ts=np.linspace(0,tf,N)

h0, t0, n0, p0=mod.get_stat(pars, pars["PFDlight_0"])
X0=np.array([h0, n0, t0])

sols=odeint(mod.get_sys, X0, ts, args=(pars,), hmax=0.001)

pfds=np.array([mod.get_PFD_osc(pars, t) for t in ts])
F=mod.fluo_TS(pars, sols[:,1], pfds)

Lcos=np.cos(2*np.pi*pars["f"]*ts)
Lsin=np.sin(2*np.pi*pars["f"]*ts)

Fn=F-F.mean()

cr=Fn@Lcos
ci=Fn@Lsin
   
fs=N/tf
Sp=fft.fft(Fn)
freqs=fft.fftfreq(len(Fn))*fs   
ax=pl.subplot(111)
ax.plot(freqs, np.abs(Sp))
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-5*pars["f"], 5*pars["f"])
pl.savefig("fft%s.png"%pars["f"])
pl.clf()
amp=np.sqrt(cr**2+ci**2)
ph=np.arctan(ci/cr)
print(amp,ph)

pl.subplot(211)
pl.plot(ts,pfds)
pl.subplot(212)
pl.plot(ts,F)
pl.savefig("lala.png")

print(time.time()-t0)

import numpy as np
import pylab as pl
import time
import os
import model2014 as mod

pars=mod.get_params()
X0=np.array([pars["pool_sizes"]["PQ"], 0.0202, 5.000, 0.0000, 0.0000, 0.0001, 1, 0.0000])
L=1000

v_PSII=mod.get_PSII(pars, X0, L)
v_b6f=mod.get_b6f(pars, X0)
v_FQR=mod.get_FQR(pars, X0)
v_PTOX=mod.get_PTOX(pars, X0)
v_NDH=mod.get_NDH(pars, X0)
v_PSI=mod.get_PSI(pars, X0, L)
v_FNR=mod.get_FNR(pars,X0)
v_ATPs=mod.get_ATPs(pars, X0)
v_ATPc=mod.get_ATPc(pars, X0)
v_NADPHc=mod.get_NADPHc(pars, X0)
v_leak=mod.get_leak(pars, X0)
v_stt7=mod.get_stt7(pars, X0)
v_Pph1=mod.get_Pph1(pars, X0)
v_deep=mod.get_deep(pars, X0)
v_epox=mod.get_epox(pars, X0)

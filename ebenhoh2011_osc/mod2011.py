import numpy as np
import pylab as pl
from scipy import optimize

def get_params():
  pars={'k2':3.4e6,
        'k3a':1.56e5,
        'k3b':3.12e4,
        'k4':50.,
        'k5':80., 
        'k6':.05,#0.05 originally
        'k7':.004,
        'k8':10.,
        'k9':20.,
        'pH_st': 7.8,
        'X':17.5,
        'A':32.,
        'D': 2.5,
        'bH':.01,
        'cpfd':4./3,
        'dG': 30.6,
        'R': .008314,
        'Temp': 298,
        'Pi': .01,
        'MtoMMCh': 4e3,
        'HPR': 14/3.,
        'n'  : 5,
        'KQ': .004,
        'T0': 10.66,
        'rkg':{
          "ch": [0.5, 0.0, 0.5, 0.0],
          "cq" : [2.0, 1.0, 1.0, 2.0],
          "ckq": [0.5, 0.29289321881345248, 1.70710678118654752, 0.16666666666666666],
          "ck": [0.5, 0.29289321881345248, 1.70710678118654752, 0.5]
        },
        'PFDdark': .8,
        'PFDlight_0': 300.,
        'PFDlight_1': 200.,
        'f'         : 100,
        'PFDflash'  : 8000.,
       }
  pars['Hstro']=pars['MtoMMCh']*10**(-pars['pH_st'])
  pars['K0']=pars['Pi']*np.exp(-pars['dG']/(pars["R"]*pars["Temp"]))         
  return pars  
  
def fluo_TS(pars, N, PFD):
  F=np.zeros(len(N))
  for i in range(len(N)):
    F[i]=get_fluo(pars['cpfd']*PFD[i], pars['k2'],pars['k3a'],pars['k3b'], pars['k4'],pars['X'], pars['D'], N[i])
  return F

def get_fluo(k1,k2,k3a,k3b,k4, X, D, N):
  f=1.-N
  alpha=k3a*k4*(1./(k1*f)+1./k2)
  beta=k3b*k4/(k1*f)
  p=[alpha-beta, k3a*D-alpha*X+2*beta*X+k4, -beta*X**2-k4*X]
  delta=p[1]**2-4*p[0]*p[2]
  P=(-p[1]+np.sqrt(delta))/(2*p[0])
  A1=(X-P)*k4/(f*k1)
  A2=(X-P)*k4/k2  
  return (1-N)*(D-A1-A2)/D

def get_stat(pars, PFD):
    h0=optimize.fsolve(nullcline_H,.0001,args=(pars, PFD))[0]
    t0=get_Teq(pars,h0)
    n0=get_Neq(pars,h0)
    p0=get_stat_P(pars['cpfd']*PFD,pars['k2'],pars['k3a'],pars['k3b'], pars['k4'],pars['X'], pars['D'], n0)
    return h0, t0, n0, p0

def nullcline_H(h, pars, PFD):
   Ns=get_Neq(pars,h)  
   Ts=get_Teq(pars,h)
   Ks=get_K(pars,h)   
   P0=get_stat_P(pars['cpfd']*PFD,pars['k2'],pars['k3a'],pars['k3b'], pars['k4'],pars['X'], pars['D'], Ns)
   dH=3*pars['k4']*(pars['X']-P0)-pars['HPR']*pars['k5']*(pars['A']-Ts*(1+1./Ks))-pars['k8']*(h-pars['Hstro'])
   return dH

def get_PFD(pars, t):
    return pars["PFDdark"]+(pars["PFDlight"]-pars["PFDdark"])*(t>250)*(t<1500)+(pars["PFDflash"]-pars["PFDdark"]-(pars["PFDlight"]-pars["PFDdark"])*(t>250)*(t<1500))*(t%120<1)

def get_PFD_osc(pars, t):
    #return pars["PFDdark"]*(t<1./pars["f"])+(pars["PFDlight_0"]+pars["PFDlight_1"]*np.cos(2*np.pi*pars["f"]*t))*(t>1./pars["f"])
    return (pars["PFDlight_0"]+pars["PFDlight_1"]*np.cos(2*np.pi*pars["f"]*t))

def get_Teq(pars,h):
  Ks=get_K(pars,h)
  return pars['A']/(1+pars['k9']/pars['k5']+1./Ks)

def get_Neq(pars,h):
  return 1/(1.+(pars['k7']/pars['k6'])*(1+(pars['KQ']/h)**pars['n']))

def get_K(pars,h):
   #print pars['K0']*(h/pars['Hstro'])**pars['HPR']
   return pars['K0']*(h/pars['Hstro'])**pars['HPR']

def get_stat_P(k1,k2,k3a,k3b,k4, X, D, N):
  f=1.-N
  alpha=k3a*k4*(1./(k1*f)+1./k2)
  beta=k3b*k4/(k1*f)
  p=[alpha-beta, k3a*D-alpha*X+2*beta*X+k4, -beta*X**2-k4*X]
  delta=p[1]**2-4*p[0]*p[2]
  return (-p[1]+np.sqrt(delta))/(2*p[0])   

def PFD_from_amps(PFDamp, PFDTs, h, pars):
  h0, t0, n0, p0=get_stat(pars, pars['PFDdark'])
  pfds=np.array([PFDamp[0]])
  ts=[0]
  for i in range(len(PFDamp)):
    t0=ts[-1]
    t=np.linspace(t0,t0+PFDTs[0],PFDTs[0]/h)
    pfd=np.ones(len(t))*PFDamp[i]
    ts=np.hstack([ts,t])
    pfds=np.hstack([pfds,pfd])
    t0=ts[-1]
    t=np.linspace(t0,t0+PFDTs[1],PFDTs[1]/h)
    pfd=np.ones(len(t))*pars['PFDflash']
    ts=np.hstack([ts,t])
    pfds=np.hstack([pfds,pfd])
  return ts, pfds

def get_As(PFD, N, pars):
   k1=pars['cpfd']*PFD
   P=get_stat_P(k1, pars['k2'],pars['k3a'],pars['k3b'], pars['k4'],pars['X'], pars['D'], N)
   f=1-N
   A1=pars['k4']*(pars['X']-P)/(f*k1)
   A2=pars['k4']*(pars['X']-P)/pars['k2']
   return np.array([A1, A2, pars['D']-A1-A2])

def get_sys(X, t, pars):
    PFD=get_PFD_osc(pars, t)
    P=get_stat_P(pars['cpfd']*PFD, pars['k2'],pars['k3a'],pars['k3b'], pars['k4'],pars['X'], pars['D'], X[1])
    K=get_K(pars, X[0])
    return [pars['bH']*(3.*pars['k4']*(pars['X']-P)-pars['HPR']*pars['k5']*(pars['A']-X[2]*(1+1./K))-pars['k8']*(X[0]-pars["Hstro"])),
            pars['k6']*(1-X[1])*X[0]**pars['n']/(X[0]**pars['n']+pars['KQ']**pars['n'])-pars['k7']*X[1],
            pars['k5']*(pars['A']-X[2]*(1+1./K))-pars['k9']*X[2]
           ] 


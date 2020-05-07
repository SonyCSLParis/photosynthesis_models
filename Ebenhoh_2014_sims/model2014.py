import numpy as np
import pylab as pl
from scipy import optimize
  
#beware th ph
#cPFD is 1 where it was .33 in the paper

def get_params():
  mod_params={
   "pool_sizes":{
            "PQ":  17.5,
            "PC":    4.,
            "AP":   60.,
            "NADP": 25.,
            "Fd":    5.,
            "PSI":  2.5,
            "PSII": 2.5
            }, 
   "rates":{
            "kb6f":     2.5,  
            "nub6f":   -2.5,  
            "VMAXFNR":1500.,
            "kFQR":      1.,
            "kATPs":    20.,  
            "kleak":    .01,#10 in the 2014 and 2011 papers  
            "kATPcons": 10.,  
            "kNADPHcons":15.,  
            "kPCox":  2500.,  
            "kFdred": 2.5e5,  
            "kPQred":  250.,  
            "kH":       5e8,  
            "kF":    6.25e7,  
            "kPchem":   5e9,  
            "kPTOX":   0.01,  
            "kNDH":   0.002,#0.004,    
            "kstt7": 0.0035,    
            "kPph1": 0.0013,    
            "kdeep":   0.05,    
            "kepox":  0.004, 
            "k0H":      5e8,    
            "kdynH":    5e9,
            "KMF":     1.56,    
            "KMN":     0.22,    
            "KMST":     0.2,    
            "KMQ":      5.8    
            },
   "ext_conc":{
            "Hstro": 10**(-7.8),#6.34e-5,
            "pHStroma": 7.8,#6.34e-5,
            "O2ext": 8.
            },
   "E0":{
            "QA":   -0.14,
            "PQ":    .354,
            "PC":     .38,
            "b6f":    .35,
            "P700":   .48,
            "FA":    -.55,
            "Fd":    -.43,
            "NADP": -.113, 
            "H2O": -1.23 
            },
   "misc":{
            "bH":   100.,
            "sig0I":  .2,
            "sig0II": 10,
            "nst":     2,
            "cPFD": 1.,#1/3.,
            "nQ":      5,
            "NADP":-.113,
            "G0_ATP"  :30600, 
            "R"   :8.3,#144621,#*10^-3?
            "Temp": 298.,
            "F"   : 96485.3365,
            "Pi_mol": 0.01 
            },
   "PFD":{
          "dark": .08,
          "light": 600.,
          "flash": 8000.,
          "f"    : 100
         },         
   "sim":{
            "sim_time" : 100,
            "dt"       : .001
           },
   "rkg":{
          "ch": [0.5, 0.0, 0.5, 0.0],
          "cq" : [2.0, 1.0, 1.0, 2.0],
          "ckq": [0.5, 0.29289321881345248, 1.70710678118654752, 0.16666666666666666],
          "ck": [0.5, 0.29289321881345248, 1.70710678118654752, 0.5]
   },
   "inp":{
            "t0":0,
            "t1":1,
            "I0":2
         },                       
        'PFDdark': .8,
        'PFDlight_0': 30.,
        'PFDlight_1': 200.,
        'f'         : 100,
        'PFDflash'  : 8000.

  }
  return mod_params

def get_init(pars):
  return np.array([pars['pool_sizes']['PQ'], 0.0202, 5.000, 0.0000, 0.0000, 0.0001, 1, 0.0000])
  """
  return np.array([  1.34885047e+01,   1.77449727e-01,   4.97551386e+00,
         5.28522073e-02,   2.05134454e-02,   5.33905829e-04,
         8.54882619e-01,   5.29530646e-05])
  """

def fluo_TS(PFD, X, pars):
  N=X.shape[0]
  F=np.zeros(N)
  for i in range(N):
    F[i]=get_fluo(PFD[i], X[i], pars)
  return F

def get_fluo(PFD, X, pars):
   sigII=pars["misc"]["sig0II"]+X[6]*(1-pars["misc"]["sig0II"]-pars["misc"]["sig0I"]) 
   kLII=pars['misc']['cPFD']*sigII*PFD
   kH=pars["rates"]["k0H"]+X[7]*pars["rates"]["kdynH"] 
   M=Mat_PSII(pars, X[0], X[5], X[7], kLII) 
   B=np.linalg.solve(M,np.array([0,0,0,pars["pool_sizes"]["PSII"]]))
   PHI=sigII*pars["rates"]["kF"]*(B[0]/(kH+pars["rates"]["kF"]+pars["rates"]["kPchem"])+B[2]/(kH+pars["rates"]["kF"]) )
   return PHI

def get_PFD(pars, t):
    return pars["PFD"]["dark"]+(pars["PFD"]["light"]-pars["PFD"]["dark"])*(t>250)*(t<1500)+(pars["PFD"]["flash"]-pars["PFD"]["dark"]-(pars["PFD"]["light"]-pars["PFD"]["dark"])*(t>250)*(t<1500))*(t%120<1)

def get_PFD_osc(pars, t):
    #return pars["PFDdark"]*(t<1./pars["f"])+(pars["PFDlight_0"]+pars["PFDlight_1"]*np.cos(2*np.pi*pars["f"]*t))*(t>1./pars["f"])
    return (pars["PFDlight_0"]+pars["PFDlight_1"]*np.cos(2*np.pi*pars["f"]*t))

def Keq_QAPQ(pars):
   dG0=-2*pars["misc"]["F"]*pars["E0"]["QA"]\
       +2*pars["misc"]["R"]*pars["misc"]["Temp"]*np.log(pars["ext_conc"]["Hstro"])\
       +2*pars["misc"]["F"]*pars["E0"]["PQ"]
   return np.exp(dG0/(pars["misc"]["R"]*pars["misc"]["Temp"])) 

def Keq_b6f(pars,H):
   dG0=-2*pars["misc"]["F"]*pars["E0"]["PQ"]-2*pars["misc"]["R"]*pars["misc"]["Temp"]*np.log(H*2.5e-4)+2*pars["misc"]["F"]*pars["E0"]["PC"]+2*pars["misc"]["R"]*pars["misc"]["Temp"]*np.log(pars["ext_conc"]["Hstro"]/(H*2.5e-4))
   return np.exp(dG0/(pars["misc"]["R"]*pars["misc"]["Temp"])) 

def Keq_ATPs(pars,H):
   dG0=pars["misc"]["G0_ATP"]+pars["misc"]["R"]*pars["misc"]["Temp"]*np.log(pars["ext_conc"]["Hstro"]/(H*2.5e-4))*14./3.
   return pars["misc"]["Pi_mol"]*np.exp(-dG0/(pars["misc"]["R"]*pars["misc"]["Temp"])) 

def Keq_FNR(pars):
   dG0=2*pars["misc"]["F"]*pars["E0"]["Fd"]-pars["misc"]["R"]*pars["misc"]["Temp"]*np.log(pars["ext_conc"]["Hstro"])-2*pars["misc"]["F"]*pars["E0"]["NADP"]
   return np.exp(-dG0/(pars["misc"]["R"]*pars["misc"]["Temp"])) 

def Keq_FAFd(pars):
   dG0=-pars["misc"]["F"]*pars["E0"]["Fd"]+pars["misc"]["F"]*pars["E0"]["FA"]
   return np.exp(-dG0/(pars["misc"]["R"]*pars["misc"]["Temp"])) 

def Keq_PCP700(pars):
   dG0=-pars["misc"]["F"]*pars["E0"]["P700"]+pars["misc"]["F"]*pars["E0"]["PC"]
   return np.exp(-dG0/(pars["misc"]["R"]*pars["misc"]["Temp"])) 

def get_PSI(pars, X, PFD):
   Keq1=Keq_FAFd(pars)
   Keq2=Keq_PCP700(pars) 
   sigI=pars["misc"]["sig0I"]+(1-X[6])*(1-pars["misc"]["sig0II"]-pars["misc"]["sig0I"]) 
   kLI=pars["misc"]["cPFD"]*sigI*PFD 
   Y0=pars["pool_sizes"]["PSI"]/( 1+kLI/(pars["rates"]["kFdred"]*X[2]) \
      + ( 1+(pars["pool_sizes"]["Fd"]-X[2])/(Keq1*X[2]) )*( X[1]/(Keq2*(pars["pool_sizes"]["PC"]-X[1]))  \
        + kLI/(pars["rates"]["kPCox"]*(pars["pool_sizes"]["PC"]-X[1])) ))
   nu_PSI=kLI*Y0
   return nu_PSI

def Mat_PSII(pars, P, H, Q, kLII):
    Keq=Keq_QAPQ(pars)
    kH=pars["rates"]["k0H"]+Q*pars["rates"]["kdynH"]
    return np.array([[-kLII-(pars["rates"]["kPQred"]/Keq)*(pars["pool_sizes"]["PQ"]-P),(kH+pars["rates"]["kF"]),pars["rates"]["kPQred"]*P,0],
                     [kLII,-(kH+pars["rates"]["kF"]+pars["rates"]["kPchem"]),0,0],
                     [0,0,kLII,-(kH+pars["rates"]["kF"])],
                     [1,1,1,1]
                   ])

def get_PSII(pars, X, PFD):
   sigII=pars["misc"]["sig0II"]+X[6]*(1-pars["misc"]["sig0II"]-pars["misc"]["sig0I"]) 
   kLII=pars['misc']['cPFD']*sigII*PFD 
   M=Mat_PSII(pars, X[0], X[5], X[7], kLII) 
   B=np.linalg.solve(M,np.array([0,0,0,pars["pool_sizes"]["PSII"]]))
   #nu_PSII=kLII*B[1]
   return pars["rates"]["kPchem"]*B[1]/2.
   
def get_b6f(pars, X):
   Keq=Keq_b6f(pars, X[5])
   nu_b6f=pars["rates"]["kb6f"]*( (pars["pool_sizes"]["PQ"]-X[0])*(X[1])**2-X[0]*(pars["pool_sizes"]["PC"]-X[1])**2/Keq)
   return max(nu_b6f, pars["rates"]["nub6f"])

def get_FNR(pars,X):
    Keq=Keq_FNR(pars)
    f=X[2]/pars["rates"]["KMF"]
    fminus=(pars["pool_sizes"]["Fd"]-X[2])/pars["rates"]["KMF"]
    nplus=(pars["pool_sizes"]["NADP"]-X[4])/pars["rates"]["KMN"] 
    n=X[4]/pars["rates"]["KMN"] 
    return pars["rates"]["VMAXFNR"]*(nplus*fminus**2-(n*f**2)/Keq)/((1+fminus+fminus**2)*(1+nplus)+(1+f+f**2)*(1+n)-1)

def get_FQR(pars, X):
    return pars["rates"]["kFQR"]*X[0]*(pars["pool_sizes"]["Fd"]-X[2])**2

def get_ATPs(pars, X):
    Keq=Keq_ATPs(pars,X[5])
    return pars["rates"]["kATPs"]*(pars["pool_sizes"]["AP"]-X[3]-X[3]/(pars["misc"]["Pi_mol"]*Keq))

def get_leak(pars, X):
    return pars["rates"]["kleak"]*(X[5]-(4e3*10**(-pars["ext_conc"]["pHStroma"])))

def get_ATPc(pars, X):
    return pars["rates"]["kATPcons"]*X[3]

def get_NADPHc(pars, X):
    return pars["rates"]["kNADPHcons"]*X[4]

def get_PTOX(pars, X):
    return pars["rates"]["kPTOX"]*pars["ext_conc"]["O2ext"]*(pars["pool_sizes"]["PQ"]-X[0])

def get_NDH(pars, X):
    return pars["rates"]["kNDH"]*X[0]

def get_stt7(pars, X):
    z=X[0]/(pars["pool_sizes"]["PQ"]*pars["rates"]["KMST"])
    return pars["rates"]["kstt7"]*(1-z**pars["misc"]["nst"]/(1+z**pars["misc"]["nst"]))*X[6]

def get_Pph1(pars, X):
    return pars["rates"]["kPph1"]*(1-X[6]) 

def get_epox(pars, X):
   return pars["rates"]["kepox"]*X[7]

def get_deep(pars,X):
   return pars["rates"]["kdeep"]*(1-X[7])*X[5]**pars['misc']['nQ']/((4e3*10**(-pars['rates']['KMQ']))**pars['misc']['nQ']+X[5]**pars['misc']['nQ'])

def get_sys(X, t, pars):
    PFD=get_PFD_osc(pars, t)
    v_PSII=get_PSII(pars, X, PFD)
    v_b6f=get_b6f(pars, X)
    v_FQR=get_FQR(pars, X)
    v_PTOX=get_PTOX(pars, X)
    v_NDH=get_NDH(pars, X)
    v_PSI=get_PSI(pars, X, PFD)
    v_FNR=get_FNR(pars,X)
    v_ATPs=get_ATPs(pars, X)
    v_ATPc=get_ATPc(pars, X)
    v_NADPHc=get_NADPHc(pars, X)
    v_leak=get_leak(pars, X)
    v_stt7=get_stt7(pars, X)
    v_Pph1=get_Pph1(pars, X)
    v_deep=get_deep(pars, X)
    v_epox=get_epox(pars, X)
    v_stt7=0
    v_Pph1=0
    return np.array([-v_PSII+v_b6f-v_FQR+v_PTOX-v_NDH,
                     -2*v_b6f+ v_PSI,
                     -v_PSI+2*v_FNR+2*v_FQR,
                     v_ATPs-v_ATPc,
                     v_FNR-v_NADPHc,
                     (2*v_PSII+4*v_b6f-v_ATPs*14./3.-v_leak)/pars["misc"]["bH"],
                     -v_stt7+v_Pph1,
                     v_deep-v_epox
                  ])

def PFD_from_amps(PFDamp, PFDTs, h, pars):
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
    pfd=np.ones(len(t))*pars["PFD"]['flash']
    ts=np.hstack([ts,t])
    pfds=np.hstack([pfds,pfd])
  return ts, pfds

def PFD_noflash(PFDamp, PFDTs, h, pars):
  pfds=np.array([PFDamp[0]])
  ts=[0]
  for i in range(len(PFDamp)):
    t0=ts[-1]
    t=np.linspace(t0,t0+PFDTs[0],PFDTs[0]/h)
    pfd=np.ones(len(t))*PFDamp[i]
    ts=np.hstack([ts,t])
    pfds=np.hstack([pfds,pfd])
  return ts, pfds

def get_PSII_ss(PFD, X, pars):
  Bs=np.zeros([4,len(PFD)])
  sigII=pars["misc"]["sig0II"]+X[6]*(1-pars["misc"]["sig0II"]-pars["misc"]["sig0I"]) 
  kLII=pars['misc']['cPFD']*sigII*PFD

  for i in range(len(PFD)):
     M=Mat_PSII(pars, X[0,i], X[5,i], X[7,i], kLII[i]) 
     Bs[:,i]=np.linalg.solve(M,np.array([0,0,0,pars["pool_sizes"]["PSII"]]))
  return Bs
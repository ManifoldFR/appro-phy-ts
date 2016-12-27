import scipy.integrate as inte
import numpy as np
import matplotlib.pyplot as plt

def ress(X,t,L,w0,l0):
    r,dr = X
    d2r = - w0*(r-l0) + L*L/(r*r*r)
    return [dr,d2r]

T = np.linspace(0,15,6000)

def solve_g(r0,dr0,dth0):
    w0 = 4*np.pi
    l0 = 0.05
    L = r0*r0*dth0
    X = inte.odeint(ress, [2,1], T, (L,w0,l0))
    return X

def build_g(r0,dr0,dth0):
    X = solve_g(r0,dr0,dth0)
    r,dr = zip(*X)
    th = 1 /np.array([1/val**2 for val in r])
    
    fig = plt.figure(figsize=(14,7))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122,projection='polar')
    
    ax0.grid()
    ax0.plot(T,r)
    
    ax1.grid(True)
    ax1.plot(r,th)
    
    fig.tight_layout()
    fig.show()
import scipy.integrate as inte
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


wx,wy = 10,10

Y,X = np.ogrid[-wx:wx:64j, -wy:wy:64j]

def mag(x,y):
    i = 1e-5
    r2 = (x**2+y**2)
    return (i/(2*np.pi)) * np.array([-y/r2,x/r2,0])

def mag_stream():
    Hx, Hy, Hz = mag(X,Y)
    fig, ax = plt.subplots(1,1)
    
    color = np.log(np.sqrt(Hx**2 + Hy**2))
    
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.grid()
    
    ax.streamplot(X,Y,Hx,Hy, arrowstyle='->', arrowsize=0.6, color=color, cmap=plt.cm.inferno,density=2)
    
    ax.set_xlim((-wx,wx))
    ax.set_ylim((-wy,wy))
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig, ax

def mag_parti():
    q = 1.6e-19
    mass = 9.1e-31
    mu = 4*np.pi*1e-7
    k = q*mu/mass
    
    cour = 0.1
    
    def force(X,t):
        x, y, z, dx, dy, dz = X
        
        # Champ d'intensité magnétique
        H = mag(x,y) 
        #H = np.array([0,0,1e-3])
        
        vites = np.array([dx,dy,dz])
        acc = k*np.cross(vites, H)
        
        return np.array([dx, dy, dz, acc[0], acc[1], acc[2]])
        
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection="3d")
    
    T = np.linspace(0,10,300)
    r0 = np.array([1, 0, 0])
    v0 = np.array([1, 0.1, 0])
    cinit = np.append(r0, v0)
    sol = inte.odeint(force, [1,0,0,1,0.1,0], T)
    
    x, y, z, vx, vy, vz = zip(*sol)
    
    ax.grid(True)
    ax.plot(x,y,z)
    fig.tight_layout()
    
    return fig
    
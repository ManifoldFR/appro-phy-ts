import scipy.integrate as inte
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

wx,wy = 600,600

Y,X = np.ogrid[-wy:wy:64j, -wx:wx:64j]

def field(q,r0,x,y):
    """Champ électrique créé par une charge q à la position r0"""
    den = ((x-r0[0])**2+(y-r0[1])**2)**1.5
    return np.array([q*(x-r0[0])/den,q*(y-r0[1])/den])

def build_field(charges):
    """Étant donnée une liste de charges (valeur,[posx,posy]), calcule le champ résultant """
    def resulting_field(x,y):
        val = np.array([0,0])
        for ch in charges:
            val = val + field(*ch,x,y)
        return val
    return resulting_field
    
def build_stream(charges, customTitle=None):
    Ex, Ey = np.zeros((64,64)), np.zeros((64,64))
    for ch in charges:
        cmx, cmy = field(*ch, x=X,y=Y)
        Ex += cmx
        Ey += cmy
    color = np.log(np.sqrt(Ex**2 + Ey**2))
    
    fig = plt.figure(1, figsize=(8,8))
    ax = fig.add_subplot(111)
    
    ax.streamplot(X,Y,Ex,Ey, arrowstyle='->', arrowsize=0.6, color=color, cmap=plt.cm.inferno,density=2.2)
    
    # Taille des charges ponctuelles
    radius = (wx*wx+wy*wy)**0.5
    
    charge_colors = {True: '#cc0000', False: '#0000aa'}
    for q, pos in charges:
        ax.add_artist(Circle(pos, 0.01*radius, color=charge_colors[q>0],zorder=2))
    
    if customTitle:
        ax.set_title(customTitle)
    else:
        ax.set_title('Champ électrique')
    ax.set_xlim((-wx,wx))
    ax.set_ylim((-wy,wy))
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig, ax # Renvoie la figure et la case où est tracé le champ



def parti(x0,v0, ch, fax = None):
    """Donne la trajectoire d'une particule chargée de charge q, de position initiale x0,
    de vitesse initiale v0, dans le dernier champ construit"""
    
    champ = build_field(ch)
    
    q = 1e11
    
    def F(r,t):
        x,y,vx,vy = r
        fx,fy = q*champ(x,y)
        return (vx,vy,fx,fy)
    
    cinit = x0 + v0
    
    tmax = 4 # Temps
    T = np.linspace(0,tmax,4096)
    
    sol = inte.odeint(F, cinit, T)
    
    x,y,vx,vy = zip(*sol)
    
    if not(fax):
        fig, ax = build_stream(ch)
        #ax.set_title("Champ électrique, trajectoire d'une particule" "\n" r"de charge $q=%g$" % q)
    else:
        fig,ax=fax
    ax.set_xlim((-wx,wx))
    ax.set_ylim((-wy,wy))
    ax.set_aspect('equal')
        
    ax.plot(x,y, 'r',linewidth=1.3)
    fig.tight_layout()
    return fig,ax



# Charge: liste de couples (valeur, pos)
ch1  = [(-1.4e-9,(0,0))]
ch2 = [(1.4e-9,(1,0)),(1.4e-9,(-1,0))]
ch3 = [(-1.4e-9,(-1,0)),(1.4e-9,(1,0))]

def dipole_eau():
    de=0.22*1.6e-19
    angHO = 104.5*np.pi/360
    lenHO = 95.8
    global x,y
    x = -lenHO*np.cos(angHO) + 40
    y = lenHO*np.sin(angHO)
    ch_eau = [(-2*de,(0,40)),(de,(y,x)),(de,(-y,x))]
    g = build_stream(ch_eau, r"Champ électrique créé par la molécule d'eau")
    g[1].set_xlabel(r'$x$ ($\mathrm{pm}$)')
    g[1].set_ylabel(r'$y$ ($\mathrm{pm}$)')
    g[0].tight_layout()
    return g[0]
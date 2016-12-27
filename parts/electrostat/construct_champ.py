import scipy.integrate as inte
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

wx,wy = 4,4

grx = np.linspace(-wx,wx,64)
gry = np.linspace(-wy,wy,64)
X,Y = np.meshgrid(grx,gry)

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
    
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    
    ax.streamplot(X,Y,Ex,Ey, arrowstyle='->',arrowsize=1,color=color, cmap=plt.cm.inferno,density=2)
    
    charge_colors = {True: '#cc0000', False: '#0000aa'}
    for q, pos in charges:
        ax.add_artist(Circle(pos, 0.05, color=charge_colors[q>0]))
    
    if customTitle:
        ax.set_title(customTitle)
    else:
        ax.set_title('Champ électrique')
    ax.set_xlim((-wx,wx))
    ax.set_ylim((-wy,wy))
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig, ax # Renvoie la figure et la case où est tracé le champ


# Charge: liste de couples (valeur, pos)
charges=[(1.4e-9,(1,0)),(-1.4e-9,(-1,0))]
charges2 = [(1.4e-9,(1,0)),(1.4e-9,(-1,0))]
charges3 = [(-1.4e-9,(-1,0)),(1.4e-9,(1,0))]


def parti(x0,v0, onStream = True):
    """Donne la trajectoire d'une particule chargée de charge q, de position initiale x0,
    de vitesse initiale v0, dans le dernier champ construit"""
    
    champ = build_field(charges3)
    
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
    
    if onStream:
        fig, ax = build_stream(charges3)
        ax.set_title("Champ électrique, trajectoire d'une particule" "\n" r"de charge $q=%g$" % q)
    else:
        fig, ax = plt.subplots(1,1)
        ax.grid()
        ax.set_title(r"Trajectoire d'une particule chargée de charge $q=%g$" "\n" r"dans le champ électrique" % q)
    ax.plot(x,y, 'r',linewidth=2)
    fig.tight_layout()
    return fig,sol
    
    
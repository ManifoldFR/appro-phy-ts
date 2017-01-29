import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as inte
import scipy.special as spec
import sympy as sp
sp.init_printing()

from matplotlib import animation, rc

## Domaine

class Domain:
    
    def __init__(self, xm, ym, I, J, origin=True, eps=0):
        begx = -origin*xm
        begy = -origin*ym
        
        self.xm = xm
        self.xs = np.linspace(begx,xm,I)
        self.I = I
        
        self.ym = ym
        self.ys = np.linspace(begy,ym,J)
        self.J = J
        
        self.rad = (self.xs**2 + self.ys**2)**0.5
        cond = self.rad > eps
        
        self.grid = np.meshgrid(self.xs[cond],
                self.ys[cond])
        
        
    
    def __call__(self):
        return self.grid

## Champ électromagnétique

class Field:
    
    def __init__(self, omega):
        self.pulsation = omega
        self.wavenum = omega/3e8
    
    def create_domain(self, D):
        self.domain = D
    
    def E(self,x,y,z,t):
        w = self.pulsation
        k = self.wavenum
        r = (x**2+y**2)**0.5
        ez = 0.5*spec.j0(k*r)*np.cos(w*t) + \
            0.5*spec.y0(k*(r-0.1))*np.sin(w*t)
        return 0,0,ez
    
    def B(self,x,y,z,t):
        omega = self.omega
        by = 0.05*np.cos(omega*(x-t))
        return 0,by,0
    
    def Force(self,state,t):
        charge = 1e4
        pos, vel = state[0:3],state[3:6]
        acc = charge*(self.E(*pos,t) + \
                np.cross(vel,self.B(*pos,t)))
        return np.concatenate((vel,acc))
    
    def trajectoire(self, r0, v0, tm, N):
        
        condinit = np.concatenate((r0,v0))
        
        times = self.times = np.linspace(0,tm,N)
        
        sol = inte.odeint(champ.Force, condinit, times)
        self.pos,self.vel = np.hsplit(sol,2)
        
        times = self.times
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True)
        ax.set_aspect('equal')
        ax.plot(*self.pos.T)
        self.graph = fig
    
    def _setup_surface(self):
        grid = self.domain()
        
        fig = plt.figure(2)
        fig.suptitle(r"Composante verticale du champ électrique")
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True)
        
        return grid,fig, ax
        
    def surface(self,t):
        grid,fig,ax = self._setup_surface()
        xg,yg = grid
        Ex, Ey, Ez = self.E(x=xg, y=yg, z=0, t=t)
        
        ax.plot_wireframe(*grid, Ez)
        
        self.surf = fig
        self.surf.show()
        return self.surf
    
    def animate(self, tm, fps=25):
        """
        tm -> intervalle [0,tm]
        fps : nombre d'images par seconde à générer
        """
        animtime = 6 # durée de l'animation
        interval = 1000/fps # temps entre deux frames
        N = int(np.ceil(animtime*fps))
        dt = tm/N # saut temporel entre chaque frame
        
        grid,fig,ax = self._setup_surface()
        def fieldo(t):
            Ex, Ey, Ez = self.E(*grid, z=0, t=t)
            return Ez
        
        Ez = fieldo(0)
        zlims = (np.nanmin(Ez),1.5*np.nanmax(Ez))
        
        surf = ax.plot_wireframe(*grid, fieldo(0),)
        time_text = ax.text2D(0.5,1,r'$t=0$',
                    horizontalalignment='center',
                    transform=ax.transAxes)
        ax.set_zlim(zlims)
        
        def update(i):
            ax.clear()
            ax.set_zlim(zlims)
            ti = i*dt/N
            Ez = fieldo(ti)
            s = r'$t={:.3e}$'.format(ti)
            time_text = ax.text2D(0.5,1,s,
                    horizontalalignment='center',
                    transform=ax.transAxes)
            data = ax.plot_wireframe(*grid, Ez)
            return data, time_text
        
        anim = animation.FuncAnimation(fig,update,
                        frames=N,interval=interval)
        self.anim = anim
        self.animfig = fig
        self.animax = ax
        fig.show()




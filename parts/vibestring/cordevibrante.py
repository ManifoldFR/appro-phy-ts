
# coding: utf-8

# In[3]:

get_ipython().magic('matplotlib inline')


# In[4]:

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import colors, rc, animation
from IPython.display import set_matplotlib_formats, HTML, Image

set_matplotlib_formats('png', 'pdf')
rc('animation', html='html5')


# In[5]:

sp.init_printing()


# Ce module sert à simuler le comportement d'une corde, fixée à ses deux extrémités, homogène et inextensible que l'on fait vibrer, comme une corde d'un violon.

# # Théorie

# L'évolution de la corde est régie par l'équation aux dérivées partielles
# \\[
# \frac{\partial^2 y}{\partial x^2} - \frac{1}{c^2}\frac{\partial^2 y}{\partial t^2} = 0
# \\]
# où $y(x,t)$ est l'altitude de la corde à la position et $x$ à l'instant $t$. Ici, $x\in[0,L]$. De plus, les conditions au bords de la corde imposent le problème de Dirichlet
# \\[
# y(0,t) = y(L,t) = 0.
# \\]

# On procède par séparation des variables, en écrivant \\[ y(x,t) = Y(x)T(t). \\]
# 
# Alors, l'équation aux dérivées partielles mène, si on cherche une solution non nulle avec donc $Y\neq 0$ et $T\neq 0$, à
# \\[
# \frac{Y''(x)}{Y(x)} = \frac{1}{c^2}\frac{T''(t)}{T(t)}
# \\]
# ce qui est donc égal à une constante indépendante des deux variables, $\lambda$.

# Pour $Y$, on a
# \\[
# Y''(x) - \frac{\lambda}{c^2}Y(x) = 0
# \\]
# donc en posant $k$ une racine carrée de $-\lambda/c^2$,
# \\[Y(x) = A\cos(kx) + B\sin(kx). \\]
# Les conditions aux limites imposent $A=0$ puis $\sin(kL) = 0$ lorsque $Y$ est une solution non triviale, d'où \\[k=\frac{n\pi}{L}\equiv k_n,\quad n\in\mathbb Z \\]
# puisque $\sin$ ne s'annule que sur $\pi\mathbb Z$. Ce mode $Y(x) = \sin(k_nx)$ s'appelle *mode propre* de vibration. On pose alors $\omega_n = k_nc=nc\pi/L$ la pulsation associée, d'où $\lambda = -\omega_n^2$.

# Alors,
# \\[
# T''(t) +\omega^2 T(t) = 0
# \\]
# ce qui mène à
# \\[
# T(t) = A\cos(\omega t) + B\sin(\omega t).
# \\]

# Les modes propres de la corde vibrante sont donc harmoniques de pulsations $(\omega_n)$, de la forme
# \\[
# \sin\left(\frac{\omega_nx}{c}\right)(A_n\cos(\omega_n t)+B_n\sin(\omega_n t)).
# \\]
# 
# La famille de fonctions $\left(\sin(\omega_nx/c),\cos(\omega_nx/c)\right)_{n\in\mathbb N}$ étant totale dans l'espace des fonctions continues sur $[0,L]$, on en déduit que la solution générale $y$ est une superposition linéaire de ces modes propres.

# ## Avec amortissement

# \\[
# \frac{\partial^2y}{\partial x^2} - 2c^2\Gamma\frac{\partial y}{\partial t} - \frac{1}{c^2}\frac{\partial^2y}{\partial t^2}=0
# \\]
# où $c=\sqrt{\frac T\mu}$ comme avant et $\Gamma = \frac \gamma {2T}$ est un facteur d'amortissement.

# Une séparation des variables mène au jeu d'équations différentielles
# \\[
# \begin{aligned}
# Y''(x) - \frac{\lambda}{c^2} Y(x) &= 0\\
# T''(t) +2\Gamma T'(t) - \lambda T(t) &= 0 
# \end{aligned}
# \\]

# Le profil est identique, de la forme $Y(x) = \sin(\omega_n x/c)$.

# Cependant, on a $\Delta = \Gamma^2 - \omega_n^2$. Donc plusieurs cas:
# 
# * $\Gamma > \omega_n$: alors $r = -\Gamma \pm \sqrt{\Gamma^2-\omega_n^2} = -\Gamma \pm \gamma_n$ et \\[T(t) = e^{-\Gamma t}\left(A\cosh(\gamma_nt)+B\sinh(\gamma_nt)\right).\\]
# * $\Gamma = \omega_n$: alors \\[T(t) = (At+B)e^{-\Gamma t}.\\]
# * $\Gamma < \omega_n$: alors $r = -\Gamma \pm i \Omega_n$ où $\Omega_n= \sqrt{\omega_n^2-\Gamma^2}$, et
# \\[ T(t) = e^{-\Gamma t}\left(A\cos\left(\Omega_nt\right) + B\sin\left(\Omega_nt\right)\right). \\]

# In[28]:

def oscillharmo(amort):
    def f(t):
        # pulsation propre = 1
        delt = amort**2 - 1
        rac = np.sqrt(delt+0j)
        if rac !=0:
            return np.exp(-amort*t)*                (np.exp(rac*t) + 3*np.exp(-rac*t))
        else:
            return (t+2*t)*np.exp(-amort*t)
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.grid(True)
    Ta = np.linspace(0,30,200)
    vals = f(Ta)
    ax.plot(Ta, vals.real)
    ax.plot(Ta, vals.imag)


# In[32]:

oscillharmo(0.99)


# # Code: harmoniques

# In[5]:

class CordeHarmo:
    omega, x, t = sp.symbols("omega0 x t", real=True)
    cel = sp.symbols("c", real = True)
    N = sp.symbols("N", integer=True)
    m = sp.Idx("m", N)
    exprHarmo = sp.sin(m*omega*x/cel)*sp.cos(m*omega*t)
    
    def __init__(self, L, c):
        """L: longueur de la corde
        c: célérité de l'onde
        Harmonique, phase à l'origine nulle."""
        self.L = L
        self.cval = c
        self.dom = np.linspace(0, L, 100)
        puls = c*sp.pi/L
        x,t = self.x,self.t
        omega = self.omega
        cel,m = self.cel,self.m
        har = self.exprHarmo.subs({cel:c,omega:puls})
        self.harm  = har
        self.funcH = sp.lambdify((x,t,m), har, "numpy")
    
    @staticmethod
    def _legendeHarm(h):
        return r"$m = %d$" % h
    
    def plotHarmo(self, t, li):
        fig = plt.figure(1, figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.grid(True)
        dom = self.dom
        
        if hasattr(li, '__iter__'):
            for m in li:
                lab = self._legendeHarm(m)
                valeurs = self.funcH(dom, t, m)
                ax.plot(dom,valeurs,label=lab)
        else:
            lab = self._legendeHarm(li)
            valeurs = self.funcH(dom, t, li)
            ax.plot(dom,valeurs, label=lab)
        ax.set_title(r"Harmoniques $m$")
        ax.legend()
        fig.tight_layout()
    
    @staticmethod
    def _limMargin(a,b):
        wind = np.absolute(b-a)
        return (a-.1*wind,b+.1*wind)
    
    def animateHarm(self, t0, t1, h):
        """Animation du h-ème harmonique"""
        dom = self.dom
        fig = plt.figure(2,figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_xlabel("Position $x$ (m)")
        ax.set_ylabel("Altitude $y(x,t)$")
        if h>1:
            ax.set_title("%d-ème harmonique" % h)
        else:
            ax.set_title("1er harmonique")
        fps=30
        animtime=10
        N = int(np.ceil(animtime*fps))
        timestep = (t1-t0)/N
        interv = 1000/fps
        
        # Valeurs
        func = self.funcH
        states=[func(dom,t0+i*timestep,h) for i in range(N)]
        self.record = states
        
        line, = ax.plot(dom,states[0])
        timetext = ax.text(0.1,0.95,
            r"$t=%f$" % t0,
            transform=ax.transAxes)
        # Fenêtre
        lims = (np.min(np.asarray(states)),
               np.max(np.asarray(states)))
        ax.set_ylim(self._limMargin(*lims))
        
        def update(i):
            ti = t0+i*timestep
            timetext.set_text(r"$t=%f$" % ti)
            line.set_data(dom,states[i])
            return line, timetext
        
        anim = animation.FuncAnimation(fig,
                update, frames=N,interval=interv,
                blit=True)
        self.anim_harm = anim


# In[6]:

cor = CordeHarmo(1, 10)


# In[7]:

cor.exprHarmo


# In[8]:

cor.plotHarmo(0, [1,2,3])


# In[15]:

cor.animateHarm(0,0.4,1)

cor.anim_harm


# In[16]:

cor.animateHarm(0,0.4,2)

cor.anim_harm


# In[30]:

cor.animateHarm(0,0.4,5)
cor.anim_harm


# In[9]:

class Corde(CordeHarmo):
    def __init__(self, L, c, coeffs):
        CordeHarmo.__init__(self, L, c)
        N = len(coeffs)
        coeffs = coeffs/(2*N)
        expr = sum(coeffs[i-1]*self.harm.subs({self.m:i}) for i in range(1,N+1))
        self.expr = expr
        func = sp.lambdify((self.x,self.t), expr, "numpy")
        self.func = func
        
    def plot(self, t):
        fig = plt.figure(3, figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.grid(True)
        dom = self.dom
        valeurs = self.func(dom, t)
        ax.plot(dom, valeurs)
    
    @staticmethod
    def _limMargin(a,b):
        wind = np.absolute(b-a)
        return (a-.1*wind,b+.1*wind)
    
    def animate(self, t0, t1):
        dom = self.dom
        fig = plt.figure(2,figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_xlabel("Position $x$ (m)")
        ax.set_ylabel("Altitude $y(x,t)$")
        ax.set_title("Corde")
        
        fps=30
        animtime=10
        N = int(np.ceil(animtime*fps))
        timestep = (t1-t0)/N
        interv = 1000/fps
        
        # Valeurs
        func = self.func
        states=[func(dom,t0+i*timestep) for i in range(N)]
        self.record = states
        
        line, = ax.plot(dom,states[0])
        timetext = ax.text(0.1,0.95,
            r"$t=%f$" % t0,
            transform=ax.transAxes)
        # Fenêtre
        lims = (np.min(np.asarray(states)),
               np.max(np.asarray(states)))
        ax.set_ylim(self._limMargin(*lims))
        
        def update(i):
            ti = t0+i*timestep
            timetext.set_text(r"$t=%f$" % ti)
            line.set_data(dom,states[i])
            return line, timetext
        
        anim = animation.FuncAnimation(fig,
                update, frames=N,interval=interv,
                blit=True)
        self.anim = anim


# In[19]:

def makeCoeffs(f,N,L):
    from scipy.fftpack import dst
    sampleSpace = np.linspace(0,L,N+1)
    samples = L*f(sampleSpace)
    return dst(samples)


# In[16]:

@np.vectorize
def pluck(x):
    if x>=2/3:
        return 1-x
    else:
        return x/2


# In[20]:

def dess2(N):
    def newpluck(x):
        return pluck(x,1)
    coeffs = makeCoeffs(newpluck, N, 1)
    from scipy.fftpack import idst
    
    vals = idst(coeffs, type=2)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    interv = np.linspace(0,1,N+1)
    ax.plot(interv, vals)


# In[14]:

dess2(80)


# In[21]:

coeffs = makeCoeffs(pluck,80,1)
vibr = Corde(1,340, coeffs)


# In[22]:

vibr.plot(0.2)


# In[23]:

vibr.animate(0,0.01)


# In[24]:

vibr.anim


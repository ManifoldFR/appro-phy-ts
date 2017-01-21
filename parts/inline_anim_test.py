
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[2]:

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import set_matplotlib_formats, HTML, Image


# In[3]:

set_matplotlib_formats('png', 'pdf')
rc('animation', html='html5')


# In[9]:

fig,ax = plt.subplots(figsize=(8,5),dpi=72)

ax.set_xlim((0, 2))
ax.set_ylim((-2,2))

line, = ax.plot([], [], lw = 2)


# In[10]:

def init():
    line.set_data([],[])
    return line,


# In[11]:

def animate(i):
    x=np.linspace(0,2,1000)
    y=np.sin(2*np.pi*(x-0.008*i))
    line.set_data(x,y)
    return line,


# In[12]:

anim = animation.FuncAnimation(fig, animate, init_func=init, 
                               frames=100, interval= 40, blit=True)


# In[13]:

anim


# # Tests
# 
# On impose une température de 10 degrés au bord sauf au sud.
# 
# **Attention** Assurez-vous que votre fonction est définie au moins au bord de votre domaine...

# In[87]:

def f(x, y, t):
    if x==1 or x==-1:
        return 35+10*y*np.sin(y)
    elif y==-1:
        return 40
    else:
        return 26


# In[89]:

Omega = Domain(1,1200, J=32, N=200)

U0 = np.zeros((32,32))

#enforceDirich(Omega, U0, f, 1)
#graphe(Omega, U0)


# In[90]:

anim = HeatAnimation(Omega, U0, f)


# In[91]:

anim.evolHeat()

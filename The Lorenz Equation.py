#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# In[6]:


def derivative_lorenz(t,r,sigma,r_,b_):
    """
    r: array that stores x, y, and z
    
    sigma, r_, b_: parameters for the Lorenz Equation
    
    """


    x = r[0]
    y = r[1]
    z = r[2]
    fx = sigma*(y-x)
    fy = r_*x - y - x*z
    fz = x*y - b_*z
    return [fx, fy, fz] 


# In[7]:


exp_fps = 1000 # samples per second
t_span = [0, 50] # simulate system for 3 seconds
t = np.arange(*t_span, 1/exp_fps)
y0 = [1,10,1] #initial conditions
sigma,r_,b_ = 10.,28,8/3

sol4 = solve_ivp(derivative_lorenz, t_span, y0, t_eval = t,  method = 'LSODA',args=(sigma,r_,b_))


# In[23]:


t = sol4.t
x = sol4.y[0,:]
y = sol4.y[1,:]
z = sol4.y[2,:]

ax = plt.figure().add_subplot(projection='3d')

ax.plot(x, y, z)
ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)
ax.set_zlabel('z',fontsize=16)
ax.set_title('The Lorenz equation solution')
plt.show()


# In[24]:


plt.plot(x,z)
plt.xlabel('x',fontsize=16)
plt.ylabel('z',fontsize=16)


# In[22]:


plt.plot(t,x,label='x',linewidth=3)
plt.plot(t,y,label='y')
plt.plot(t,z,label='z')
plt.xlabel('t',fontsize=16)
plt.legend()


# In[ ]:





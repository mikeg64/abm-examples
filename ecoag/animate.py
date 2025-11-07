#!/usr/bin/env python
# coding: utf-8

# https://towardsdatascience.com/introduction-to-mesa-agent-based-modeling-in-python-bcb0596e1c9a
# 
# https://mesa.readthedocs.io/en/latest/tutorials/intro_tutorial.html
# 
# https://notebook.community/projectmesa/mesa-examples/examples/ForestFire/.ipynb_checkpoints/Forest%20Fire%20Model-checkpoint
# 
# To install Mesa, simply:
# 
# $ pip install mesa https://mesa.readthedocs.io/en/latest/apis/space.html

# In[1]:


import random

import numpy as np
import math

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.animation import FuncAnimation



#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import mesa
print(mesa.__version__)
from mesa import Model, Agent
#from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
#from mesa.time import RandomActivation

from ecoagclasses import *
from ecoagagents import *
from humanag import *


#from mesa.datacollection import DataCollector
#from mesa.batchrunner import BatchRunner


# In[2]:


def draw_boids(model):
    x_vals = []
    y_vals = []
    t_vals=[]
    cols = []
    
    for boid in model.agents:
        atype=1.5
        x, y = boid.pos
        if boid.atype == 0:
            atype=1
            #cols.append(60)
            cols.append('red')
            #cols.append(1+atype)
        if boid.atype == 1:
            #cols.append(2+boid.fruit)
            cols.append('green')
            atype=10
        if boid.atype == 2:
            #cols.append(2+boid.honey)
            cols.append('blue')
            atype=50
        #cols.append(1)
        x_vals.append(x)
        y_vals.append(y)
        t_vals.append(100*atype)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.scatter(x_vals, y_vals, s=t_vals, c=cols, alpha=0.5)


# In[ ]:
    
def update(frame_number):
    model.step()
    colsp=np.array([])
    xp, yp = np.array([]), np.array([])
    for boid in model.agents:
        x, y = boid.pos
        if boid.atype == 0:
            atype=1
            #ncols=(1+atype)
            ncols=('red')
        if boid.atype == 1:
            #ncols=(2+boid.pollen)
            ncols=('green')
            atype=10
        if boid.atype == 2:
            #ncols=(2+boid.honey)
            ncols=('blue')
            atype=50
            #cols.append(1)

        #x_vals.append(x)
        #y_vals.append(y)
        #t_vals.append(100*atype)
        xp = np.append(xp, x)
        yp = np.append(yp, y)
        colsp=np.append(colsp,ncols)
    
    #upos=np.concatenate(xp,yp,1)
    scatter.set_offsets(np.transpose([xp,yp]))
    #scatter.set_array(colsp)
    print(frame_number)    





# In[16]:


#setupmodelparams
ecoag = EcoagParams(NP=5, width=200, height=200)
print(ecoag.NP)       # Access
ecoag.pseparation = 0.9     # Modify
ecoag.speed = 3

#The environmental parameters and social parameters are set in the model initialisation these are used to seed a field array value which defines a value of these at each point of the model i.e. a 2d array of values the initial parameters define a 
# Next tasks
# * define the functions which set up initialenvironment and initialsocial
# * define minerals, factories, people, trees, plants, minerals, creatures
# * set up rules for evolving and changing social parameters
# * set up rules for changing environmental parameters

# In[17]:


#setup environmental params
env = EnvironmentalParams(carbon_level=420, biodiversity_index=0.75)
print(env.soil_health)
env.carbon_level=10


# In[18]:


#setup socialparams
soc = SocialParams(food_access=0.85, income_distribution=0.65)
print(soc.education_level)
soc.health_index=0.7


# In[19]:


model = EcoagModel(4, 5, 8, 3, 3,3, 100, 100, speed=10, vision=100, pseparation=4, bseparation=1, qseparation=10,params=ecoag,envparams=env,socparams=soc)
model.space._build_agent_cache()



# In[20]:


#for i in range(8):
#    #if len(model.space._agent_points) == 0:
#    #    print("Warning: No agents placed in space yet.")
#    #else:
#    model.step()
#    print(model.time)
#    #draw_boids(model)





#draw_boids(model)

model.step


x_vals = []
y_vals = []
t_vals=[]
cols = []
    
for boid in model.agents:
    x, y = boid.pos
    if boid.atype == 0:
        atype=1
        cols.append('red')
    if boid.atype == 1:
        cols.append('green')
        atype=10
    if boid.atype == 2:
        cols.append('blue')
        atype=50
    #cols.append(1)
    x_vals.append(x)
    y_vals.append(y)
    t_vals.append(100*atype)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
scatter=ax.scatter(x_vals, y_vals, s=t_vals, c=cols, alpha=0.5)

print("start animation")    
anim = FuncAnimation(fig, update, interval=10)
plt.show()  









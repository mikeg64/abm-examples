# -*- coding: utf-8 -*-
"""
Runecoagagents
Created on Fri Oct  3 22:26:45 2025

@author: mikeg
"""


import random

import numpy as np
import math

import matplotlib.pyplot as plt
%matplotlib inline
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



#setupmodelparams
ecoag = EcoagParams(NP=5, width=200, height=200)
print(ecoag.NP)       # Access
ecoag.pseparation = 0.9     # Modify


#setup environmental params
env = EnvironmentalParams(carbon_level=420, biodiversity_index=0.75)
print(env.soil_health)
env.carbon_level=10


#setup socialparams
soc = SocialParams(food_access=0.85, income_distribution=0.65)
print(soc.education_level)
soc.health_index=0.7


model = EcoagModel(4, 5, 8, 3, 3,3, 100, 100, speed=1, vision=100, pseparation=4, bseparation=1, qseparation=10,params=ecoag,envparams=env,socparams=soc)
model.space._build_agent_cache()




for i in range(8):
    #if len(model.space._agent_points) == 0:
    #    print("Warning: No agents placed in space yet.")
    #else:
    model.step()
    print(model.time)
    #draw_boids(model)









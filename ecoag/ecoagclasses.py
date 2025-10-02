# -*- coding: utf-8 -*-
"""
Hive Classes

Created on Sun Jul  3 12:36:05 2022
@author: louis
"""

import random

import numpy as np
import math

import matplotlib.pyplot as plt
#%matplotlib inline  #for the jupyter notebook

from mesa import Model, Agent
#from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
#from mesa.time import RandomActivation
from ecoagagents import *
from humanag import *

#from mesa.datacollection import DataCollector
#from mesa.batchrunner import BatchRunner
class EnvironmentalParams:
    def __init__(self,
                 carbon_level=400,           # ppm
                 biodiversity_index=0.8,     # 0–1 scale
                 soil_health=0.7,            # 0–1 scale
                 water_availability=0.6,     # 0–1 scale
                 pollution_rate=0.2):        # 0–1 scale
        self.carbon_level = carbon_level
        self.biodiversity_index = biodiversity_index
        self.soil_health = soil_health
        self.water_availability = water_availability
        self.pollution_rate = pollution_rate


class SocialParams:
    def __init__(self,
                 food_access=0.9,            # 0–1 scale
                 education_level=0.85,       # 0–1 scale
                 income_distribution=0.7,    # Gini-like index
                 health_index=0.8,           # 0–1 scale
                 housing_security=0.75,
                 peace_justice=0.5):     # 0–1 scale
        self.food_access = food_access
        self.education_level = education_level
        self.income_distribution = income_distribution
        self.health_index = health_index
        self.housing_security = housing_security
        self.peace_justice = peace_justice

class EcoagParams:
    def __init__(self,
                 NP=10, NB=5, NQ=3,NA=3,NF=3,NM=3,
                 width=100, height=100,
                 speed=0.1, vision=5,
                 pseparation=1, bseparation=1, qseparation=1):
        self.NP = NP  # Number of plants
        self.NB = NB  # Number of humans
        self.NQ = NQ  # Number of trees or other agents
        self.NF=NF #initial number of factories is zeros humans build them
        self.NA = NA
        self.NM=NM
        self.width = width
        self.height = height
        self.speed = speed
        self.vision = vision
        self.pseparation = pseparation
        self.bseparation = bseparation
        self.qseparation = qseparation





class EcoagModel(Model):
    '''
    Hive model class. Handles agent creation, placement and scheduling.
    '''

    def __init__(self, NP, NB, NQ, NF, NA,NM, width, height, speed, vision, pseparation, bseparation, qseparation, params: EcoagParams, envparams: EnvironmentalParams, socparams: SocialParams):
        '''
        Create a new Flockers model.

        Args:
            N: Number of Boids
            width, height: Size of the space.
            speed: How fast should the Boids move.
            vision: How far around should each Boid look for its neighbors
            separtion: What's the minimum distance each Boid will attempt to
                       keep from any other
            NP plants
            NB human
            NQ trees
            NA creatures/animals
            NF factories (initially zero)
            NM minerals
            
        '''
        
        #ContinuousSpace(x_max: float, y_max: float, torus: bool, x_min: float = 0, y_min: float = 0)
        super().__init__()  # Required in Mesa 3.x
        self.params=params
        self.envparams=envparams
        self.socparams=socparams
        self.NP = params.NP
        self.NB = params.NB
        self.NQ = params.NQ
        self.NF=params.NF #initial number of factories is zeros humans build them
        self.NA = params.NA
        self.NM=params.NM
        
        self.vision = params.vision
        self.speed = params.speed
        self.qseparation = params.qseparation
        self.pseparation = params.pseparation
        self.bseparation = params.bseparation
        #self.schedule = RandomActivation(self)
        #self.agents.add(agent)
        self.space = ContinuousSpace(width, height, True, 10, 10)
        self.agents_initialized = False
        self.make_agents()
        self.agents_initialized = True
        self.intialsocial()
        self.initialenvironment()
        self.running = True
        self.time =0
        
        

    def make_agents(self):
        '''
        Create N agents, with random positions and starting headings.
        '''
        
        
        #    NP plants
        #    NB human
        #    NQ trees
        #    NA creatures/animals
        #    NF factories (initially zero)
        #    NM minerals
        
        print("vision "+str(self.vision))
        for i in range(self.NB):
            #x = random.random() * self.space.x_max
            #y = random.random() * self.space.y_max
            #pos = (x, y)
            pos = (random.uniform(0, self.params.width), random.uniform(0, self.params.height))
            ang=2*math.pi*np.random.random(1)
            heading=np.array((math.cos(ang),math.sin(ang)))
            #heading = np.random.random(2) * 2 - np.array((1, 1))
            #heading /= np.linalg.norm(heading)
            
            #self, unique_id, model, pos, fruit=5, health=5, speed=0.1, heading=None,
            #             vision1=5, separation=1, atype=0
            
            human_agent = human(i, self, pos, 5,5,self.speed, heading,self.vision,self.bseparation)
 
            
            self.agents.add(human_agent)
            #if human_agent.pos is not None:
            #    self.space.remove_agent(human_agent)
            self.space.place_agent(human_agent, pos)
            self.space.move_agent(human_agent,pos)
            #self.schedule.add(human)
            
            #print(f"Agent {human_agent.unique_id} placed at {human_agent.pos}")

        ntot=self.NB
        for i in range(self.NP):
            #x = random.random() * self.space.x_max
            #y = random.random() * self.space.y_max
            #pos = (x, y)
            pos = (random.uniform(0, self.params.width), random.uniform(0, self.params.height))
            
            #(self, unique_id, model, pos, fruit =5, vision = 2, separation=1, atype=1):
            plant_agent = Plant(i+ntot, self, pos, 5, self.vision, self.pseparation)
            #if plant_agent.pos is not None:
            #    self.space.remove_agent(plant_agent)
            self.space.place_agent(plant_agent, pos)
            #self.schedule.add(plant)
            self.space.move_agent(plant_agent,pos)
            self.agents.add(plant_agent)
        ntot+=self.NP
        for i in range(self.NQ):
            #x = random.random() * self.space.x_max
            #y = random.random() * self.space.y_max
            #pos = (x, y)
            pos = (random.uniform(0, self.params.width), random.uniform(0, self.params.height))
            
            #(self, unique_id, model, pos, proccessedfood=5, honey=5, vision=5, separation=1, atype=2)
            tree_agent = Tree(i+ntot, self, pos, 5, 5, self.vision,self.qseparation)
            #if factory_agent.pos is not None:
            #    self.space.remove_agent(factory_agent)
            self.space.place_agent(tree_agent, pos)
            #self.schedule.add(factory)
            self.space.move_agent(tree_agent,pos)
            self.agents.add(tree_agent)
        ntot+=self.NQ
        for i in range(self.NA):
             #x = random.random() * self.space.x_max
             #y = random.random() * self.space.y_max
             #pos = (x, y)
             pos = (random.uniform(0, self.params.width), random.uniform(0, self.params.height))
             ang=2*math.pi*np.random.random(1)
             heading=np.array((math.cos(ang),math.sin(ang)))
             #(self, unique_id, model, pos, fruit=5, health=5, speed=0.1, heading=None, vision=5, separation=1, atype=0)
             creature_agent = creature(i+ntot, self, pos, 5, 5, 0.1, heading, self.vision,self.qseparation)
             #if creature_agent.pos is not None:
             #    self.space.remove_agent(creature_agent)
             self.space.place_agent(creature_agent, pos)
             #self.schedule.add(factory)
             self.space.move_agent(creature_agent,pos)
             self.agents.add(creature_agent)
        ntot+=self.NA        
        print("factories="+str(self.NF))        
        for i in range(self.NF):
            #x = random.random() * self.space.x_max
            #y = random.random() * self.space.y_max
            #pos = (x, y)
            pos = (random.uniform(0, self.params.width), random.uniform(0, self.params.height))
            print("create factory "+str(i))
            #(self, unique_id, model, pos, proccessedfood=5, honey=5, vision=5, separation=1, atype=2)
            factory_agent = factory(i+ntot, self, pos, 5, 5, self.vision,self.qseparation)
            #if factory_agent.pos is not None:
            #    self.space.remove_agent(factory_agent)
            self.space.place_agent(factory_agent, pos)
            #self.schedule.add(factory)
            self.space.move_agent(factory_agent,pos)
            self.agents.add(factory_agent)
        ntot+=self.NF
        
        for i in range(self.NM):
             #x = random.random() * self.space.x_max
             #y = random.random() * self.space.y_max
             #pos = (x, y)
             pos = (random.uniform(0, self.params.width), random.uniform(0, self.params.height))
             #(self, unique_id, model, pos, fruit =5, vision = 2, separation=1, atype=1)
             mineral_agent = Mineral(i+ntot, self, pos, 5, self.vision, self.qseparation)
             #if mineral_agent.pos is not None:
             #    self.space.remove_agent(mineral_agent)
             self.space.place_agent(mineral_agent, pos)
             #self.schedule.add(factory)
             self.space.move_agent(mineral_agent,pos)
             self.agents.add(mineral_agent) 
        self.space._build_agent_cache()
        print("Agent points shape:", self.space._agent_points.shape)
        print("Agent points:", self.space._agent_points)


    def initialenvironment(self):
        pars=self.params
        envp=self.envparams
        socp=self.socparams
        
        self.water_availability= np.random.rand(pars.width, pars.height) * envp.water_availability

        
    def intialsocial(self):
        pars=self.params
        envp=self.envparams
        socp=self.socparams
        self.food_access= np.random.rand(pars.width, pars.height) * socp.food_access

    def step(self):
        self.time = self.time+1
        #self.schedule.step()
        self.agents.shuffle().do("step")


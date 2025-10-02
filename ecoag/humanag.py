# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 17:38:00 2025

@author: mikeg
"""

import random

import numpy as np
import math


# %matplotlib inline  #for the jupyter notebook

from mesa import Model, Agent
#from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
#from mesa.time import RandomActivation

from ecoagagents import *


class human(Agent):
    '''
    A human Boid-style flocker agent.

    The agent follows three behaviors to flock:
        - Cohesion: steering towards neighboring  factory agents.
        - Separation: avoiding getting too close to any other worker agent.
        - Alignment: try to fly in the same direction as the neighbors.

    humans have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and heading (a unit vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.
    '''

    def __init__(self, unique_id, model, pos, fruit=5, health=5, speed=0.1, heading=None,
                 vision1=5, separation=1, atype=0):
        super().__init__(model)
        #super().__init__(unique_id, model)
        '''
        Create a new human flocker agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            heading: numpy vector for the Boid's direction of movement.
            vision: Radius to look around for nearby factorys/Plants.
            separation: Minimum distance to maintain from other Boids.
        '''

        self.pos = pos
        self.fruit = fruit
        self.health = health
        self.fruit = fruit
        self.speed = speed
        self.atype = atype
        self.honey = 0
        self.randomstateflag = 0
        self.randomstatecycle = 6
        self.randomstatestep = 0
        self.stepswithfactory = 0
        self.model = model
        if heading is not None:
            self.heading = heading
        else:
            ang = 2*math.pi*np.random.random(1)
            self.heading = np.array((math.cos(ang), math.sin(ang)))
            #self.heading = 2*np.random.random(2)
            #center = np.array([-1, -1])
            # self.heading+=np.float64(center)
            #self.heading /= np.linalg.norm(self.heading)

        self.vision = vision1
        self.separation = separation
        
        #print("vis1"+str(vision1))
        #print("vis2"+str(self.vision))

    def cohere(self, neighbors):
        '''
        Return the vector toward the center of mass of the local neighbors.
        '''
        center = np.array([0.0, 0.0])
        count = 0
        for neighbor in neighbors:
            if neighbor.atype == 0:
                center += np.array(neighbor.pos)
                count = count + 1

        if count > 0:
            heading = center / count
        else:
            ang = 2*math.pi*np.random.random(1)
            heading = np.array((math.cos(ang), math.sin(ang)))
            # heading=2*np.random.random(2)
            #center = np.array([-1, -1])
            # self.heading+=np.float64(center)
            # heading/=np.linalg.norm(heading)
        return np.float64(heading)

    def cohereplant(self, neighbors):
        '''
        Return the vector toward the center of mass of the local neighbors.
        '''

        center = np.array([0.0, 0.0])
        count = 0
        nsep = 9e99
        nhead = np.random.random(2)
        for neighbor in neighbors:
            if neighbor.atype == 1:
                sepv = np.float64(np.array(neighbor.pos)-self.pos)
                sep = np.linalg.norm(sepv)
                if sep < nsep:
                    nsep = sep
                    nhead = sepv
                count = count + 1

        if count <= 0:
            ang = 2*math.pi*np.random.random(1)
            nhead = np.array((math.cos(ang), math.sin(ang)))
            # nhead=2*np.random.random(2)
            #center = np.array([-1, -1])
            # self.heading+=np.float64(center)
            # nhead/=np.linalg.norm(nhead)
        return nhead

    def coherefactory(self, neighbors):
        '''
        Return the vector toward the locationn of the nearest factory.
        '''

        center = np.array([0.0, 0.0])
        count = 0
        nsep = 9e99
        nhead = np.random.random(2)
        for neighbor in neighbors:
            if neighbor.atype == 2:
                sepv = np.float64(np.array(neighbor.pos)-self.pos)
                sep = np.linalg.norm(sepv)
                if sep < nsep:
                    nsep = sep
                    nhead = sepv
                count = count + 1

        if count <= 0:
            # nhead=2*np.random.random(2)
            #center = np.array([-1, -1])
            # self.heading+=np.float64(center)
            # nhead/=np.linalg.norm(nhead)
            ang = 2*math.pi*np.random.random(1)
            nhead = np.array((math.cos(ang), math.sin(ang)))

        return nhead

    def staywithfactory(self, neighbors):
        stay = 0
        for neighbor in neighbors:
            if neighbor.atype == 2:
                my_pos = self.pos
                their_pos = np.float64(np.array(neighbor.pos))
                dist = np.linalg.norm(my_pos - their_pos)
                if dist <= 15:
                    stay = 1

        return stay

    def separate(self, neighbors):
        '''
        Return a vector away from any neighbors closer than separation dist. Only do this if not another human
        and if not near a plant or a factory
        '''
        my_pos = np.float64(np.array(self.pos))
        sep_vector = np.float64(np.array([0, 0]))
        for neighbor in neighbors:
            if neighbor.atype == 0:
                their_pos = np.float64(np.array(neighbor.pos))
                dist = np.linalg.norm(my_pos - their_pos)
                if dist < self.separation:
                    sep_vector -= np.float64(their_pos - my_pos)
        return np.float64(sep_vector)

    def randommove(self):

        # put the human into a random search
        if np.random.random(1) > 0.8 and self.randomstateflag == 0:
            self.randomstateflag = 1
            self.randomstatestep = 0

        if self.randomstateflag == 1:
            self.randomstatestep += 1
            if self.randomstatestep > self.randomstatecycle:
                self.randomstateflag = 0
                self.randomstatestep = 0
            ang = 2*math.pi*np.random.random(1)
            self.heading = np.array((math.cos(ang), math.sin(ang)))

        return None

    def step(self):

        #print('here1')

        if not self.model.agents_initialized:
            return
        #if self.model.space._agent_points is None or len(self.model.space._agent_points) == 0:
        #    return  # Skip stepping if space is not ready

        #print('here2')
        '''
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        '''
        if not hasattr(self.model.space, "_agent_points") or self.model.space._agent_points.shape[1] != 2:
            print(f"Agent {self.unique_id} skipping step due to malformed agent_points")
            return
        print("human "+str(self.unique_id))
        #neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        #print("r is "+str(r))
        #r = 5
        r = self.vision  # or any other radius you define
        neighbors = self.model.space.get_neighbors(self.pos, r, include_center=False)

        # Do something with the neighbors
        #for neighbor in neighbors:
        #    # Example: print their ID
        #    print(f"Neighbor ID: {neighbor.unique_id}")

        if len(neighbors) > 0:
            cohere_vector = np.float64(self.cohere(neighbors))
            separate_vector = np.float64(self.separate(neighbors))
            self.heading += np.float64((cohere_vector +
                                        separate_vector))
            if np.int64(np.linalg.norm(self.heading)) != 0:
                self.heading = np.float64(self.heading) / np.int64(np.linalg.norm(self.heading))

        # need to check the heading very carefully as this is resulting in nan values!!@!!!!!!
        if len(neighbors) > 0:
            if self.honey < 2 and self.fruit > 1:
                self.heading = self.coherefactory(neighbors)
            if self.fruit <= 1 and self.honey > 2:
                self.heading = self.cohereplant(neighbors)

        # rule to set random heading for the humans
        self.randommove()

        if self.randomstateflag == 1:
            self.randomstatestep += 1
            if self.randomstatestep > self.randomstatecycle:
                self.randomstateflag = 0
                self.randomstatestep = 0
            ang = 2*math.pi*np.random.random(1)
            self.heading = np.array((math.cos(ang), math.sin(ang)))

        if len(neighbors) > 0:
            stay = self.staywithfactory(neighbors)
            if stay == 1:
                if self.stepswithfactory > 10:
                    self.speed = 0.5
                    self.stepswithfactory = 0
                else:
                    self.speed = 0
                    self.stepswithfactory += 1

        new_pos = np.array(self.pos) + self.heading * self.speed
        new_x, new_y = new_pos
        if math.isnan(new_x) or math.isnan(new_y):
            print("nan pos detected ")
            ang = 2*math.pi*np.random.random(1)
            self.heading = np.array((math.cos(ang), math.sin(ang)))
            #self.heading = np.random.random(2)
            new_pos = np.array(self.pos) + self.heading * self.speed
            new_x, new_y = new_pos
        else:
            if new_x > self.model.space.x_max:
                new_x = self.model.space.x_min
            if new_x < self.model.space.x_min:
                new_x = self.model.space.x_max
            if new_y > self.model.space.y_max:
                new_y = self.model.space.y_min
            if new_y < self.model.space.y_min:
                new_y = self.model.space.y_max
            self.model.space.move_agent(self, (new_x, new_y))

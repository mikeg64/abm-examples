# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 17:12:36 2025

@author: mikeg
"""
import random

import numpy as np
import math


#%matplotlib inline  #for the jupyter notebook

from mesa import Model, Agent
#from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
#from mesa.time import RandomActivation
from humanag import * 
            
class Plant(Agent):
    '''
    A Plant agent.

    Plant produes fruit humans take fruit from plant

    Separation is their desired minimum distance from
    any other Boid.
    '''
    
    def __init__(self, unique_id, model, pos, fruit =5, vision = 2, separation=1, atype=1):
        super().__init__( model)
        '''
        Create a new Plant agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            separation: Minimum distance to maintain from other Boids to give fruit.
        '''
        
        self.pos = pos
        self.separation = separation
        self.fruit = 5
        self.atype = atype
        self.vision = vision
        self.time = 0
        self.numneighbours=0


    def step(self):
        '''
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        '''
        if not self.model.agents_initialized:
            return  # Skip stepping until agents are placed
        #if self.model.space._agent_points is None or len(self.model.space._agent_points) == 0:
        #    return  # Skip stepping if space is not ready

        #neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        
        r = self.vision  # or any other radius you define
        neighbors = self.model.space.get_neighbors(self.pos, r, include_center=False)
        print("plant "+str(self.unique_id))
        # Do something with the neighbors
        for neighbor in neighbors:
            # Example: print their ID
            print(f"Neighbor ID: {neighbor.unique_id}")
        
        
        time=self.model.time
        my_pos = np.array(self.pos)
        '''
        self.numneighbours=0
        for neighbor in neighbors:
            if neighbor.atype == 0:
                their_pos = np.array(neighbor.pos)
                dist = np.linalg.norm(my_pos - their_pos)
                #self.separation=8
                #print (" before transfer", dist, self.separation)
                if dist < self.separation:
                    self.numneighbours+=1        
        
        for neighbor in neighbors:
            if neighbor.atype == 0:
                their_pos = np.array(neighbor.pos)
                dist = np.linalg.norm(my_pos - their_pos)
                #self.separation=8
                #print (" before transfer", dist, self.separation)
                if dist < self.separation:
                    #print("fruit transfer", self.fruit, neighbor.fruit)
                    if self.fruit > 0:
                        neighbor.fruit = neighbor.fruit + 2
                        self.fruit = self.fruit - 2
                        if self.fruit > 0 and neighbor.fruit<5 and self.numneighbours<2:
                            neighbor.speed=0.05
                        else:
                            neighbor.speed=0.5
                        #print ('fruit transfered', self.fruit, neighbor.fruit)
        '''
        
        if self.model.time%2 == 0:
            #print("plant hooray more fruit")
            self.fruit=self.fruit+random.random()/20




class Tree(Agent):
    '''
    A Tree agent.
   live a long time very good at cleaning the environment
   good raw material for manufacturing
    Plant produes fruit humans take fruit from plant

    Separation is their desired minimum distance from
    any other Boid.
    '''
    
    def __init__(self, unique_id, model, pos, fruit =5, vision = 2, separation=1, atype=1):
        super().__init__( model)
        '''
        Create a new Plant agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            separation: Minimum distance to maintain from other Boids to give fruit.
        '''
        #print("create tree "+str(self.unique_id))
        self.pos = pos
        self.separation = separation
        self.fruit = 5
        self.atype = atype
        self.vision = vision
        self.time = 0
        self.numneighbours=0


    def step(self):
        '''
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        '''
        if not self.model.agents_initialized:
            return  # Skip stepping until agents are placed
        #if self.model.space._agent_points is None or len(self.model.space._agent_points) == 0:
        #    return  # Skip stepping if space is not ready

        #neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        print("tree "+str(self.unique_id))
        r = self.vision  # or any other radius you define
        neighbors = self.model.space.get_neighbors(self.pos, r, include_center=False)

        # Do something with the neighbors
        for neighbor in neighbors:
            # Example: print their ID
            print(f"Neighbor ID: {neighbor.unique_id}")
        
        
        time=self.model.time
        my_pos = np.array(self.pos)
        '''
        self.numneighbours=0
        for neighbor in neighbors:
            if neighbor.atype == 0:
                their_pos = np.array(neighbor.pos)
                dist = np.linalg.norm(my_pos - their_pos)
                #self.separation=8
                #print (" before transfer", dist, self.separation)
                if dist < self.separation:
                    self.numneighbours+=1        
        
        for neighbor in neighbors:
            if neighbor.atype == 0:
                their_pos = np.array(neighbor.pos)
                dist = np.linalg.norm(my_pos - their_pos)
                #self.separation=8
                #print (" before transfer", dist, self.separation)
                if dist < self.separation:
                    #print("fruit transfer", self.fruit, neighbor.fruit)
                    if self.fruit > 0:
                        neighbor.fruit = neighbor.fruit + 2
                        self.fruit = self.fruit - 2
                        if self.fruit > 0 and neighbor.fruit<5 and self.numneighbours<2:
                            neighbor.speed=0.05
                        else:
                            neighbor.speed=0.5
                        #print ('fruit transfered', self.fruit, neighbor.fruit)
        '''
        
        if self.model.time % 1 == 0:
            print("tree hooray more fruit")
            self.fruit=self.fruit+random.random()/20


            
class factory(Agent):
    '''
    A factory agent.

    Uses fruit to make proccessedfood
    get fruit from human
    factory make proccessedfood
    factory give proccessedfood to human

    Separation is their desired minimum distance from
    any other Boid.
    '''
    
    def __init__(self, unique_id, model, pos, proccessedfood=5, honey=5, vision=5, separation=1, atype=2):
        super().__init__(model)
        '''
        Create a new factory agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            heading: numpy vector for the Boid's direction of movement.
            vision: Radius to look around for nearby factorys/Plants.
            separation: Minimum distance to maintain from other Boids.
        '''
        print("create factory"+str(unique_id))
        self.pos = pos
        self.separation = separation
        self.proccessedfood = proccessedfood
        self.honey = honey
        self.atype = atype
        self.vision = vision

    def step(self):
        '''
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        '''
        if not self.model.agents_initialized:
            return  # Skip stepping until agents are placed
        #if self.model.space._agent_points is None or len(self.model.space._agent_points) == 0:
        #    return  # Skip stepping if space is not ready

        #neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        nworkers=0
        health=0
        my_pos = np.array(self.pos)
        #self.separation = 10
        print("factory "+str(self.unique_id))
        r = self.vision  # or any other radius you define
        neighbors = self.model.space.get_neighbors(self.pos, r, include_center=False)

        # Do something with the neighbors
        for neighbor in neighbors:
            # Example: print their ID
            print(f"Neighbor ID: {neighbor.unique_id}")
        '''
        for neighbor in neighbors:
            if neighbor.atype == 0:
                their_pos = np.array(neighbor.pos)
                dist = np.linalg.norm(my_pos - their_pos)
                if dist < 15:
                    nworkers=nworkers+1
                    health=health+neighbor.health
                    if neighbor.fruit > 0:
                        neighbor.fruit = neighbor.fruit - 1
                        self.proccessedfood = self.proccessedfood + 1
                if dist < 15:
                    nworkers=nworkers+1
                    print("Honey before",self.honey, neighbor.honey)
                    #if self.honey > 0:
                    neighbor.honey = neighbor.honey + 1
                    self.honey = self.honey - 1
                    print("honey given", self.honey, neighbor.honey, self.unique_id)
                    if neighbor.fruit>0:
                        neighbor.speed = 0
                    else: 
                        neighbor.speed = 0.1
    
        #calculate number of workers near and use to make honey
        if nworkers>0:
            if self.proccessedfood>0:
                self.honey=self.honey+self.proccessedfood*(health/nworkers)
                self.proccessedfood=self.proccessedfood-nworkers
        '''
            
class creature(Agent):
    '''
    A creature Boid-style flocker agent.
    The agent follows three behaviors to flock:
        - Cohesion: steering towards neighboring  queen agents.
        - Separation: avoiding getting too close to any other worker agent.
        - Alignment: try to fly in the same direction as the neighbors.
    creatures have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and heading (a unit vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.
    '''
    #creature(i+self.NB+self.NP, self, pos, 5, 5, self.vision,self.qseparation)
    def __init__(self, unique_id, model, pos, fruit=5, health=5, speed=0.1, heading=None,
                 vision=5, separation=1, atype=0):
        super().__init__( model)
        '''
        Create a new creature flocker agent.
        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            heading: numpy vector for the Boid's direction of movement.
            vision: Radius to look around for nearby Queens/Plants.
            separation: Minimum distance to maintain from other Boids.
        '''
        
        self.pos = pos
        self.fruit = fruit
        self.health = health
        self.fruit = fruit
        self.speed = speed
        self.atype=atype
        self.honey=0
        self.randomstateflag = 0
        self.randomstatecycle =6
        self.randomstatestep = 0
        self.stepswithqueen=0
        self.model=model
        if heading is not None:
            self.heading = heading
        else:
            ang=2*math.pi*np.random.random(1)
            self.heading=np.array((math.cos(ang),math.sin(ang)))
            #self.heading = 2*np.random.random(2)
            #center = np.array([-1, -1])
            #self.heading+=np.float64(center)
            #self.heading /= np.linalg.norm(self.heading)

        self.vision = vision
        self.separation = separation

    def cohere(self, neighbors):
        '''
        Return the vector toward the center of mass of the local neighbors.
        '''
        center = np.array([0.0, 0.0])
        count=0
        for neighbor in neighbors:
            if neighbor.atype == 0:
                center += np.array(neighbor.pos)
                count = count +1
                
        if count>0: 
            heading=center / count
        else:
            ang=2*math.pi*np.random.random(1)
            heading=np.array((math.cos(ang),math.sin(ang)))
            #heading=2*np.random.random(2)
            #center = np.array([-1, -1])
            #self.heading+=np.float64(center)            
            #heading/=np.linalg.norm(heading)
        return np.float64(heading)
    
    def cohereplant(self, neighbors):
        '''
        Return the vector toward the center of mass of the local neighbors.
        '''

        center = np.array([0.0, 0.0])
        count=0
        nsep=9e99
        nhead=np.random.random(2)
        for neighbor in neighbors:
            if neighbor.atype == 1:
                sepv = np.float64(np.array(neighbor.pos)-self.pos)
                sep=np.linalg.norm(sepv)
                if sep<nsep:
                    nsep=sep
                    nhead=sepv
                count = count +1
 
        if count<=0: 
            ang=2*math.pi*np.random.random(1)
            nhead=np.array((math.cos(ang),math.sin(ang)))
            #nhead=2*np.random.random(2)
            #center = np.array([-1, -1])
            #self.heading+=np.float64(center)            
            #nhead/=np.linalg.norm(nhead)
        return nhead
          
        

    def coherequeen(self, neighbors):
        '''
        Return the vector toward the locationn of the nearest queen.
        '''

        center = np.array([0.0, 0.0])
        count=0
        nsep=9e99
        nhead=np.random.random(2)
        for neighbor in neighbors:
            if neighbor.atype == 2:
                sepv = np.float64(np.array(neighbor.pos)-self.pos)
                sep=np.linalg.norm(sepv)
                if sep<nsep:
                    nsep=sep
                    nhead=sepv
                count = count +1
                
        if count<=0: 
            #nhead=2*np.random.random(2)
            #center = np.array([-1, -1])
            #self.heading+=np.float64(center)            
            #nhead/=np.linalg.norm(nhead)
            ang=2*math.pi*np.random.random(1)
            nhead=np.array((math.cos(ang),math.sin(ang)))

        return nhead
    
    
    def staywithqueen(self, neighbors):
        stay=0
        for neighbor in neighbors:
            if neighbor.atype == 2:
                my_pos=self.pos
                their_pos = np.float64(np.array(neighbor.pos))
                dist = np.linalg.norm(my_pos - their_pos)
                if dist <= 15:
                    stay=1
                   
        return stay

    
    def separate(self, neighbors):
        '''
        Return a vector away from any neighbors closer than separation dist. Only do this if not another creature
        and if not near a plant or a queen
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

        #put the creature into a random search
        if np.random.random(1)>0.8 and self.randomstateflag == 0:
            self.randomstateflag=1
            self.randomstatestep=0
            
           
        if self.randomstateflag == 1:
            self.randomstatestep += 1
            if self.randomstatestep>self.randomstatecycle:
                self.randomstateflag=0
                self.randomstatestep=0
            ang=2*math.pi*np.random.random(1)
            self.heading=np.array((math.cos(ang),math.sin(ang)))        
        
        return None
    


    def step(self):
        '''
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        '''
        if not self.model.agents_initialized:
                return
        #if self.model.space._agent_points is None or len(self.model.space._agent_points) == 0:
        #        return  # Skip stepping if space is not ready

        #neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        r = self.vision  # or any other radius you define
        neighbors = self.model.space.get_neighbors(self.pos, r, include_center=False)
        print("creature "+str(self.unique_id))
        # Do something with the neighbors
        for neighbor in neighbors:
            # Example: print their ID
            print(f"Neighbor ID: {neighbor.unique_id}")
        '''
        if len(neighbors) > 0:
            cohere_vector = np.float64(self.cohere(neighbors))
            separate_vector = np.float64(self.separate(neighbors))
            self.heading += np.float64((cohere_vector +
                             separate_vector ))
            self.heading = np.float64(self.heading)/np.int64(np.linalg.norm(self.heading))
        
        
        
        #need to check the heading very carefully as this is resulting in nan values!!@!!!!!!            
        if len(neighbors) > 0:
            if self.honey < 2 and self.fruit>1:
                self.heading=self.coherequeen(neighbors)
            if self.fruit <= 1 and self.honey>2:
                self.heading=self.cohereplant(neighbors)


   

            
        #rule to set random heading for the creatures
        self.randommove()    
            
    
            
           
        if self.randomstateflag == 1:
            self.randomstatestep += 1
            if self.randomstatestep>self.randomstatecycle:
                self.randomstateflag=0
                self.randomstatestep=0
            ang=2*math.pi*np.random.random(1)
            self.heading=np.array((math.cos(ang),math.sin(ang)))
            
        if len(neighbors) > 0:
            stay=self.staywithqueen(neighbors)
            if stay==1:
                if self.stepswithqueen>10:
                    self.speed=0.5
                    self.stepswithqueen=0
                else:
                    self.speed=0
                    self.stepswithqueen+=1
        

            
        
        new_pos = np.array(self.pos) + self.heading * self.speed
        new_x, new_y = new_pos
        if math.isnan(new_x) or math.isnan(new_y):
            print("nan pos detected ")
            ang=2*math.pi*np.random.random(1)
            self.heading=np.array((math.cos(ang),math.sin(ang)))
            #self.heading = np.random.random(2)
            new_pos = np.array(self.pos) + self.heading * self.speed
            new_x, new_y = new_pos
        else:
            if new_x>self.model.space.x_max:
                new_x=self.model.space.x_min
            if new_x<self.model.space.x_min:
                new_x=self.model.space.x_max
            if new_y>self.model.space.y_max:
                new_y=self.model.space.y_min
            if new_y<self.model.space.y_min:
                new_y=self.model.space.y_max
            self.model.space.move_agent(self, (new_x, new_y))
        '''
            
 
            
class Mineral(Agent):
    '''
    A Mineral agent.
    Mineral produes fruit creatures take fruit from plant
    Separation is their desired minimum distance from
    any other Boid.
    '''
    
    def __init__(self, unique_id, model, pos, fruit =5, vision = 2, separation=1, atype=1):
        super().__init__( model)
        '''
        Create a new Plant agent.
        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            separation: Minimum distance to maintain from other Boids to give fruit.
        '''
        
        self.pos = pos
        self.separation = separation
        self.fruit = 5
        self.atype = atype
        self.vision = vision
        self.time = 0
        self.numneighbours=0


    def step(self):
        '''
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        '''
        
        if not self.model.agents_initialized:
                return
        #if self.model.space._agent_points is None or len(self.model.space._agent_points) == 0:
        #        return  # Skip stepping if space is not ready


        #neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        time=self.model.time
        my_pos = np.array(self.pos)
        
        r = self.vision  # or any other radius you define
        neighbors = self.model.space.get_neighbors(self.pos, r, include_center=False)
        print("mineral "+str(self.unique_id))
        '''
        self.numneighbours=0
        for neighbor in neighbors:
            if neighbor.atype == 0:
                their_pos = np.array(neighbor.pos)
                dist = np.linalg.norm(my_pos - their_pos)
                #self.separation=8
                #print (" before transfer", dist, self.separation)
                if dist < self.separation:
                    self.numneighbours+=1        
        
        for neighbor in neighbors:
            if neighbor.atype == 0:
                their_pos = np.array(neighbor.pos)
                dist = np.linalg.norm(my_pos - their_pos)
                #self.separation=8
                #print (" before transfer", dist, self.separation)
                if dist < self.separation:
                    #print("fruit transfer", self.fruit, neighbor.fruit)
                    if self.fruit > 0:
                        neighbor.fruit = neighbor.fruit + 2
                        self.fruit = self.fruit - 2
                        if self.fruit > 0 and neighbor.fruit<5 and self.numneighbours<2:
                            neighbor.speed=0.05
                        else:
                            neighbor.speed=0.5
                        #print ('fruit transfered', self.fruit, neighbor.fruit)
        '''
        
        if self.model.time % 1 == 0:
            print("mineral hooray more fruit")
            self.fruit=self.fruit+random.random()/20


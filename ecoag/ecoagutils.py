# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 22:55:13 2025

@author: mikeg
"""

import json



from ecoagagents import *
from ecoagclasses import *
from humanag import *



def save_params_to_file(env_params, soc_params, ecoag_params, filename="params.json"):
    data = {
        "environmental": env_params.to_dict(),
        "social": soc_params.to_dict(),
        "ecoag": ecoag_params.to_dict()
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def load_params_from_file(filename="params.json"):
    with open(filename, "r") as f:
        data = json.load(f)
    env = EnvironmentalParams.from_dict(data["environmental"])
    soc = SocialParams.from_dict(data["social"])
    ecoag = EcoagParams.from_dict(data["ecoag"])
    return env, soc, ecoag



def save_humans_to_file(human_list, filename="humans.json"):
    data = [agent.to_dict() for agent in human_list]
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
        
        
def load_humans_from_file(model, filename="humans.json"):
    with open(filename, "r") as f:
        data = json.load(f)
    return [human.from_dict(agent_data, model) for agent_data in data]
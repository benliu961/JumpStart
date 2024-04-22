import gym
import random as rand

'''
Classes and function necessary for JSRL. 
Includes functions for curriculum and random switching bases approaches.

Authors: Eugene Liu and Benjamin Liu

'''

class JSRL():
    '''
    Creates an object to initial JSRL
    '''
    def __init__(self, exploration_policy, guide_policy, horizon:int) -> None:
        self.exploration_policy = exploration_policy
        self.guide_policy = guide_policy
        self.horizon = horizon

    def train_policy(self):
        pass

    def evaluate_policy(self):
        pass

    def exploration_policy_return(self):
        return self.exploration_policy
    
    def guide_policy_return(self):
        return self.guide_policy


def __random_switch(horizon: int, n: int) -> list:
    '''Helper function to determine guide steps based on random switching'''

    guide_steps = []

    for i in range(n):
        guide_steps.append(rand.randint(horizon))

    return guide_steps
    

def __curriculum(horizon: int, n:int) -> list:
    '''Helper function to determine guide steps based on curriculum'''

    constant = horizon/n
    current = horizon

    guide_steps = []
    while current != 0:
        guide_steps.append(current - constant)
        current = current - constant
    
    return guide_steps


def __evaluation(policy):
    '''Helper function that evaluates the combined policy'''
    pass

def update(policy):
    '''Helper function that updates the exploration policy based on learned trajectory'''
    pass

def trainer(object: JSRL, horizon: int, n:int, environment, random_switch=False):


    # initializing guide steps based on either random_switch or curriculum
    if random_switch:
        guide_steps = __random_switch(horizon, n)
    
    else:
        guide_steps = __curriculum(horizon, n)

    # main training loop
    for i, guide_step in enumerate(guide_steps):
        
        # getting policy
        guide_policy = JSRL.guide_policy_return()
        exploration_policy = JSRL.exploration_policy_return()

        combine_policy = (i + 1)/len(guide_steps) * exploration_policy + (1 - (i + 1)/len(guide_steps)) * guide_policy
        
        # setting up
        guided_steps = guide_step
        exploration_steps = horizon - guide_step

        combine_policy = (i + 1)/len(guide_steps) * exploration_policy + (1 - (i + 1)/len(guide_steps)) * guide_policy

        

        
        
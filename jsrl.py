import gym

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


def __random_switch() -> list:
    '''Helper function to determine guide steps based on random switching'''
    pass

def __curriculum() -> list:
    '''Helper function to determine guide steps based on curriculum'''
    pass
        


def trainer(object: JSRL, horizon: int, environment, random_switch=False):

    guide_policy = JSRL.guide_policy_return()
    exploration_policy = JSRL.exploration_policy_return()

    # initializing guide steps based on either random_switch or curriculum
    if random_switch:
        guide_steps = __random_switch(horizon)
    
    else:
        guide_steps = __curriculum(horizon)

    # main training loop
    for guide_step in guide_steps:
        
        guided_steps = guide_step
        exploration_steps = horizon - guide_step

        
        
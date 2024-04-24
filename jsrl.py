import gym
import random as rand
import torch
import numpy as np
import copy
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

    def exploration_agent_return(self):
        return self.exploration_policy
    
    def guide_agent_return(self):
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

def __epsilon_greedy_policy(env, q_values, epsilon):
        # Creating epsilon greedy probabilities to sample from.
        p = np.random.uniform(0, 1)
        if p < epsilon:
            return env.action_space.sample()
        else:
            return torch.argmax(q_values).item()
        
def __combined_policy(guide_agent, exp_agent, state, prob):
    '''getting action based on combined policy'''
    guide_qvals = guide_agent.q_network.model.forward((state.reshape(1, -1)))
    exp_qvals = exp_agent. q_network.model.forward((state.reshape(1, -1)))

    p = np.random.uniform(0, 1)
    if p < prob:
        return torch.argmax(guide_qvals).item()
    else:
        return torch.argmax(exp_qvals).item()

def __evaluation(environment, guide_agent, exp_agent, prob, episodes=100):
    '''Helper function that evaluates the combined policy'''
    
    state = torch.from_numpy(environment.reset())

    rewards = 0
    i = 0

    while i < episodes:
        action = __combined_policy(guide_agent, exp_agent, state, prob)
        next_state, reward, done, info = environment.step(action)
        rewards += reward
        
        state = next_state
    
    return rewards

def trainer(object: JSRL, horizon: int, n:int, environment, epsilon, random_switch=False):

    # initializing guide steps based on either random_switch or curriculum
    if random_switch:
        guide_steps = __random_switch(horizon, n)
    
    else:
        guide_steps = __curriculum(horizon, n)

    # main training loop
    for i, guide_step in guide_steps:
        
        go = False
        reward = -np.Infinity
        rewards = []

        while go is False:
            # getting agents
            guide_agent = object.guide_agent_return()
            exploration_agent = object.exploration_agent_return()

            # getting states and q values from guide agent
            state = torch.from_numpy(environment.reset())
            q_values_guide = guide_agent.q_network.model.forward((state.reshape(1, -1)))

            # sampling trajectories from guide policy and training exploration agent on sampled data
            i = 0
            while i < guide_step:
                action = __epsilon_greedy_policy(environment, q_values_guide, epsilon)
                next_state, reward, done, info = environment.step(action)
                next_state = torch.from_numpy(next_state)
                rewards += reward
                next_q_values = guide_agent.q_network.model.forward((next_state.reshape(1, -1)))

                exploration_agent.memory.append(state, action, reward, next_state, done)
                state = copy.deepcopy(next_state.detach())
                q_values_guide = copy.deepcopy(next_q_values.detach())

                exploration_agent.train_dqn()
                i += 1

            # exploration steps
            q_values_exp = exploration_agent.forward((next_state.reshape(1, -1))) # picking up from the last state
            while i < horizon:

                action = __epsilon_greedy_policy(environment, q_values_exp, epsilon)
                next_state, reward, done, info = environment.step(action)
                next_state = torch.from_numpy(next_state)
                rewards += reward
                next_q_values = exploration_agent.q_network.model.forward((next_state.reshape(1, -1)))

                exploration_agent.memory.append(state, action, reward, next_state, done)
                state = copy.deepcopy(next_state.detach())
                q_values_guide = copy.deepcopy(next_q_values.detach())

                exploration_agent.train_dqn()
                i += 1

            # evaluate policy
            prob = guide_step/horizon
            reward_eval = __evaluation(environment, guide_agent, exploration_agent, prob)

            if reward_eval > reward:
                reward = reward_eval
                rewards.append(reward)
                go = True

    return rewards
        

        

        
        
import os, copy, json, gym, torch
import matplotlib.pyplot as plt
from datetime import datatime
from IPython import display as ipythondisplay

import torch.nn as nn
'''
General DQN class that handles class declaration, training, and evaluation. Based off ECE433 HW4 solutions.

Authors: Eugene Liu and Benjamin Liu

'''

class DQN(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = torch.nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output),
        )

    def forward(self, x):
        probs = self.model(x)
        return probs
    
class QNetwork():
    # This class essentially defines the network architecture.
    # It is NOT the PyTorch Q-network model (nn.Module), but a wrapper
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, args, input, output, learning_rate):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        self.weights_path = 'models/%s/%s' % (args['env'], datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        # Network architecture.
        self.hidden = 128
        self.model = DQN(input, self.hidden, output)

        # Loss and optimizer.
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if args['model_file'] is not None:
            print('Loading pretrained model from', args['model_file'])
            self.load_model_weights(args['model_file'])

    def save_model_weights(self, step):
        # Helper function to save your model / weights.
        if not os.path.exists(self.weights_path): os.makedirs(self.weights_path)
        torch.save(self.model.state_dict(), os.path.join(self.weights_path, 'model_%d.h5' % step))

    def load_model_weights(self, weight_file):
        # Helper function to load model weights.
        self.model.load_state_dict(torch.load(weight_file))

class Replay_Memory():
    def __init__(self, state_dim, action_dim, memory_size=50000, burn_in=10000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) way to implement the memory is as a list of transitions.
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.states = torch.zeros((self.memory_size, state_dim))
        self.next_states = torch.zeros((self.memory_size, state_dim))
        self.actions = torch.zeros((self.memory_size, 1))
        self.rewards = torch.zeros((self.memory_size, 1))
        self.dones = torch.zeros((self.memory_size, 1))
        self.ptr = 0
        self.burned_in = False
        self.not_full_yet = True

    def append(self, states, actions, rewards, next_states, dones):
        self.states[self.ptr] = states
        self.actions[self.ptr, 0] = actions
        self.rewards[self.ptr, 0] = rewards
        self.next_states[self.ptr] = next_states
        self.dones[self.ptr, 0] = dones
        self.ptr += 1

        if self.ptr > self.burn_in:
            self.burned_in = True

        if self.ptr >= self.memory_size:
            self.ptr = 0
            self.not_full_yet = False

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        if self.not_full_yet:
            idxs = torch.from_numpy(np.random.choice(self.ptr, batch_size, False))
        else:
            idxs = torch.from_numpy(np.random.choice(self.memory_size, batch_size, False))

        states = self.states[idxs]
        next_states = self.next_states[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]
        return states, actions, rewards, next_states, dones
    
class DQN_Agent():
    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, args):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.

        # Inputs
        self.args = args
        self.environment_name = self.args['env']
        self.render = self.args['render']
        self.epsilon = args['epsilon']
        self.network_update_freq = args['network_update_freq']
        self.log_freq = args['log_freq']
        self.test_freq = args['test_freq']
        self.save_freq = args['save_freq']
        self.learning_rate = args['learning_rate']

        # Env related variables
        if self.environment_name == 'CartPole-v0':
            self.env = gym.make(self.environment_name, render_mode='rgb_array')
            self.discount_factor = 0.99
            self.num_episodes = 200
        elif self.environment_name == 'MountainCar-v0':
            self.env = gym.make(self.environment_name, render_mode='rgb_array')
            self.discount_factor = 0.999
            self.num_episodes = 10000
        else:
            raise Exception("Unknown Environment")

        # Other Classes
        print(self.env.observation_space.shape, self.env.action_space.n, self.learning_rate)
        self.q_network = QNetwork(args, self.env.observation_space.shape[0], self.env.action_space.n, self.learning_rate)
        self.target_q_network = QNetwork(args, self.env.observation_space.shape[0], self.env.action_space.n, self.learning_rate)
        self.memory = Replay_Memory(self.env.observation_space.shape[0], self.env.action_space.n, memory_size=args['memory_size'])

        # Plotting
        self.rewards = []
        self.td_error = []
        self.batch = list(range(32))

        # Save hyperparameters
        self.logdir = 'logs/%s/%s' % (self.environment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        with open(self.logdir + '/hyperparameters.json', 'w') as outfile:
            json.dump((self.args), outfile, indent=4)

    def epsilon_greedy_policy(self, q_values, epsilon):
        # Creating epsilon greedy probabilities to sample from.
        p = np.random.uniform(0, 1)
        if p < epsilon:
            return self.env.action_space.sample()
        else:
            return torch.argmax(q_values).item()

    def greedy_policy(self, q_values):
        return torch.argmax(q_values).item()

    def train(self):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # When use replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        self.burn_in_memory()
        for step in range(self.num_episodes):
            # Generate Episodes using Epsilon Greedy Policy and train the Q network.
            self.generate_episode(policy=self.epsilon_greedy_policy, mode='train',
                epsilon=self.epsilon, frameskip=self.args['frameskip'])

            # Test the network.
            if step % self.test_freq == 0:
                test_reward, test_error = self.test(episodes=20)
                self.rewards.append([test_reward, step])
                self.td_error.append([test_error, step])

            # Update the target network.
            if step % self.network_update_freq == 0:
                self.hard_update()

            # Logging.
            if step % self.log_freq == 0:
                print("Step: {0:05d}/{1:05d}".format(step, self.num_episodes))

            # Save the model.
            if step % self.save_freq == 0:
                self.q_network.save_model_weights(step)

            step += 1
            self.epsilon_decay()

            # Render and save the video with the model.
            if step % int(self.num_episodes / 3) == 0 and self.args['render']:
                # test_video(self, self.environment_name, step)
                self.q_network.save_model_weights(step)

    def td_estimate (self, state, action):
        q_values = self.q_network.model(state)
        q_values = q_values.gather(1, action.long()).squeeze()
        return q_values

    def td_target (self, reward, next_state, done):
        with torch.no_grad():
            target_next_q = self.target_q_network.model(next_state)
            best_action = torch.argmax(target_next_q, axis=1)
            q_values = target_next_q.gather(1, best_action.unsqueeze(1)).squeeze()

            return reward.squeeze() + self.discount_factor * (1 - done.squeeze()) * q_values

    def train_dqn(self):
        # Sample from the replay buffer.
        state, action, rewards, next_state, done = self.memory.sample_batch(batch_size=32)

        # Network Input - S | Output - Q(S,A) | Error - |Y - Q(S,A)|
        # compute td targets and estimate for loss
        td_estimate = self.td_estimate(state, action)
        td_target = self.td_target(rewards, next_state, done)

        if (td_target.shape != td_estimate.shape):
            pass

        # compute loss and backpropogate
        loss = F.smooth_l1_loss(td_estimate, td_target)
        loss.backward()
        self.q_network.optim.step()
        self.q_network.optim.zero_grad()

        return loss

    def hard_update(self):
        self.target_q_network.model.load_state_dict(self.q_network.model.state_dict())

    def test(self, model_file=None, episodes=100):
        # Evaluate the performance of your agent over 100 episodes, by calculating cumulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        cum_reward = []
        td_error = []
        for count in range(episodes):
            reward, error = self.generate_episode(policy=self.epsilon_greedy_policy,
                mode='test', epsilon=0.05, frameskip=self.args['frameskip'])
            cum_reward.append(reward)
            td_error.append(error)
        cum_reward = torch.tensor(cum_reward)
        td_error = torch.tensor(td_error)

        print(cum_reward, td_error)
        print("\nTest Rewards: {0} | TD Error: {1:.4f}\n".format(torch.mean(cum_reward), torch.mean(td_error)))
        return torch.mean(cum_reward), torch.mean(td_error)

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        while not self.memory.burned_in:
            self.generate_episode(policy=self.epsilon_greedy_policy, mode='burn_in',
                epsilon=self.epsilon, frameskip=self.args['frameskip'])
        print("Burn Complete!")

    def generate_episode(self, policy, epsilon, mode='train', frameskip=1):
        """
        Collects one rollout from the policy in an environment.
        """
        done = False
        state = torch.from_numpy(self.env.reset())
        rewards = 0
        q_values = self.q_network.model.forward((state.reshape(1, -1)))
        td_error = []
        while not done:
            action = policy(q_values, epsilon)
            i = 0
            while (i < frameskip) and not done:
                next_state, reward, done, info = self.env.step(action)
                next_state = torch.from_numpy(next_state)
                rewards += reward
                i += 1
            next_q_values = self.q_network.model.forward((next_state.reshape(1, -1)))
            if mode in ['train', 'burn_in'] :
                self.memory.append(state, action, reward, next_state, done)
            else:
                td_error.append(abs(reward + self.discount_factor * (1 - done) * torch.max(next_q_values) - q_values))
            if not done:
                state = copy.deepcopy(next_state.detach())
                q_values = copy.deepcopy(next_q_values.detach())

            # Train the network.
            if mode == 'train':
                self.train_dqn()
        if td_error == []:
          return rewards, []
        return rewards, torch.mean(torch.stack(td_error))

    def plots(self):
        """
        Plots:
        1) Avg Cummulative Test Reward over 20 Plots
        2) TD Error
        """
        reward, time =  zip(*self.rewards)
        plt.figure(figsize=(8, 3))
        plt.subplot(121)
        plt.title('Cummulative Reward')
        plt.plot(time, reward)
        plt.xlabel('iterations')
        plt.ylabel('rewards')
        plt.legend()
        plt.ylim([0, None])

        loss, time =  zip(*self.td_error)
        plt.subplot(122)
        plt.title('Loss')
        plt.plot(time, loss)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.show()

    def epsilon_decay(self, initial_eps=1.0, final_eps=0.05):
        if(self.epsilon > final_eps):
            factor = (initial_eps - final_eps) / 10000
            self.epsilon -= factor
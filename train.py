from weights.strategy1_config import strategy1_config
from weights.strategy2_config import strategy2_config
from weights.strategy3_config import strategy3_config

import math
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def get_pole_sequence(config):
    lengths = config['pole_lengths'].copy() 
    if config['pole_order'] == 'random':
        return np.random.permutation(lengths) #randomize order of pole lengths
    # if you want another pole sequence, add it here
    pass 


def select_pole_length(episode, pole_sequence, config):
    max_episodes = config['episodes']
    sequence_length = len(pole_sequence)
    index = math.ceil(episode * (sequence_length / max_episodes))
    return pole_sequence[index]


def apply_reward_function(state, reward, done, config):
    pass

def train_step(q_network, target_network, replay_buffer, optimizer, config):
    """Single training step using experience replay"""
    batch = random.sample(replay_buffer, config['batch_size'])
    
    states = torch.tensor([s[0] for s in batch], dtype=torch.float32)
    actions = torch.tensor([s[1] for s in batch], dtype=torch.long)
    rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32)
    next_states = torch.tensor([s[3] for s in batch], dtype=torch.float32)
    dones = torch.tensor([s[4] for s in batch], dtype=torch.float32)
    
    current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

    with torch.no_grad():
        next_q = target_network(next_states).max(1)[0]
        target_q = rewards + config['gamma'] * next_q * (1 - dones)
    
    loss = nn.MSELoss()(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_dqn(config):
    env = gym.make('CartPole-v1')

    state_dim = 4
    action_dim = 2

    q_network = QNetwork(state_dim, action_dim)

    optimizer = optim.Adam(q_network.parameters(), lr=config['learning_rate'])
    replay_buffer = deque(maxlen=config['buffer_size'])

    epsilon = config['epsilon_start']

    pole_sequence = get_pole_sequence(config)

    max_episodes = config['max_episodes']
    for episode in range(max_episodes):

        pole_length = select_pole_length(episode, pole_sequence, config)
        env.unwrapped.length = pole_length
        
        state = env.reset()[0]
        done = False
        episode_reward = 0

        while not done:

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = q_network(state_tensor).argmax().item()
            
            next_state, reward, done, _, _ = env.step(action)

            #modified_reward = apply_reward_function(state, reward, done, config)
            
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            
            if len(replay_buffer) >= config['batch_size']:
                train_step(q_network, target_network, replay_buffer, optimizer, config)
        
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])


        if episode % 50 == 0:
                print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")
    
    torch.save(q_network.state_dict(), f'weights/{config["name"]}.pth')
    print(f"Model saved as {config['name']}.pth")

if __name__ == "__main__":
    print("Training Strategy 1")
    train_dqn(strategy1_config)
    
    print("\nTraining Strategy 2")
    train_dqn(strategy2_config)
    
    print("\nTraining Strategy 3")
    train_dqn(strategy3_config)
import math
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from weights.strategy1_config import strategy1_config
from weights.strategy2_config import strategy2_config
from weights.strategy3_config import strategy3_config
#from weights.strategy4_config import strategy4_config


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


def select_pole_length(episode, pole_lengths, config):
    order = config.get('pole_order', 'random')
    pls = list(pole_lengths)
    if order == 'random':
        return float(random.choice(pls))
    elif order == 'sequential':
        return float(pls[episode % len(pls)])
    elif order == 'curriculum_short_to_long':
        idx = min(episode, len(pls) - 1)
        return float(pls[idx])
    else:
        return float(random.choice(pls))


def get_pole_sequence(config):
    lengths = config['pole_lengths'].copy()
    if config['pole_order'] == 'random':
        return np.random.permutation(lengths)
    return lengths


def apply_reward_function(state, reward, done, config):
    if done:
        return -10  # penalty for failure

    cart_pos, cart_vel, pole_angle, pole_vel = state
    reward_type = config.get('reward_type', 'basic')

    if reward_type == 'basic':
        return reward
    elif reward_type == 'angle_based':
        return 1 - abs(pole_angle) / (math.pi / 2)
    elif reward_type == 'position_based':
        return 1 - abs(cart_pos) / 2.4
    elif reward_type == 'combined':
        reward_angle = 1 - abs(pole_angle) / (math.pi / 2)
        reward_position = 1 - abs(cart_pos) / 2.4
        return 0.7 * reward_angle + 0.3 * reward_position
    elif reward_type == 'creative':
        reward_angle = 1 - abs(pole_angle) / (math.pi / 2)
        reward_velocity = 1 - min(abs(pole_vel) / 3.0, 1.0)
        reward_position = 1 - abs(cart_pos) / 2.4
        return 0.5 * reward_angle + 0.3 * reward_velocity + 0.2 * reward_position

    return reward


def train_step(q_network, target_network, replay_buffer, optimizer, config):
    if config['name'] == "replay_buffer":
        batch = []
        if len(replay_buffer[2]) >= config['phase_3_batch_size'] * 1.5: #draws experiences from all 3 phases
            batch.extend(random.sample(replay_buffer[0], (config['phase_1_batch_size'])))
            batch.extend(random.sample(replay_buffer[1], (config['phase_2_batch_size'])))
            batch.extend(random.sample(replay_buffer[2], (config['phase_3_batch_size'])))
        elif len(replay_buffer[1]) >= config['phase_2_batch_size'] * 1.5: #devotes part of the batch size to train on experiences from phase 2
            batch.extend(random.sample(replay_buffer[0], (config['phase_1_batch_size'] + config['phase_3_batch_size'])))
            batch.extend(random.sample(replay_buffer[1], (config['phase_2_batch_size'])))
        else: #uses full batch size if only in phase 1
            batch.extend(random.sample(replay_buffer[0], (config['phase_1_batch_size'] + config['phase_2_batch_size'] + config['phase_3_batch_size'])))
        
    else:
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

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Online and target networks
    online_network = QNetwork(state_dim, action_dim)
    target_network = QNetwork(state_dim, action_dim)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(online_network.parameters(), lr=config['learning_rate'])
    
    if config['name'] == "replay_buffer":
        phase_1_buffer = deque(maxlen=config['phase_1_buffer_size'])
        phase_2_buffer = deque(maxlen=config['phase_2_buffer_size'])
        phase_3_buffer = deque(maxlen=config['phase_3_buffer_size'])
        replay_buffer = [phase_1_buffer, phase_2_buffer, phase_3_buffer]
    else: 
        replay_buffer = deque(maxlen=config['buffer_size'])

    epsilon = config['epsilon_start']
    pole_sequence = get_pole_sequence(config)

    print(f"\nStarting training: {config['name']} | Reward type: {config.get('reward_type')}")

    step_number = 0

    for episode in range(config['episodes']):
        pole_length = select_pole_length(episode, pole_sequence, config)
        env.unwrapped.length = pole_length

        state = env.reset()[0]
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            steps += 1
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = online_network(state_tensor).argmax().item()

            next_state, reward, terminated, _, _ = env.step(action)
            done = terminated 
            modified_reward = apply_reward_function(next_state, reward, done, config)

            if config['name'] == "replay_buffer":
                if step_number < 500:        #if the environment is currently in phase 1
                    replay_buffer[0].append((state, action, reward, next_state, done))
                elif step_number > 1000:     #if the environment is currently in phase 3
                    replay_buffer[2].append((state, action, reward, next_state, done))
                else:                   #if the environment is currently in phase 2
                    replay_buffer[1].append((state, action, reward, next_state, done))
            else:
                replay_buffer.append((state, action, modified_reward, next_state, float(done)))
            
            state = next_state
            episode_reward += modified_reward
            
            if config['name'] == "replay_buffer":
                if len(replay_buffer[0]) >= config['phase_1_batch_size'] + config['phase_2_batch_size'] + config['phase_3_batch_size']:
                    train_step(online_network, target_network, replay_buffer, optimizer, config)
                else:
                    print("buffer too small")
                
            else:
                if len(replay_buffer) >= config['batch_size']:
                    train_step(online_network, target_network, replay_buffer, optimizer, config)

            step_number += 1 

        # Epsilon decay
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])

        # Update target network
        if (episode + 1) % config['target_update'] == 0:
            target_network.load_state_dict(online_network.state_dict())

        print(f"Episode {episode+1:4d} | Reward: {episode_reward:6.2f} | Epsilon: {epsilon:.3f} | Steps: {steps}")

    torch.save(online_network.state_dict(), f'weights/{config["name"]}.pth')
    print(f"\nModel saved as weights/{config['name']}.pth")


if __name__ == "__main__":
    train_dqn(strategy1_config)
    train_dqn(strategy2_config)
    train_dqn(strategy3_config)
    #train_dqn(strategy4_config)

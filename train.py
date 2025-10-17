from weights.strategy1_config import strategy1_config
from weights.strategy2_config import strategy2_config
from weights.strategy3_config import strategy3_config

import math
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt



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


def select_pole_length(episode, pole_sequence, config):
    max_episodes = config['episodes']
    sequence_length = len(pole_sequence)
    index = int(episode * (sequence_length / max_episodes))
    return pole_sequence[index]
def get_pole_sequence(config):
    lengths = config['pole_lengths'].copy() 
    if config['pole_order'] == 'random':
        return np.random.permutation(lengths) #randomize order of pole lengths
    # if you want another pole sequence, add it here
    else:
        return lengths

def apply_reward_function(state, reward, done, config):
    if done:
        return -10  # Strong penalty for failure

    if config.get('reward_type') == 'basic':
        return reward

    cart_pos, cart_vel, pole_angle, pole_vel = state

    if config.get('reward_type') == 'angle_based':
        return 1 - abs(pole_angle) / (math.pi / 2)
def select_pole_length(episode, pole_lengths, config):
    """Pick a pole length for this episode based on the strategy."""
    order = config.get('pole_order', 'random')

    # Ensure we can sample even if pole_lengths is a numpy array
    pls = list(pole_lengths)

    if order == 'random':
        return float(random.choice(pls))
    elif order == 'sequential':
        return float(pls[episode % len(pls)])
    elif order == 'curriculum_short_to_long':
        idx = min(episode, len(pls) - 1)
        return float(pls[idx])
    else:
        # fallback
        return float(random.choice(pls))

    if config.get('reward_type') == 'position_based':
        return 1 - abs(cart_pos) / 2.4

    if config.get('reward_type') == 'combined':
        reward_angle = 1 - abs(pole_angle) / (math.pi / 2)
        reward_position = 1 - abs(cart_pos) / 2.4
        return 0.7 * reward_angle + 0.3 * reward_position

    if config.get('reward_type') == 'creative':
        reward_angle = 1 - abs(pole_angle) / (math.pi / 2)
        reward_velocity = 1 - min(abs(pole_vel) / 3.0, 1.0)
        reward_position = 1 - abs(cart_pos) / 2.4
        return 0.5 * reward_angle + 0.3 * reward_velocity + 0.2 * reward_position

    return reward


def train_step(q_network, target_network, replay_buffer, optimizer, config):
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

    q_network = QNetwork(state_dim, action_dim)
    target_network = QNetwork(state_dim, action_dim)
    target_network.load_state_dict(q_network.state_dict())

    # Target net init
    target_network = QNetwork(state_dim, action_dim)
    target_network.load_state_dict(q_network.state_dict())  
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=config['learning_rate'])
    if config['name'] == "replay_buffer":
        phase_1_buffer = deque(maxlen=config['phase_1_buffer_size'])
        phase_2_buffer = deque(maxlen=config['phase_2_buffer_size'])
        phase_3_buffer = deque(maxlen=config['phase_3_buffer_size'])
        replay_buffer = [phase_1_buffer, phase_2_buffer, phase_3_buffer]
    else: 
        replay_buffer = deque(maxlen=config['buffer_size'])

    epsilon = config['epsilon_start']

    pole_sequence = get_pole_sequence(config)

    max_episodes = 500

    print(f"\nStarting training: {config['name']} with reward type: {config.get('reward_type')}\n")
    print(max_episodes)
    for episode in range(1, max_episodes + 1):

    #lengths
    pole_lengths = config['pole_lengths']
    pole_sequence = get_pole_sequence(config)

    max_episodes = config['episodes']
    for episode in range(max_episodes):

        pole_length = select_pole_length(episode, pole_sequence, config)
        pole_length = select_pole_length(episode, pole_sequence, config)
        env.unwrapped.length = pole_length

        state = env.reset()[0]
        done = False
        episode_reward = 0
        step = 0
        step_number = 0 

        while not done:
            step += 1
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = q_network(state_tensor).argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            modified_reward = apply_reward_function(next_state, reward, done, config)

            replay_buffer.append((state, action, modified_reward, next_state, done))
            #modified_reward = apply_reward_function(state, reward, done, config)
            
            replay_buffer.append((state, action, reward, next_state, float(done)))
            state = next_state
            episode_reward += modified_reward

            if len(replay_buffer) >= config['batch_size']:
                train_step(q_network, target_network, replay_buffer, optimizer, config)

        # Decay epsilon
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])

        # Update target network every 10 episodes
        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())

        print(f"Episode {episode:4d} | Reward: {episode_reward:6.2f} | Epsilon: {epsilon:.3f} | Steps: {step}")
            
            step_number += 1

            if step_number % 100 == 0: #update target network every 100 steps (number can be changed)
                target_network.load_state_dict(q_network.state_dict())

        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])

        # Every N episodes, copy online network weights to target network for stability
        if (episode + 1) % config['target_update'] == 0:
            target_network.load_state_dict(q_network.state_dict())  # θ^- ← θ

    torch.save(q_network.state_dict(), f'weights/{config["name"]}.pth')
    print(f"\nModel saved as weights/{config['name']}.pth")


if __name__ == "__main__":
    train_dqn(strategy1_config)
    train_dqn(strategy2_config)
    train_dqn(strategy3_config)

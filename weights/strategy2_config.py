import numpy as np
strategy2_config = {
    'name': 'combined',
    'pole_lengths': np.linspace(0.4, 1.8, 15),
    'pole_order': 'random',
    'reward_function': 'range_based',  
    'reward_type': "combined",  
    'episodes': 1000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'learning_rate': 0.001,
    'gamma': 0.99,
    'batch_size': 64,
    'buffer_size': 10000,
    'target_update': 10,
}


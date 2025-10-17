import numpy as np
strategy3_config = {
    'name': 'pole_order',
    'pole_lengths': np.linspace(0.4, 1.8, 15),
    'pole_order': 'curriculum_short_to_long', 
    'reward_function': 'standard',
    'reward_type': "basic",
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
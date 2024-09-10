import torch 
import torch.optim as optim
import torch.nn.functional as F
import torch.nn
import gymnasium as gym
from replay_buffer import ReplayBufferDQN
import random
import numpy as np
import os
import time
from utils import exponential_decay
import typing
import config

class DQN:
    def __init__(self, env:typing.Union[gym.Env,gym.Wrapper],
                 model: torch.nn.Module,
                 model_kwargs: dict={},
                 lr: float=0.001, gamma:float = 0.99,
                 buffer_size: int=config.DQN['buffer_size'], 
                 batch_size: int=config.DQN['batch_size'],
                 loss_fn: str='mse_loss',
                 device: str='cpu',
                 seed: int=42,
                 epsilon_scheduler=exponential_decay(1,700,0.1),
                 save_path_weights:str=None,
                 save_path_history:str=None):
        """Initializes the DQN algorithm

        Args:
            env (gym.Env|gym.Wrapper): the environment to train on
            model (torch.nn.Module): the model to train
            model_kwargs (dict, optional): the keyword arguments to pass to the model. Defaults to {}.
            lr (float, optional): the learning rate to use in the optimizer. Defaults to 0.001.
            gamma (float, optional): discount factor. Defaults to 0.99
            buffer_size (int, optional): the size of the replay buffer. Defaults to 10000.
            batch_size (int, optional): the batch size. Defaults to 32.
            loss_fn (str, optional): the name of the loss function to use. Defaults to 'mse_loss'.
            device (str, optional): Defaults to 'cpu'.
            seed (int, optional): the seed to use for reproducibility. Defaults to 42.
            epsilon_scheduler ([type], optional): the epsilon scheduler to use, must have a __call__ method that returns a float between 0 and 1
            save_path (str, optional): Defaults to None.        
        """

        self.env = env
        self._set_seed(seed)

        self.observation_space = self.env.observation_space.shape
        self.model = model(
            self.observation_space,
            self.env.action_space.n, **model_kwargs
        ).to(device)
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

        self.replay_buffer = ReplayBufferDQN(buffer_size)
        self.batch_size = batch_size
        self.i_update = 0
        self.device = device
        self.epsilon_decay = epsilon_scheduler
        self.save_path_weights = save_path_weights if save_path_weights is not None else "./"
        self.save_path_history = save_path_history if save_path_history is not None else "./"

        if loss_fn == 'smooth_l1_loss':
            self.loss_fn = F.smooth_l1_loss
        elif loss_fn == 'mse_loss':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError('loss_fn must be either smooth_l1_loss or mse_loss')
        
    def train(self, n_episodes: int=1000, validate_every: int=100, n_validation_episodes: int=10,
              n_test_episodes: int=10, save_every: int=100):
        train_reward_history, train_loss_history = [], []
        val_reward_history, val_std_history = [], []
        best_val_reward = -np.inf

        # Create directories for the saved weights and run histories
        os.makedirs(self.save_path_weights, exist_ok=True)
        os.makedirs(self.save_path_history, exist_ok=True)

        for episode in range(n_episodes):
            # Reset the environment
            state,_ = self.env.reset()
            done = False
            truncated = False
            total_reward = 0
            i = 0
            loss = 0
            start_time = time.time()
            epsilon = self.epsilon_decay()

            while (not done) and (not truncated):
                # While the game is not over, take an action and add that experience to the replay buffer
                action = self._sample_action(state, epsilon)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

                # Optimize the model
                not_warm_starting, l = self._optimize_model()
                if not_warm_starting:
                    loss += l
                    epsilon = self.epsilon_decay()
                    i += 1
            if i!=0:
                # The optimization process has already started
                print(f"Episode: {episode} Time: {time.time()-start_time} Total Reward: {total_reward} Avg_Loss: {loss/i}")
                train_reward_history.append(total_reward)
                train_loss_history.append(loss/i)
            if episode%validate_every == validate_every-1:
                mean_reward, std_reward = self.validate(n_validation_episodes)
                print(f"Validation Mean Reward: {mean_reward} Validation Std Reward: {std_reward}")
                val_reward_history.append(mean_reward)
                val_std_history.append(std_reward)
                if mean_reward > best_val_reward:
                    best_val_reward = mean_reward
                    self._save('best')
            if episode%save_every == save_every-1:
                self._save(str(episode))
        
        self._save('final')
        self.load_model('best')
        mean_reward, std_reward = self.validate(n_test_episodes)
        print(f"Test Mean Reward: {mean_reward} Test Std Reward: {std_reward}")
        return train_reward_history, train_loss_history, val_reward_history, val_std_history
    
    
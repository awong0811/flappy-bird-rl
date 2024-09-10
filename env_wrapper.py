import cv2
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from utils import preprocess
import config

class EnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env:gym.Env,
        skip_frames:int=1,
        stack_frames:int=4,
        initial_no_op:int=0,
        do_nothing_action:int=0,
        **kwargs
    ):
        """
        Args:
            env (gym.Env): the original environment
            skip_frames (int, optional): the number of frames to skip, in other words we will
            repeat the same action for `skip_frames` steps. Defaults to 1.
            stack_frames (int, optional): the number of frames to stack, we stack 
            `stack_frames` frames to form the state and allow agent understand the motion of the car. Defaults to 4.
            initial_no_op (int, optional): the initial number of no-op steps to do nothing at the beginning of the episode. Defaults to 50.
            do_nothing_action (int, optional): the action index for doing nothing. Defaults to 0, which should be correct unless you have modified the 
            discretization of the action space.
        """
        super(EnvWrapper, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(stack_frames, config.DATA['resolution'], config.DATA['resolution']),
            dtype=np.float32
        )
        self.do_nothing_action = do_nothing_action

    def reset(self, **kwargs):
        # call the enviroment reset
        s, info = self.env.reset(**kwargs)

        # Do nothing for the next self.initial_no_op` steps
        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step(self.do_nothing_action)
        
        # Crop and resize the frame (we'll use the rendered frame, not the returned state)
        rgb_image = self.env.render()
        s = preprocess(rgb_image)

        # stack the frames to form the initial state
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [
        return self.stacked_state, info
    
    def step(self, action):
        '''
        Instead of reading in state frame by frame, we'll treat each state as a collection of frames.
        This allows us to not just read in the position of the bird but also its velocity.
        '''
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break

        # Crop and resize the frame (we'll use the rendered frame, not the returned state)
        rgb_image = self.env.render()
        s = preprocess(rgb_image)
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, reward, terminated, truncated, info
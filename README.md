# flAIppy bird: Reinforcement Learning (DQN) for Flappy Bird
OS: Windows 11 <br>
Python Version: 3.10.0 <br>
Setup: Make sure to install the flappy bird environment as shown below before installing requirements.txt.

This project was inspired by my ECE 239AS project. This project serves as a further exploration of DQN; I figured I could try DQN on some other games to familiarize myself with the start-to-finish process of implementing DQN, from the environment wrapper to the replay buffer to the entire training function. If things go well, I might consider an RL project for my capstone!

The idea of applying DQN to Flappy Bird is not new. There have been very successful [projects](https://www.youtube.com/watch?v=THhUXIhjkCM&t=302s) in the past, but I noticed most of them used simpler environments (i.e. removing the background, bird doesn't rotate). Thus, this not only was a great learning experience for me but provided a challenge to see if I could reproduce the same results in a newer version of Flappy Bird that we are familiar with. I'll update the README and provide more thorough documentation of the design process (mostly for my own educational benefit) as I go.

Here are the results for a non-optimized vanilla DQN agent after approximately 3000 episodes. My next agent will be trained on 100000 episodes, with epsilon tuning at the 50000 episode mark. Please note that while the environment is seeded for reproducibility, the seeds in my demo are randomly generated. I aim to be consistent and honest about my approach.

<br>

<p align="center">
  <img align="center" 
       src="https://github.com/awong0811/flappy-bird-rl/blob/main/imgs/flappy_bird_dqn_3000.gif?raw=true" 
       width="200"/> <br>
  <a href="https://youtu.be/zXcjuuonosw" target="_blank">Agent after 3000 training episodes</a> <br>
  <a href="https://youtu.be/q_ziL8iavNA?si=iALgVAg7HgXW2a-0" target="_blank">Agent after 50000 training episodes</a> <br>
  <a href="https://youtu.be/IQdAhJrQnyU" target="_blank">Agent after 10000 training episodes (with different epsilon-greedy decay rate)</a>
</p>


<!-- [Youtube](https://youtu.be/zXcjuuonosw) -->

Shoutout to [@Talendar](https://github.com/Talendar) for building the [original environment](https://github.com/Talendar/flappy-bird-gym) and [@markub3327](https://github.com/markub3327) for providing the [updated version](https://github.com/markub3327/flappy-bird-gymnasium). The forked README continues below.

# Flappy Bird for Gymnasium

![Python versions](https://img.shields.io/pypi/pyversions/flappy-bird-gymnasium)
[![PyPI](https://img.shields.io/pypi/v/flappy-bird-gymnasium)](https://pypi.org/project/flappy-bird-gymnasium/)
[![License](https://img.shields.io/github/license/markub3327/flappy-bird-gymnasium)](https://github.com/markub3327/flappy-bird-gymnasium/blob/master/LICENSE)

This repository contains the implementation of Gymnasium environment for
the Flappy Bird game. The implementation of the game's logic and graphics was
based on the [flappy-bird-gym](https://github.com/Talendar/flappy-bird-gym) project, by
[@Talendar](https://github.com/Talendar). 

## State space

The "FlappyBird-v0" environment, yields simple numerical information about the game's state as
observations representing the game's screen.

### `FlappyBird-v0`
There exist two options for the observations:  
1. option
* The LIDAR sensor 180 readings (Paper: [Playing Flappy Bird Based on Motion Recognition Using a Transformer Model and LIDAR Sensor](https://www.mdpi.com/1424-8220/24/6/1905))

2. option
* the last pipe's horizontal position
* the last top pipe's vertical position
* the last bottom pipe's vertical position
* the next pipe's horizontal position
* the next top pipe's vertical position
* the next bottom pipe's vertical position
* the next next pipe's horizontal position
* the next next top pipe's vertical position
* the next next bottom pipe's vertical position
* player's vertical position
* player's vertical velocity
* player's rotation

## Action space

* 0 - **do nothing**
* 1 - **flap**

## Rewards

* +0.1 - **every frame it stays alive**
* +1.0 - **successfully passing a pipe**
* -1.0 - **dying**
* −0.5 - **touch the top of the screen**

<br>

<p align="center">
  <img align="center" 
       src="https://github.com/markub3327/flappy-bird-gymnasium/blob/main/imgs/dqn.gif?raw=true" 
       width="200"/>
</p>

## Installation

To install `flappy-bird-gymnasium`, simply run the following command:

    $ pip install flappy-bird-gymnasium
    
## Usage

Like with other `gymnasium` environments, it's very easy to use `flappy-bird-gymnasium`.
Simply import the package and create the environment with the `make` function.
Take a look at the sample code below:

```python
import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()
```

## Playing

To play the game (human mode), run the following command:

    $ flappy_bird_gymnasium
    
To see a random agent playing, add an argument to the command:

    $ flappy_bird_gymnasium --mode random

To see a Deep Q Network agent playing, add an argument to the command:

    $ flappy_bird_gymnasium --mode dqn

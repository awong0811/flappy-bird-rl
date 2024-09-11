import cv2
import utils
import numpy as np
import flappy_bird_gymnasium
import gymnasium as gym
from env_wrapper import EnvWrapper
import torch
import model
import DQN
import os
import argparse
from tqdm import tqdm
import heapq
import random

def top_n_indices(numbers, n):
    # Get the top n largest values with their original indices
    top_n_values_with_indices = heapq.nlargest(n, enumerate(numbers), key=lambda x: x[1])    
    # Extract the indices
    top_indices = [index for index, value in top_n_values_with_indices]
    return top_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="visualize",
        description="Create a video of a trained agent navigating an environment"
    )
    parser.add_argument(
        'output_path',
        type=str,
        help = "Output path."
    )
    parser.add_argument(
        'ckpt',
        type = str,
        help = "Best model checkpoint."
    )
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['DQN','HardDQN','SoftDQN'],
        default='DQN',
        help = "Select from DQN, Hard Update DQN, and Soft Update DQN"
    )
    parser.add_argument(
        '--top',
        type = int,
        help = "Analyze 50 seeds to select the top # of seeds."
    )
    parser.add_argument(
        '-s', '--seed',
        type = int,
        nargs='*',
        help = "Seed(s) to use"
    )
    parser.add_argument(
        '-r','--random',
        action='store_true',
        help = "Use a random seed"
    )
    args = parser.parse_args()

    output_path = args.output_path
    ckpt = args.ckpt
    mode = args.mode
    top = args.top
    seed = args.seed
    use_random = args.random
    if seed is not None and top is not None or top is not None and use_random or use_random and seed is not None:
        raise Exception("Please only use one of the following arguments: --top, --seed, --random")
    min_seed = 0
    max_seed = 100000
    num_seeds_to_select = 50
    if use_random:
        random_seed = random.sample(range(min_seed, max_seed), 1)[0]
    if top is not None:
        random_seeds = random.sample(range(min_seed, max_seed), num_seeds_to_select)

    eval_env = gym.make(
        "FlappyBird-v0", audio_on=False, render_mode="rgb_array", use_lidar=False
    )
    eval_env = EnvWrapper(eval_env)

    # All models are seeded by default anyways for reproducibility
    if mode=='DQN':
        evalDQN = DQN.DQN(EnvWrapper(eval_env),
                        model.CNN,
                        device = 'cpu', seed=42)
    elif mode=='HardDQN':
        evalDQN = DQN.HardUpdateDQN(EnvWrapper(eval_env),
                        model.CNN,
                        device = 'cpu', seed=42)
    else:
        evalDQN = DQN.SoftUpdateDQN(EnvWrapper(eval_env),
                        model.CNN,
                        device = 'cpu', seed=42)

    evalDQN.model.load_state_dict(torch.load(ckpt, weights_only=True))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if top is None:
        if seed is None and not use_random:
            print('Analyzing seed...')
            total_rewards, frames = evalDQN.play_episode(0,True,42)
            frames = [frames]
            selected_seeds=[42]
        elif use_random:
            print('Analyzing seed...')
            total_rewards, frames = evalDQN.play_episode(0,True,random_seed)
            frames = [frames]
            selected_seeds=[random_seed]
        else:
            print('Analyzing seeds...')
            frames = []
            for s in tqdm(seed, unit='seed'):
                total_rewards, f = evalDQN.play_episode(0,True,s)
                frames.append(f)
            selected_seeds=seed
    else:
        frames_list = []
        rewards_list = []
        print('Analyzing seeds...')
        for seed in tqdm(random_seeds, unit='seeds'):
            total_rewards, frames = evalDQN.play_episode(0,True,seed)
            rewards_list.append(total_rewards)
            frames_list.append(frames)
        indices = top_n_indices(rewards_list, top)
        indices = np.array(indices[::-1])
        selected_seeds = []
        frames = []
        for i in indices:
            frames.append(frames_list[i])
            selected_seeds.append(random_seeds[i])
    total_frames = sum(len(sublist) for sublist in frames)
    filename = output_path
    fps = 30.0
    res = (frames[0][0].shape[1], frames[0][0].shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, res)
    print('Writing frames...')
    progress_bar = tqdm(total=total_frames, desc="Processing", unit="frames")
    for i in range(len(frames)):
        for f in frames[i]:
            seed = selected_seeds[i]
            frame = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f'Seed: {seed}', org=(3, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0,0,255), thickness=2)
            out.write(frame)
            progress_bar.update(1)
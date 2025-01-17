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
import random

### For ASCII art
from urllib.request import urlopen
import re
import sys
import platform
import subprocess
###

def check_internet_connection(host='google.com'):
    if platform.system()=='Windows':
        cmd = ['ping','-n','1',host]
    else:
        cmd = ['ping', '-c', '1', host]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode==0:
            if 'Reply' in result.stdout or 'bytes from' in result.stdout:
                return True
        return False
    except Exception as e:
        print(f"An error occured: {e}")
        return False

def get_ascii(text, font='graffiti'):
    url = r"http://www.network-science.de/ascii/ascii.php?TEXT=" + text + r"&x=32&y=13&FONT=" + font + r"&RICH=no&FORM=left&STRE=no&WIDT=150"
    
    f = urlopen(url)
    html = f.read().decode('utf-8')
    f.close()

    matched = re.search(r'<PRE>.*?</PRE>.*?<PRE>(.*?)</PRE>', html, re.DOTALL)
    if matched:
        ascii = matched.group(1)
        ascii=ascii.replace('&gt;', '>').replace('&lt;', '<')
        return ascii
    else:
        return None

def replace_transparent_pixels(image_path, bgr_color):
    # Read the image with the alpha channel
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    # Check if the image has an alpha channel
    if image.shape[2] == 4:
        # Split the channels
        b, g, r, a = cv2.split(image)
        # Create a mask for transparent pixels (alpha channel is 0)
        transparent_mask = (a == 0)
        # Create an array for the BGR color
        bgr_color_array = np.full((image.shape[0], image.shape[1], 3), bgr_color, dtype=np.uint8)
        # Replace transparent pixels with the BGR color
        image[transparent_mask] = np.concatenate([bgr_color_array[transparent_mask], np.zeros((np.sum(transparent_mask), 1), dtype=np.uint8)], axis=1)
        # Save the output image
        return image[:,:,:3]
    else:
        raise ValueError("The image does not have an alpha channel.")

def annotate_frame(frame, logo, total_reward):
    H,W,_ = logo.shape
    frame[(-15-H):-15,15:(15+W),:] = logo
    cv2.putText(frame, f'Seed: {random_seed}', org=(3, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0,0,255), thickness=2)
    cv2.putText(frame, f'Total Reward: {total_reward:.1f}', org=(3, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255,255,0), thickness=2)
    cv2.putText(frame, f'{death_count}', org=(25+W,frame.shape[0]-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,0),thickness=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="visualize",
        description="Create a video of a trained agent navigating an environment"
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
        '--rad',
        action='store_true',
        help = "Use a cooler icon."
    )
    parser.add_argument(
        '-w', '--write',
        type=str,
        help = "Specify this option to record and write to an output path."
    )
    args = parser.parse_args()

    mode = args.mode
    ckpt = args.ckpt
    rad = args.rad
    write = args.write
    if write:
        filename = write
        fps = 30.0
        res = (288,512)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, res)

    min_seed = 0
    max_seed = 100000
    seed = random.sample(range(min_seed, max_seed), 1)[0]
    initial_episode_seed = random.sample(range(min_seed, max_seed), 1)[0]
    print('Model seed: ', seed)
    eval_env = gym.make(
        "FlappyBird-v0", audio_on=False, render_mode="rgb_array", use_lidar=False
    )
    eval_env = EnvWrapper(eval_env)

    # All models are seeded by default anyways for reproducibility
    if mode=='DQN':
        evalDQN = DQN.DQN(EnvWrapper(eval_env),
                        model.CNN,
                        device = 'cpu', seed=seed)
    elif mode=='HardDQN':
        evalDQN = DQN.HardUpdateDQN(EnvWrapper(eval_env),
                        model.CNN,
                        device = 'cpu', seed=seed)
    else:
        evalDQN = DQN.SoftUpdateDQN(EnvWrapper(eval_env),
                        model.CNN,
                        device = 'cpu', seed=seed)
    evalDQN.model.load_state_dict(torch.load(ckpt, weights_only=True))
    
    print('Chillax bro, Agent Flappy is warming up...')
    _,_ = evalDQN.play_episode(0,True,initial_episode_seed)
    death_count = 0
    death_count_image_path = r'.\imgs\flappy_bird_rad_icon.png' if rad else r'.\imgs\flappy_bird_icon.png'
    bg_rgb = (149,216,222)
    death_count_image = replace_transparent_pixels(death_count_image_path, bg_rgb)
    logo_W, logo_H = death_count_image.shape[1], death_count_image.shape[0]
    new_logo_W, new_logo_H = int(logo_W/6), int(logo_H/6)
    death_count_image = cv2.resize(death_count_image, (new_logo_W, new_logo_H))
    running_total_reward = 0
    while True:
        random_seed = random.sample(range(min_seed, max_seed), 1)[0]
        print('Respawn seed: ', random_seed)
        state,_ = evalDQN.env.reset(seed=random_seed)
        done = False
        total_reward = 0
        with torch.no_grad():
            while not done:
                action = evalDQN._sample_action(state,0)
                next_state, reward, terminated, truncated, _ = evalDQN.env.step(action)
                total_reward += reward
                done = terminated or truncated
                state = next_state
                frame = evalDQN.env.render()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # frame[-55:-30,5:30,:] = death_count_image
                annotate_frame(frame,death_count_image,total_reward)
                cv2.imshow('Frame', frame)
                if write:
                    out.write(frame)
                if cv2.waitKey(30) & 0xFF==27:
                    cv2.destroyAllWindows()
                    if check_internet_connection():
                        print(get_ascii("mission%20passed!"))
                        print(get_ascii("respect%2B"))
                    if death_count!=0: # Avoid division by 0
                        if death_count==1:
                            print(f"Infinite Run Summary:\nYou killed Agent Flappy {death_count} time!\nAverage Reward per Run: {(running_total_reward/death_count):.4f}")
                        else:
                            print(f"Infinite Run Summary:\nYou killed Agent Flappy {death_count} times!\nAverage Reward per Run: {(running_total_reward/death_count):.4f}")
                    else:
                        print(f"Infinite Run Summary:\nYou killed Agent Flappy 0 times!")
                    exit()
            running_total_reward+=total_reward
            death_count+=1
            if death_count==1:
                print(f"Agent Flappy died 1 time!")
            else:
                print(f"Agent Flappy has died {death_count} times!")
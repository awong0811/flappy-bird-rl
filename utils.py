import cv2
import matplotlib.pyplot as plt
import matplotlib.animation
import os
import numpy as np
import config

def preprocess(img):
    img = img[0:400,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    dim = config.ENVWRAPPER['resolution']
    img = cv2.resize(img, (dim,dim))
    return img

def animate(frames):
    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')
    im = plt.imshow(frames[0])
    def animate(i):
        im.set_array(frames[i])
        return im,
    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=len(frames))
    return anim

def plotgraphs(train_reward_history, train_loss_history, val_reward_history, val_std_history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(train_reward_history)), train_reward_history)
    plt.title('Training Rewards Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Training Rewards')
    plt.subplot(1,2,2)
    plt.plot(np.arange(len(train_loss_history)), train_loss_history)
    plt.title('Training Loss Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Training Loss')

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(len(val_reward_history)), val_reward_history)
    plt.title('Validation Rewards Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Validation Rewards')
    plt.subplot(1,2,2)
    plt.plot(np.arange(len(val_std_history)), val_std_history)
    plt.title('Validation STD Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Validation STD')

# Training Functions
# class exponential_decay:
#     def __init__(self, epsilon:float, half_life:int, min_epsilon:float):
#         self.epsilon = epsilon
#         self.decay_rate = 0.5 ** (1 / half_life)
#         self.epsilon = self.epsilon/self.decay_rate
#         self.min_epsilon = min_epsilon
        
#     def __call__(self):
#         self.epsilon = max(self.epsilon * self.decay_rate, self.min_epsilon)
#         return self.epsilon
class exponential_decay:
    def __init__(self, initial_epsilon:float, num_updates:int, final_epsilon:float):
        self.epsilon = initial_epsilon
        self.decay_rate = (final_epsilon/initial_epsilon)**(1/num_updates)
        self.epsilon = self.epsilon/self.decay_rate
        self.final_epsilon = final_epsilon

    def __call__(self):
        self.epsilon = max(self.epsilon * self.decay_rate, self.final_epsilon)
        return self.epsilon

class linear_decay:
    def __init__(self, epsilon:float, decay_time:int, min_epsilon:float):
        self.epsilon = epsilon
        self.decay_rate = (epsilon - min_epsilon) / decay_time
        self.epsilon = self.epsilon + self.decay_rate
        self.min_epsilon = min_epsilon
        
    def __call__(self):
        self.epsilon = max(self.epsilon - self.decay_rate, self.min_epsilon)
        return self.epsilon    

def get_save_path(suffix,directory):
    save_path = os.path.join(directory,suffix)
    #find the number of run directories in the directory
    try:
        runs = [d for d in os.listdir(save_path) if "run" in d]
        runs = sorted(runs,key = lambda x: int(x.split("run")[1]))
        last_run = runs[-1]
        last_run = int(last_run.split("run")[1])
        save_path = os.path.join(save_path,f"run{last_run+1}")
    except:
        save_path = os.path.join(save_path,"run0")
    print("saving to",save_path)
    return save_path
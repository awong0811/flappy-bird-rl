import cv2
import matplotlib.pyplot as plt
import matplotlib.animation

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
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

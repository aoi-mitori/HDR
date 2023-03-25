import cv2
import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker


def read_files(dir_name):
    images = []
    for filename in np.sort(os.listdir(dir_name)):
        if os.path.splitext(filename)[1] in ['.png', '.jpg', '.JPG']: # Only read png or jpg files
            img = cv2.imread(os.path.join(dir_name, filename))
            images.append(img)
    return images


def load_exp_time(dir_name, file_name):
    Speed = []
    f = open(os.path.join(dir_name, file_name))
    
    Speed += [float(i) for i in f]
    print("Speed Time: {0}".format(Speed))
    exp_times = [1 / Speed[i] for i in range(len(Speed))]
    print("Exposure Time: {0}".format(exp_times))
    return exp_times



# Reference : https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
#             https://stackoverflow.com/questions/25983218/scientific-notation-colorbar-in-matplotlib
def fmt(x, pos):
    return "{:.2f}".format(np.exp(x))  

def plot_radiance(hdr, dir):
    radiance = 0.2126 * hdr[:, :, 0] + 0.7152 * hdr[:, :, 1]  + 0.0722 * hdr[:, :, 2]
    log_radiance = np.log(radiance)
    # print(np.max(radiance), np.min(radiance))
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(log_radiance, cmap='jet')
    fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fmt))
    plt.savefig(dir + '/radiance_map.png')


def plot_response_curve(g, dir):
    color = ['blue', 'green', 'red']
    fig, ax = plt.subplots(2,2,figsize=(10,10))
    for x in range(2):
        for y in range(2):
            if x+y != 2:
                index = x*2+y
                ax[x][y].scatter(x=g[index], y=np.arange(256), c=color[index], s=1)
                ax[x][y].set_title(color[index])
            else:
                for channel in range(3):
                    ax[x][y].scatter(x=g[channel], y=np.arange(256), c=color[channel], s=1)
                ax[x][y].set_title('all')
            ax[x][y].set_xlabel('log exposure X')
            ax[x][y].set_ylabel('pixel value Z')
    fig.tight_layout(pad=2.0)
    fig.savefig(dir + '/response_curves.png')


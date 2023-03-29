import cv2
import os
import numpy as np
from tqdm import tqdm
from options import args
from utils import read_files, load_exp_time, plot_radiance, plot_response_curve
from mtb import mtb
from tone_mapping import global_tone_mapping, local_tone_mapping


# Paul E. Debevec's method
# Reference: https://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf
# Solving response curve
def gsolve(Z, B, l, w):

    # Matlab start from 1 while Python start from 0
    n = 256
    A = np.zeros(shape = (np.size(Z, 0) * np.size(Z, 1) + n + 1, n + np.size(Z, 0)))
    b = np.zeros(shape = (np.size(A, 0), 1))

    # Include the data−fitting equations
    k = 0
    for i in range(np.size(Z, 0)):
        for j in range(np.size(Z, 1)):
            wij = w[Z[i, j]]
            A[k, Z[i, j]] = wij 
            A[k, n + i] = (-1) * wij
            b[k, 0] = wij * B[j]
            k = k + 1
    
    # Fix the curve by setting its middle value to 0
    A[k, 127] = 1
    k = k + 1

    # Include the smoothness equations
    for i in range(n - 1):
        A[k, i - 1]= l * w[i] 
        A[k, i] = (-2) * l * w[i] 
        A[k, i + 1] = l * w[i] 
        k = k + 1

    # Solve the system using SVD
    x = np.linalg.lstsq(A, b, rcond = None)[0] # solve the answer of x from Ax = b 
    g = x[:n].reshape(-1) # the log exposure corresponding to pixel value z
    lE = x[n:].reshape(-1) # the log film irradiance at pixel location i

    return g, lE


def response_curve(images, exp_times):
    
    # Z: the pixel values of pixel location number i in image j
    smallRow = 10
    smallCol = 10
    
    # Resize the picture into smallRow x smallCol
    Z = [cv2.resize(i, (smallRow, smallCol)) for i in images]
    Z = np.reshape(Z, (len(Z), -1, 3)) # (#images, w * h, channel)
    Z = np.transpose(Z, (1, 0, 2)) # (w * h, #images, channel)

    # B: the log delta t, or log shutter speed, for image j
    B = np.log(exp_times)
    
    # l: the constant that determines the amount of smoothness
    l = 30   

    # w: the weighting function value for pixel value z
    w = [i if i <= 0.5 * 256 else 256 - i for i in range(256)]

    g = np.zeros((3, 256))
    lE = np.zeros((3, smallRow * smallCol))

    # R, G and B channels
    for channel in range(3):
        g[channel], lE[channel] = gsolve(Z[:, :, channel], B, l, w)


    # Recover Radiance
    print("\nRecover Radiance of RGB Channels")
    height, width, ch = images[0].shape
    lnE = np.zeros((height, width, ch))
    for channel in range(ch):
        for i in tqdm(range(height)):
            for j in range(width):
                weightSum = 0
                for image in range(len(images)):
                    z = images[image][i, j, channel]
                    weightSum += w[z]
                    lnE[i, j, channel] += w[z] * (g[channel][z] - B[image])
                if weightSum != 0:
                    lnE[i, j, channel] /= weightSum
    E = np.exp(lnE)

    return E, g

def hdr():
    src_dir = args.src_dir
    out_dir = args.out_dir
    shutter_speed_dir = args.exposure_file

    isExist = os.path.exists(out_dir)
    if not isExist:
        os.makedirs(out_dir)

    # read files
    images = read_files(src_dir)
    
    # alignment
    if args.mtb:
        images = mtb(images)
    
    # load exposures
    exp_times = load_exp_time(src_dir, shutter_speed_dir)
    
    # HDR
    E, g = response_curve(images, np.array(exp_times, dtype = np.float32))
    cv2.imwrite(out_dir + '/' + src_dir + ".hdr", E * 255)
    
    # global tone mapping
    print("Global Tone Mapping")
    global_ldr, _ = global_tone_mapping(E, a = args.a, l_white = args.lw)
    cv2.imwrite(out_dir + '/' + src_dir + "_global_tone.png", global_ldr)
    
    # local tone mapping
    print("Local Tone Mapping")
    local_ldr = local_tone_mapping(E, a = args.a, l_white = args.lw)
    cv2.imwrite(out_dir + '/' + src_dir + "_local_tone.png", local_ldr)

    # Plot Response Curve
    if args.plot_curve:
        plot_response_curve(g, out_dir)

    # Plot Radiance
    if args.plot_radiance:
        plot_radiance(E, out_dir)



hdr()

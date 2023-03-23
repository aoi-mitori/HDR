import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from mtb import read_files, mtb
from gtm import global_tone_mapping
from ltm import local_tone_mapping

# Read images
dir_name = "exposures"
# dir_name = "exposures"

#images = []
# images_rgb = []

# for filename in np.sort(os.listdir(dir_name)):
#     if os.path.splitext(filename)[1] in ['.png', '.jpg']: # Only read png or jpg files
#         img = cv2.imread(os.path.join(dir_name, filename))
#         images.append(img)
#         images_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# read files
images = read_files(dir_name)

# alignment
images = mtb(images)

# fig, ax = plt.subplots(4, 4, figsize=(15, 15))
# for p in range(len(images)):
#     row = int(p / 4)
#     col = int(p % 4)
#     ax[row, col].imshow(images_rgb[p])
# plt.show()
    
# Exposure time
# speed = np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
# exp_times = [1 / speed[i] for i in range(len(speed))]
# print(speed)
# print(exp_times)

def load_exp_time(dir_name):
    Speed = []
    f = open(os.path.join(dir_name + "/" + dir_name + '_image_list.txt'))
    for i in f:
        if (i[0] == '#'):
            continue
        (filename, speed, *rest) = i.split()
        Speed += [float(speed)]
    exp_times = [1 / Speed[i] for i in range(len(Speed))]
    return exp_times

# Align input images
# alignMTB = cv2.createAlignMTB()
# alignMTB.process(images, images)

# # Camera response function (CRF)
# calibrateDebevec = cv2.createCalibrateDebevec()
# responseDebevec = calibrateDebevec.process(images, np.array(exp_times, dtype = np.float32))

# # Merge images into HDR linear images
# mergeDebevec = cv2.createMergeDebevec()
# hdrDebevec = mergeDebevec.process(images, np.array(exp_times, dtype = np.float32), responseDebevec)
# cv2.imwrite("test_cv2_lib.hdr", hdrDebevec)

# # Tone mapping
# tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
# ldrDrago = tonemapDrago.process(hdrDebevec)
# ldrDrago = 3 * ldrDrago
# cv2.imwrite("test_cv2_lib_tone.png", ldrDrago * 255)

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
        A[k, i]= l * w[i + 1] 
        A[k, i + 1] = (-2) * l * w[i + 1] 
        A[k, i + 2] = l * w[i + 1] 
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
    # print(1, np.asarray(Z).shape)
    Z = np.reshape(Z, (len(Z), -1, 3)) 
    # print(2, Z.shape) # (#images, w * h, channel)
    Z = np.transpose(Z, (1, 0, 2)) 
    # print(3, Z.shape) # (w * h, #images, channel)

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
    height, width, ch = images[0].shape
    lnE = np.zeros((height, width, 3))
    for channel in range(3):
        for i in range(height):
            for j in range(width):
                weightSum = 0
                for image in range(len(images)):
                    z = images[image][i, j, channel]
                    weightSum += w[z]
                    lnE[i, j, channel] += w[z] * (g[channel][z] - B[image])
                if weightSum != 0:
                    lnE[i, j, channel] /= weightSum
    E = np.exp(lnE)
    return E


exp_times = load_exp_time(dir_name)
E = response_curve(images, np.array(exp_times, dtype = np.float32))
cv2.imwrite("hdr.hdr", E * 255)

GL_LDR, _ = global_tone_mapping(E, 0.18, 0.9)
cv2.imwrite("g_ldr_white.png", GL_LDR)

local_ldr = local_tone_mapping(E, a = 0.5, l_white = 0.9)
cv2.imwrite("l_ldr_white.png", local_ldr)








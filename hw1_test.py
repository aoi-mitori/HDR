import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from copy import deepcopy
import math


# Read images
dir_name = "Memorial_SourceImages"

images = []
images_rgb = []

for filename in np.sort(os.listdir(dir_name)): # 讀出這個路徑下的檔案
    if os.path.splitext(filename)[1] in ['.png', '.jpg']: # 只選png或jpg的檔案
	    img = cv2.imread(os.path.join(dir_name, filename))
	    images.append(img)
	    images_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
# Number of images
# P = len(images)
# print('P =', P)

# height, width, channel = images[0].shape
# print('image shape:', images[0].shape)


# Exposure time
exp_times = []
speed = np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
for i in range (len(speed)):
	exp_times.append(1 / speed[i]) # let speed be changed as time
	# print(exp_times[i])


# Align input images
alignMTB = cv2.createAlignMTB()
alignMTB.process(images, images)

# Camera response function (CRF)
# calibrateDebevec = cv2.createCalibrateDebevec()
# responseDebevec = calibrateDebevec.process(images, exp_times)

# Merge images into HDR linear images
# mergeDebevec = cv2.createMergeDebevec()
# hdrDebevec = mergeDebevec.process(images, exp_times, responseDebevec)

# Save image
# cv2.imwrite("test_cv2_lib.hdr", hdrDebevec)



# Paul E. Debevec's method
# Reference: https://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf
# Solving response curve
def gsolve(Z, B, l, w):

	# Matlab start from 1 while Python start  from 0
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
	g = x[:n].reshape(-1)
	lE = x[n:].reshape(-1)

	return g, lE

def recovered_responseCurve(images, exposureTimes):
    smallImages = deepcopy(images)
    smallRow = 10
    smallCol = 10
    
    for i in range(0, len(images)):
        smallImages[i] = cv2.resize(images[i], (smallRow, smallCol))

    smallImages = np.array(smallImages)
    smallImages = np.reshape(smallImages, (len(smallImages),-1,3))  # (nImage, w*h, channel)
    smallImages = np.transpose(smallImages, (1, 0, 2))              # (w*h, nImage, channel)
    
    # w
    weight = np.zeros(256)
    for i in range(256):
        weight[i] = min(i, 256-i)
        
    logTimes = np.log(exposureTimes)
    g = np.zeros((3, 256))
    lE = np.zeros((3, smallRow*smallCol))

    for channel in range(3):
        g[channel], lE[channel] = gsolve(smallImages[:,:,channel], logTimes, 30, weight)
    return g


def recovered_radiance(images, responseCurve, exposureTime):
    logExposureTime = np.log(exposureTime)

    weight = np.zeros(256)
    for i in range(256):
        weight[i] = min(i,256-i)

    h = images[0].shape[0]
    w = images[0].shape[1]

    lE = np.zeros((h,w,3))
    for channel in range(3): 
        for i in range(h):
            for j in range(w):
                weightSum = 0
                for image in range(len(images)):
                    z = images[image][i,j,channel]
                    weightSum += weight[z]
                    lE[i,j,channel] += weight[z]*(responseCurve[channel][z]-logExposureTime[image])
                if weightSum != 0:
                    lE[i,j,channel] /= weightSum
    E = np.exp(lE)
    return E

g = recovered_responseCurve(images, exp_times)
E = recovered_radiance(images, g, exp_times)
cv2.imwrite("test.hdr", E * 255)








import cv2
import numpy as np
import math
from tqdm import tqdm

# Global tone mapping
# Big a: bright
# Small a: dark
def global_tone_mapping(radiance, a = 0.5, l_white = 0.9):
    e = 1e-6  # avoid log 0
    lw = radiance

    lw_bar = np.exp(np.mean(np.log(e + 0.2126 * lw[:, :, 0] 
        + 0.7152 * lw[:, :, 1] 
        + 0.0722 * lw[:, :, 2])))
    lm = a * lw / lw_bar
    l_white *= np.max(lm)
    ld = lm * (1 + lm / math.pow(l_white, 2)) / (1 + lm)
    reconstructedLDR = np.clip(np.rint(ld * 255), 0, 255).astype(np.uint8)

    return reconstructedLDR, lm



# Local Tone Mapping
def gaussian_blur(l_m, s):
    return cv2.GaussianBlur(l_m, (s, s), 0, 0)

def local_tone_mapping(hdr, a = 0.5, l_white = 0.9, phi = 8, epsilon = 0.05):
    # a: key value
    # phi: sharpening parameter
    # epsilon: threshold
    _, l_m= global_tone_mapping(hdr, a, l_white)

    l_blurs = []
    for s in range(1, 34, 2):
        l_blurs += [gaussian_blur(l_m, s)]
    
    s_max_index = 0
    for i, l_blur in tqdm(enumerate(l_blurs[:-1])):
        v = np.abs( (l_blur - l_blurs[i + 1]) / (2 ** phi * a / s ** 2 + l_blur) )
        if np.all(v < epsilon):
            s_max_index = i
    
    l_d = l_m / (1+l_blurs[s_max_index])
    ldr = np.clip(np.rint(l_d * 255), 0, 255).astype(np.uint8)

    return ldr
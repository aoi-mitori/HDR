# Local Tone Mapping
import cv2
import numpy as np
from gtm import global_tone_mapping

def gaussian_blur(l_m, s):
    return cv2.GaussianBlur(l_m, (s,s), 0, 0)

def local_tone_mapping(hdr, a=0.5, l_white=0.9, phi=8, epsilon=0.05):
    # a: key value
    # phi: sharpening parameter
    # epsilon: threshold
    _, l_m= global_tone_mapping(hdr, a, l_white)

    l_blurs = []
    for s in range(1,34,2):
        l_blurs += [gaussian_blur(l_m, s)]
    
    s_max_index = 0
    for i, l_blur in enumerate(l_blurs[:-1]):
        v = np.abs( (l_blur - l_blurs[i+1]) / (2**phi*a/s**2 + l_blur) )
        if np.all(v < epsilon):
            s_max_index = i
    
    l_d = l_m / (1+l_blurs[s_max_index])
    ldr = np.clip(np.rint(l_d * 255), 0, 255).astype(np.uint8)

    return ldr

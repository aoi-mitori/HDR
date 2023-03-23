import numpy as np
import math

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


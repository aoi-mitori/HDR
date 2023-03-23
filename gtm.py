import numpy as np

# Global tone mapping
# Big a: bright
# Small a: dark
def GlobalToneMapping(radiance, a = 0.5, l_white = 0.9):
    e = 1e-15  # avoid log 0
    lw = radiance

    lw_bar = np.exp(0.2126 * np.mean(np.log(e + lw[:, :, 0])) 
        + 0.7152 * np.mean(np.log(e + lw[:, :, 1])) 
        + 0.0722 * np.mean(np.log(e + lw[:, :, 2])))
    
    lm = a * lw / lw_bar
    ld = lm / (1 + lm)
    # ld = lm * (1 + lm / math.pow(l_white, 2)) / (1 + lm)

    reconstructedLDR = np.clip(np.rint(ld * 255), 0, 255).astype(np.uint8)

    return reconstructedLDR, lm


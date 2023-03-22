# Local Tone Mapping
from gtm import GlobalToneMapping

def local_tone_mapping(hdr, a=0.5, l_white=0.9, phi=8, epsilon=0.05):
    # a: key value
    # phi: sharpening parameter
    # epsilon: threshold
    _, l_m= GlobalToneMapping(hdr, a, l_white)

    pr
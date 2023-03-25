import argparse

parser = argparse.ArgumentParser(description = 'HDR Imaging')

parser.add_argument('--src_dir', type = str, default = 'memorial', help = 'The directory where the input images are stored.')
parser.add_argument('--out_dir', type = str, default = 'outputs', help = 'The directory where the outputs will be stored.')
parser.add_argument('--exposure_file', type = str, default = 'shutter_speed.txt', help = 'The file name of shutter speeds file.')

parser.add_argument('-a', default=0.18, type=float, help="Scene's key, determine how light or how dark it is.")
parser.add_argument('--lw', default=0.9, type=float, help="L_white, the smallest luminance to be mapped to 1.")

parser.add_argument('--plot_curve', default = True, help = 'Plot response curve or not.')
parser.add_argument('--plot_radiance', default = True, help = 'Plot radiance map or not.')

parser.add_argument('--mtb', action='store_true', help = 'Do mtb algorithm or not.')
parser.add_argument('--no-mtb', dest='mtb', action='store_false', help = 'Do mtb algorithm or not.')
parser.set_defaults(mtb=True)

args = parser.parse_args()
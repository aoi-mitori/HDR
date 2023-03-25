import argparse

parser = argparse.ArgumentParser(description = 'HDR Imaging')

parser.add_argument('--src_dir', type = str, default = 'memorial', help = 'the directory where the input images are stored.')
parser.add_argument('--out_dir', type = str, default = 'outputs', help = 'the directory where the outputs will be stored.')
parser.add_argument('--exposure_file', type = str, default = 'shutter_speed.txt', help = 'the file name of shutter speeds file.')

parser.add_argument('-a', default=0.18, type=float, help="scene's key, determine how light or how dark it is.")
parser.add_argument('--lw', default=0.9, type=float, help="l_white, the smallest luminance to be mapped to 1.")

parser.add_argument('--plot_curve', action='store_true', help = 'plot response curve.')
parser.add_argument('--no-plot_curve', dest='plot_curve', action='store_false', help = "don't plot response curve.")
parser.set_defaults(plot_curve=True)

parser.add_argument('--plot_radiance', action='store_true', help = 'plot radiance map.')
parser.add_argument('--no-plot_radiance', dest='plot_radiance', action='store_false', help = "don't plot radiance map.")
parser.set_defaults(plot_radiance=True)

parser.add_argument('--mtb', action='store_true', help = 'do mtb algorithm.')
parser.add_argument('--no-mtb', dest='mtb', action='store_false', help = "don't do mtb algorithm.")
parser.set_defaults(mtb=True)

args = parser.parse_args()
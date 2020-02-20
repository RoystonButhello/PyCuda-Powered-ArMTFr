'''Configure algorithm operation via this file'''
import os
from enum import Enum

# Path to set working directory
PATH = os.path.dirname(os.path.abspath( __file__ )) + "\\"

# Input image name and extension
IMG = "mountain1080"
EXT = ".png"

# Key paths
TEMP = "temp\\"             # Folder used to store intermediary results
SRC  = "images\\"           # Folder containing input and output
FRAC = "fractals\\"         # Folder containing fractal images

# Input/Output images
ENC_IN =  SRC + IMG + EXT               # Input image for encryption
ENC_OUT = SRC + IMG + "_encrypted.png"  # Final Encrypted Image
DEC_OUT = SRC + IMG + "_decrypted.png"  # Final Decrypted Image

# Log Files
LOG     = TEMP + "log.txt"          # Store Image Dimensions, Image Hash, ArMap Iterations
P1LOG   = TEMP + "p1log.txt"        # Store parameters for column-rotation vector
P2LOG   = TEMP + "p2log.txt"        # Store parameters for row-rotation vector

#Flags
DO_HISTEQ = False
DEBUG_TIMER  = True     # Print timing statistics in console

#Constants
PERMINTLIM = 32     # Used by genRelocVec()
PERM_ROUNDS = 8     # No. of rounds to run permutation kernel
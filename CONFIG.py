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

# Intermediary Images
LOG     = TEMP + "log.txt"          # Store Image Dimensions, Image Hash, ArMap Iterations
P1LOG   = TEMP + "p1log.txt"        # Store parameters for column-rotation vector
P2LOG   = TEMP + "p2log.txt"        # Store parameters for row-rotation vector
ARMAP   = TEMP + "1armap.png"       # Arnold-mapped Image
XOR     = TEMP + "2xor.png"         # XOR'd Image
PERM    = TEMP + "3permute.png"     # Permuted Image
UNPERM  = TEMP + "4unpermute.png"   # Un-Permuted Image
UNXOR   = TEMP + "5unxor.png"       # Un-XOR'd Image

#Flags
DO_HISTEQ = False
DEBUG_IMAGES = False    # View original and equalized image
DEBUG_TIMER  = True     # Print timing statistics in console

#Constants
PERMINTLIM = 32     # Used by genRelocVec()
PERM_ROUNDS = 7     # No. of rounds to run permutation kernel
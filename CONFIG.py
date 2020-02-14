'''Configure algorithm operation via this file'''
import os

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
LOG     = TEMP + "log.txt"           # Store Image Dimensions, Image Hash, ArMap Iterations
P1LOG   = TEMP + "p1log.txt"         # Store parameters for column-rotation vector
P2LOG   = TEMP + "p2log.txt"         # Store parameters for row-rotation vector
HISTEQ  = TEMP + "2histeq.png"       # Histogram-equalized square Image
ARMAP   = TEMP + "3armap.png"        # Arnold-mapped Image
XOR     = TEMP + "4xorfractal.png"   # Fractal-XOR'd Image
MT      = TEMP + "5mtshuffle.png"    # MT-Shuffled Image
UnMT    = TEMP + "6mtunshuffle.png"  # MT-UnShuffled Image
UnXOR   = TEMP + "7xorunfractal.png" # Fractal-UnXOR'd Image

#Flags
DO_HISTEQ    = False    # Perform histogram equalization
DEBUG_IMAGES = True     # View original and equalized image
DEBUG_TIMER  = True     # Print timing statistics in console

#Constants
MASK_BITS = 16      # Used by Serial MTShuffle() and MTUnShuffle()
BUFF_SIZE = 65536   # Used by CoreFunctions.sha2HashFile()
PERMINTLIM = 32     # Used by genRelocVec()
PERM_ROUNDS = 7     # No. of rounds to run permutation kernel
import cv2                  # OpenCV
import os                   # Path setting and file-retrieval
import glob                 # File counting
import random               # Obviously neccessary
import numpy as np          # See above
import CONFIG as cfg        # Module with Debug flags and other constants
import hashlib              # For SHA256

#PyCUDA Import
import pycuda.autoinit
from pycuda.compiler import SourceModule

os.chdir(cfg.PATH)

# Return SHA256 Hash of file as integer
def sha2HashFile(filename):
    hashobj = hashlib.sha256()
    with open(filename,'rb') as f:
        while True:
            block = f.read(cfg.BUFF_SIZE)
            if not block:
                break
            hashobj.update(block)
    return int(hashobj.hexdigest(),16)

# Returns SHA256 Hash of flattened OpenCV Image as integer
def sha2HashImage(img, N=256):
    cv2.resize(img,(N,N))
    imgflat = img.flatten()
    hashobj = hashlib.sha256()
    hashobj.update(imgflat)
    return int(hashobj.hexdigest(),16)

# Returns the estimated ArMap cycle length
def ArMapLen(n):
    x, y = random.randint(0,n), random.randint(0,n)
    x1, y1 = (2*x+y)%n, (x+y)%n
    iterations = 1
    while x!=x1 and y!=y1:
        x1, y1 = (2*x1+y1)%n, (x1+y1)%n
        iterations += 1
    return iterations

# Arnold's Cat Map
def ArCatMap(img_in):
    dim = img_in.shape
    N = dim[0]
    img_out = np.zeros([N, N, dim[2]])

    for x in range(N):
        for y in range(N):
            img_out[x][y] = img_in[(x+y)%N][(2*x+y)%N]

    return img_out

# Mersenne-Twister Intra-Column-Shuffle
def MTShuffle(img_in, imghash):
    mask = 2**cfg.MASK_BITS - 1   # Default: 16 bits
    temphash = imghash
    dim = img_in.shape
    N = dim[0]
    img_out = img_in.copy()

    for j in range(N):
        random.seed(temphash & mask)
        MTmap = list(range(N))
        random.shuffle(MTmap)
        temphash = temphash>>cfg.MASK_BITS
        if temphash==0:
            temphash = imghash
        for i in range(N):
            index = int(MTmap[i])
            img_out[i][j] = img_in[index][j]
    return img_out

# Mersenne-Twister Intra-Column-Shuffle
def MTUnShuffle(img_in, imghash):
    mask = 2**cfg.MASK_BITS - 1   # Default: 8 bits
    temphash = imghash
    dim = img_in.shape
    N = dim[0]
    img_out = img_in.copy()

    for j in range(N):
        random.seed(temphash & mask)
        MTmap = list(range(N))
        random.shuffle(MTmap)
        temphash = temphash>>cfg.MASK_BITS
        if temphash==0:
            temphash = imghash
        for i in range(N):
            index = int(MTmap[i])
            img_out[index][j] = img_in[i][j]
    return img_out

# XOR Image with a Fractal
def FracXor(img_in, imghash):

    #Select a file for use based on hash
    fileCount = len(glob.glob1("fractals","*.png"))
    fracID = (imghash % fileCount) + 1
    filename = cfg.FRAC + str(fracID) + ".png"
    #Read the file, resize it, then XOR
    fractal = cv2.imread(filename, 1)
    dim = img_in.shape
    fractal = cv2.resize(fractal,(dim[0],dim[1]))
    img_out = cv2.bitwise_xor(img_in,fractal)

    return img_out

# Create folder for intermediary files or clear it if it exists
def TempClear():
    files = os.listdir(cfg.TEMP)
    for f in files:
        os.remove(os.path.join(cfg.TEMP, f))


# Clear ArMap debug files
def ArMapClear():
    files = os.listdir(cfg.ARTEMP)
    for f in files:
        os.remove(os.path.join(cfg.ARTEMP, f))
        
mod = SourceModule("""
    #include <stdint.h>
    __global__ void ArCatMap(uint8_t *in, uint8_t *out)
    {
        int nx = (blockIdx.x + blockIdx.y) % gridDim.x;
        int ny = (2*blockIdx.x + blockIdx.y) % gridDim.y;
        int blocksize = blockDim.x * blockDim.y * blockDim.z;
        int InDex = ((gridDim.x)*blockIdx.y + blockIdx.x) * blocksize  + threadIdx.x;
        int OutDex = ((gridDim.x)*ny + nx) * blocksize + threadIdx.x;
        out[OutDex] = in[InDex];
    }
  """)

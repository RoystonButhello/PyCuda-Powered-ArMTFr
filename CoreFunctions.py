import cv2              # OpenCV
import hashlib          # For SHA256
import secrets          # For genRelocVec()
import numpy as np      # Naturally needed
import CONFIG as cfg    # Module with Debug flags and other constants
import os               # Path setting & File Counting
from random import randint

#PyCUDA Import
import pycuda.autoinit
from pycuda.compiler import SourceModule

os.chdir(cfg.PATH)

# Returns the estimated ArMap cycle length
def ArMapLen(n):
    x, y = randint(0,n), randint(0,n)
    x1, y1 = (2*x+y)%n, (x+y)%n
    iterations = 1
    while x!=x1 and y!=y1:
        x1, y1 = (2*x1+y1)%n, (x1+y1)%n
        iterations += 1
    return iterations

# Generate and return rotation vector of length n containing values < m
def genRelocVec(m, n, logfile, ENC=True):
    # Initialize constants
    if ENC:
        secGen = secrets.SystemRandom()
        a = secGen.randint(2,cfg.PERMINTLIM)
        b = secGen.randint(2,cfg.PERMINTLIM)
        c = 1 + a*b
        x = secGen.uniform(0.0001,1.0)
        y = secGen.uniform(0.0001,1.0)
        offset = secGen.randint(1,cfg.PERMINTLIM)
        # Log parameters for decryption
        with open(logfile, 'a+') as f:
            f.write(str(a) +"\n")
            f.write(str(b) +"\n")
            f.write(str(x) +"\n")
            f.write(str(y) +"\n")
            f.write(str(offset) + "\n")
    else:
        with open(logfile, "r") as f:
            fl = f.readlines()
            a = int(fl[0])
            b = int(fl[1])
            c = 1 + a*b
            x = float(fl[2])
            y = float(fl[3])
            offset = int(fl[4])
    unzero = 0.0000001

    # Skip first <offset> values
    for i in range(offset):
        x = (x + a*y)%1 + unzero
        y = (b*x + c*y)%1 + unzero
    
    # Start writing intermediate values
    ranF = np.zeros((n),dtype=np.float)
    for i in range(n//2):
        x = (x + a*y)%1 + unzero
        y = (b*x + c*y)%1 + unzero
        ranF[2*i] = x
        ranF[2*i+1] = y
    
    # Generate relocation vector
    exp = 10**14
    vec = np.zeros((n),dtype=np.uint16)
    for i in range(n):
        vec[i] = np.uint16((ranF[i]*exp)%m)
    return vec

# XOR Image with a Fractal
def FracXor(img, fracID=-1):
    # Read/Write fractal filename based on mode
    if fracID==-1:
        fileCount = len(os.listdir(cfg.FRAC))
        fracID = (randint(0,img.shape[0]) % fileCount) + 1
        with open(cfg.LOG, 'a+') as f:
            f.write(str(fracID)+"\n")

    #Read the file, resize it, then XOR
    filename = cfg.FRAC + str(fracID) + ".png"
    fractal = cv2.imread(filename, 1)
    dim = img.shape
    fractal = cv2.resize(fractal,(dim[1],dim[0]))
    return cv2.bitwise_xor(img,fractal)

mod = SourceModule("""
    #include <stdint.h>
    __global__ void ArMapImg(uint8_t *in, uint8_t *out)
    {
        int nx = (2*blockIdx.x + blockIdx.y) % gridDim.x;
        int ny = (blockIdx.x + blockIdx.y) % gridDim.y;
        int InDex = ((gridDim.x)*blockIdx.y + blockIdx.x) * 3  + threadIdx.x;
        int OutDex = ((gridDim.x)*ny + nx) * 3 + threadIdx.x;
        out[OutDex] = in[InDex];
    }

    __global__ void ArMapTable(uint32_t *in, uint32_t *out)
    {
        int nx = (2*blockIdx.x + blockIdx.y) % gridDim.x;
        int ny = (blockIdx.x + blockIdx.y) % gridDim.y;
        int InDex = ((gridDim.x)*blockIdx.y + blockIdx.x);
        int OutDex = ((gridDim.x)*ny + nx);
        out[OutDex] = in[InDex];
    }

    __global__ void ArMapTabletoImg(uint8_t *in, uint8_t *out, uint32_t *table)
    {
        uint32_t idx = ((gridDim.x)*blockIdx.y + blockIdx.x);
        uint32_t InDex = idx * 3 + threadIdx.x;
        uint32_t OutDex = table[idx] * 3 + threadIdx.x;
        out[OutDex] = in[InDex];
    } 
    
    __global__ void Enc_GenCatMap(uint8_t *in, uint8_t *out, uint16_t *colRotate, uint16_t *rowRotate)
    {
        int colShift = colRotate[blockIdx.y];
        int rowShift = rowRotate[(blockIdx.x + colShift)%gridDim.x];
        int InDex    = ((gridDim.y)*blockIdx.x + blockIdx.y) * 3  + threadIdx.x;
        int OutDex   = ((gridDim.y)*((blockIdx.x + colShift)%gridDim.x) + (blockIdx.y + rowShift)%gridDim.y) * 3  + threadIdx.x;
        out[OutDex]  = in[InDex];
    }

    __global__ void Dec_GenCatMap(uint8_t *in, uint8_t *out, uint16_t *colRotate, uint16_t *rowRotate)
    {
        int colShift = colRotate[blockIdx.y];
        int rowShift = rowRotate[(blockIdx.x + colShift)%gridDim.x];
        int OutDex   = ((gridDim.y)*blockIdx.x + blockIdx.y) * 3  + threadIdx.x;
        int InDex    = ((gridDim.y)*((blockIdx.x + colShift)%gridDim.x) + (blockIdx.y + rowShift)%gridDim.y) * 3  + threadIdx.x;
        out[OutDex]  = in[InDex];
    }
  """)
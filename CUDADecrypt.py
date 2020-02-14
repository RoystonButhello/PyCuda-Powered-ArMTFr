import os                   # Path setting and file-retrieval
import cv2                  # OpenCV
import time                 # Timing Execution
import random               # Obviously neccessary
import numpy as np          # See above
import CONFIG as cfg        # Debug flags and constants
import CoreFunctions as cf  # Common functions

#PyCUDA Import
import pycuda.driver as cuda
import pycuda.autoinit

os.chdir(cfg.PATH)

# Driver function
def Decrypt():
    #Initialize Timer
    timer = np.zeros(4)
    overall_time = time.perf_counter()

    #Open the image
    imgMT = cv2.imread(cfg.ENC_OUT, 1)
    if imgMT is None:
        print("File does not exist!")
        raise SystemExit(0)
    dim = imgMT.shape

    # Read log file
    f = open(cfg.LOG, "r")
    fl = f.readlines()
    f.close()
    width, height = int(fl[0]), int(fl[1])
    srchash = int(fl[2])
    rounds = int(fl[3])
    
    timer[0] = time.perf_counter()
    # Inverse Permutation: Intra-row/column rotation
    U = cf.genRelocVec(dim[0],dim[1],cfg.P1LOG, ENC=False) # Col-rotation | len(U)=n, values from 0->m
    V = cf.genRelocVec(dim[1],dim[0],cfg.P2LOG, ENC=False) # Row-rotation | len(V)=m, values from 0->n
    
    imgArray  = np.asarray(imgMT).reshape(-1)
    gpuimgIn  = cuda.mem_alloc(imgArray.nbytes)
    gpuimgOut = cuda.mem_alloc(imgArray.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgArray)
    gpuU = cuda.mem_alloc(U.nbytes)
    gpuV = cuda.mem_alloc(V.nbytes)
    cuda.memcpy_htod(gpuU, U)
    cuda.memcpy_htod(gpuV, V)
    func = cf.mod.get_function("Dec_GenCatMap")

    func(gpuimgIn, gpuimgOut, gpuU, gpuV, grid=(dim[0],dim[1],1), block=(3,1,1))
    temp = gpuimgIn
    gpuimgIn = gpuimgOut
    gpuimgOut = temp
    for i in range(cfg.PERM_ROUNDS):
        func(gpuimgIn, gpuimgOut, gpuU, gpuV, grid=(dim[0],dim[1],1), block=(3,1,1))
        temp = gpuimgIn
        gpuimgIn = gpuimgOut
        gpuimgOut = temp

    cuda.memcpy_dtoh(imgArray, gpuimgIn)
    imgAr = (np.reshape(imgArray,dim)).astype(np.uint8)
    timer[0] = time.perf_counter() - timer[0]

    cv2.imwrite(cfg.UnMT, imgAr)

    timer[1] = time.perf_counter()
    # Inverse Fractal XOR Phase
    imgFr = cf.FracXor(imgMT, srchash)
    timer[1] = time.perf_counter() - timer[1]
    imgAr = imgFr
    cv2.imwrite(cfg.UnXOR, imgFr)

    # Ar Phase: Cat-map Iterations
    dim = imgAr.shape
    imgAr_In = np.asarray(imgAr).reshape(-1)
    imgShuffle = np.arange(start=0, stop=len(imgAr_In)/3, dtype=np.uint32)
    gpuimgIn = cuda.mem_alloc(imgShuffle.nbytes)
    gpuimgOut = cuda.mem_alloc(imgShuffle.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgShuffle)
    func = cf.mod.get_function("ArMapTable")

    iteration = 1
    func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(1,1,1))
    temp = gpuimgOut
    gpuimgOut = gpuimgIn
    gpuimgIn = temp
    # Recalculate mapping to generate lookup table
    timer[2] = time.perf_counter()
    while iteration<rounds:
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(1,1,1))
        temp = gpuimgOut
        gpuimgOut = gpuimgIn
        gpuimgIn = temp
        iteration+=1
    timer[2] = time.perf_counter() - timer[2]

    # Apply mapping
    gpuShuffle = gpuimgIn
    gpuimgIn = cuda.mem_alloc(imgAr_In.nbytes)
    gpuimgOut = cuda.mem_alloc(imgAr_In.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgAr_In)
    func = cf.mod.get_function("ArMapTabletoImg")
    timer[3] = time.perf_counter()
    func(gpuimgIn, gpuimgOut, gpuShuffle, grid=(dim[0],dim[1],1), block=(3,1,1))
    timer[3] = time.perf_counter() - timer[3]
    cuda.memcpy_dtoh(imgAr_In, gpuimgOut)

    imgAr = (np.reshape(imgAr_In,dim)).astype(np.uint8)

    # Resize image to OG dimensions if needed
    if height!=width:
        imgAr = cv2.resize(imgAr,(height,width),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(cfg.DEC_OUT, imgAr)

    overall_time = time.perf_counter() - overall_time
    misc = overall_time - np.sum(timer)

    # Print timing statistics
    if cfg.DEBUG_TIMER:
        print("Target: {} ({}x{})".format(cfg.ENC_IN, dim[1], dim[0]))
        print("PRNG Shuffle:\t{0:9.7f}s ({1:5.2f}%)".format(timer[0], timer[0]/overall_time*100))
        print("Fractal XOR:\t{0:9.7f}s ({1:5.2f}%)".format(timer[1], timer[1]/overall_time*100))
        print("LUT Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(timer[2], timer[2]/overall_time*100))
        print("Mapping Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(timer[3], timer[3]/overall_time*100))
        print("Misc. ops: \t{0:9.7f}s ({1:5.2f}%)".format(misc, misc/overall_time*100))
        print("Net Time:\t{0:7.5f}s\n".format(overall_time))

Decrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()
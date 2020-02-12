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

    # Read log file
    f = open(cfg.LOG, "r")
    fl = f.readlines()
    f.close()
    width, height = int(fl[0]), int(fl[1])
    srchash = int(fl[2])
    rounds = int(fl[3])

    timer[0] = time.perf_counter()
    # Inverse MT Phase: Intra-column pixel unshuffle
    imgMT = cf.MTUnShuffle(imgMT, srchash)
    timer[0] = time.perf_counter() - timer[0]
    cv2.imwrite(cfg.UnMT, imgMT)

    timer[1] = time.perf_counter()
    # Inverse Fractal XOR Phase
    imgFr = cf.FracXor(imgMT, srchash)
    timer[1] = time.perf_counter() - timer[1]
    imgAr = imgFr
    cv2.imwrite(cfg.UnXOR, imgFr)
    
    timer[2] = time.perf_counter()
    # Ar Phase: Cat-map Iterations
    cv2.imwrite(cfg.DEC_OUT, imgAr)
    dim = imgAr.shape
    imgAr_In = np.asarray(imgAr).reshape(-1)
    imgShuffle = np.arange(start=0, stop=len(imgAr_In)/3, dtype=np.uint32)
    gpuimgIn = cuda.mem_alloc(imgShuffle.nbytes)
    gpuimgOut = cuda.mem_alloc(imgShuffle.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgShuffle)
    func = cf.mod.get_function("ArMapTable")

    iteration = 0
    # Recalculate mapping to generate lookup table
    while (iteration<rounds):
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(1,1,1))
        temp = gpuimgOut
        gpuimgOut = gpuimgIn
        gpuimgIn = temp
        iteration+=1

    # Apply mapping
    gpuShuffle = gpuimgIn
    gpuimgIn = cuda.mem_alloc(imgAr_In.nbytes)
    gpuimgOut = cuda.mem_alloc(imgAr_In.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgAr_In)
    func = cf.mod.get_function("ArMapTabletoImg")
    func(gpuimgIn, gpuimgOut, gpuShuffle, grid=(dim[0],dim[1],1), block=(3,1,1))
    cuda.memcpy_dtoh(imgAr_In, gpuimgOut)

    timer[2] = time.perf_counter() - timer[2]
    imgAr = (np.reshape(imgAr_In,dim)).astype(np.uint8)

    # Resize image to OG dimensions if needed
    if height!=width:
        imgAr = cv2.resize(imgAr,(height,width),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(cfg.DEC_OUT, imgAr)

    overall_time = time.perf_counter() - overall_time

    # Print timing statistics
    print("MT Unshuffle completed in " + str(timer[0]) +"s")
    print("Fractal XOR Inversion completed in " + str(timer[1]) +"s")
    print("Arnold UnMapping completed in " + str(timer[2]) +"s")
    print("Decryption took " + str(np.sum(timer)) + "s out of " + str(overall_time) + "s of net execution time")

Decrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()
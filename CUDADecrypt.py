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

    # Read hash from sent file
    f = open(cfg.HASH, "r")
    srchash = int(f.read())
    f.close()

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

    #Clear ArMap debug files
    if cfg.DEBUG_ARMAP:
        cf.ArMapClear()
    
    timer[2] = time.perf_counter()
    # Ar Phase: Cat-map Iterations
    cv2.imwrite(cfg.DEC_OUT, imgAr)
    dim = imgAr.shape
    imgAr_In = np.asarray(imgAr).reshape(-1)
    gpuimgIn = cuda.mem_alloc(imgAr_In.nbytes)
    gpuimgOut = cuda.mem_alloc(imgAr_In.nbytes)
    func = cf.mod.get_function("ArCatMap")

    while (cf.sha2HashImage(imgAr)!=srchash):
        cuda.memcpy_htod(gpuimgIn, imgAr_In)
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(dim[2],1,1))
        cuda.memcpy_dtoh(imgAr_In, gpuimgOut)
        imgAr = (np.reshape(imgAr_In,dim)).astype(np.uint8)
    timer[2] = time.perf_counter() - timer[2]

    # Read image dimensions from sent file and resize if change
    f = open(cfg.DIM, "r")
    y, x = [int(i) for i in next(f).split()]
    f.close()

    if x!=y:
        imgAr = cv2.resize(imgAr,(x,y),interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(cfg.DEC_OUT, imgAr)

    overall_time = time.perf_counter() - overall_time

    # Print timing statistics
    print("Fractal XOR Inversion completed in " + str(timer[0]) +"s")
    print("MT Unshuffle completed in " + str(timer[1]) +"s")
    print("Arnold UnMapping completed in " + str(timer[2]) +"s")
    print("Decryption took " + str(np.sum(timer)) + "s out of " + str(overall_time) + "s of net execution time")

Decrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2                  # OpenCV
import numpy as np          # See above
import CONFIG as cfg        # Debug flags and constants
import CoreFunctions as cf  # Common functions
from os import chdir        # Path-setting
from time import perf_counter

#PyCUDA Import
import pycuda.driver as cuda
import pycuda.autoinit

chdir(cfg.PATH)

# Driver function
def Decrypt():
    #Initialize Timer
    perf_timer = np.zeros(5)
    misc_timer = np.zeros(6)
    overall_time = perf_counter()

    misc_timer[0] = overall_time
    # Read input image
    img = cv2.imread(cfg.ENC_OUT, 1)
    if img is None:
        print("File does not exist!")
        raise SystemExit(0)
    dim = img.shape
    misc_timer[0] = perf_counter() - misc_timer[0]

    misc_timer[1] = perf_counter()
    # Read log file
    with open(cfg.LOG, "r") as f:
        width = int(f.readline())
        height = int(f.readline())
        rounds = int(f.readline())
        fracID = int(f.readline())
    misc_timer[1] = perf_counter() - misc_timer[1]
    
    # Inverse Permutation: Intra-row/column rotation
    perf_timer[0] = perf_counter()
    U = cf.genRelocVec(dim[0],dim[1],cfg.P1LOG, ENC=False) # Col-rotation | len(U)=n, values from 0->m
    V = cf.genRelocVec(dim[1],dim[0],cfg.P2LOG, ENC=False) # Row-rotation | len(V)=m, values from 0->n
    perf_timer[0] = perf_counter() - perf_timer[0]
    
    misc_timer[2] = perf_counter()
    imgArr  = np.asarray(img).reshape(-1)
    gpuimgIn  = cuda.mem_alloc(imgArr.nbytes)
    gpuimgOut = cuda.mem_alloc(imgArr.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgArr)
    gpuU = cuda.mem_alloc(U.nbytes)
    gpuV = cuda.mem_alloc(V.nbytes)
    cuda.memcpy_htod(gpuU, U)
    cuda.memcpy_htod(gpuV, V)
    func = cf.mod.get_function("Dec_GenCatMap")

    func(gpuimgIn, gpuimgOut, gpuU, gpuV, grid=(dim[0],dim[1],1), block=(3,1,1))
    temp = gpuimgIn
    gpuimgIn = gpuimgOut
    gpuimgOut = temp
    perf_timer[1] = perf_counter()
    for i in range(cfg.PERM_ROUNDS):
        func(gpuimgIn, gpuimgOut, gpuU, gpuV, grid=(dim[0],dim[1],1), block=(3,1,1))
        temp = gpuimgIn
        gpuimgIn = gpuimgOut
        gpuimgOut = temp
    perf_timer[1] = perf_counter() - perf_timer[1]

    cuda.memcpy_dtoh(imgArr, gpuimgIn)
    img = (np.reshape(imgArr,dim)).astype(np.uint8)

    if cfg.DEBUG_IMAGES:
        cv2.imwrite(cfg.UNPERM, img)
    misc_timer[2] = perf_counter() - misc_timer[2] - perf_timer[1]

    # Inverse Fractal XOR Phase
    perf_timer[2] = perf_counter()
    img, misc_timer[3] = cf.FracXor(img, fracID)
    perf_timer[2] = perf_counter() - perf_timer[2]

    if cfg.DEBUG_IMAGES:
        cv2.imwrite(cfg.UNXOR, img)

    misc_timer[4] = perf_counter()
    # Ar Phase: Cat-map Iterations
    dim = img.shape
    imgArr = np.asarray(img).reshape(-1)
    imgShuffle = np.arange(start=0, stop=len(imgArr)/3, dtype=np.uint32)
    gpuimgIn = cuda.mem_alloc(imgShuffle.nbytes)
    gpuimgOut = cuda.mem_alloc(imgShuffle.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgShuffle)
    func = cf.mod.get_function("ArMapTable")
    misc_timer[4] = perf_counter() - misc_timer[4]

    iteration = 0
    func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(1,1,1))
    temp = gpuimgOut
    gpuimgOut = gpuimgIn
    gpuimgIn = temp
    
    # Recalculate mapping to generate lookup table
    perf_timer[3] = perf_counter()
    while iteration<rounds:
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(1,1,1))
        temp = gpuimgOut
        gpuimgOut = gpuimgIn
        gpuimgIn = temp
        iteration+=1
    perf_timer[3] = perf_counter() - perf_timer[3]

    misc_timer[5] = perf_counter()
    # Apply mapping
    gpuShuffle = gpuimgIn
    gpuimgIn = cuda.mem_alloc(imgArr.nbytes)
    gpuimgOut = cuda.mem_alloc(imgArr.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgArr)
    func = cf.mod.get_function("ArMapTabletoImg")
    perf_timer[4] = perf_counter()
    func(gpuimgIn, gpuimgOut, gpuShuffle, grid=(dim[0],dim[1],1), block=(3,1,1))
    perf_timer[4] = perf_counter() - perf_timer[4]
    cuda.memcpy_dtoh(imgArr, gpuimgOut)

    img = (np.reshape(imgArr,dim)).astype(np.uint8)

    # Resize image to OG dimensions if needed
    if height!=width:
        img = cv2.resize(img,(height,width),interpolation=cv2.INTER_CUBIC)
        dim = img.shape
    misc_timer[5] = perf_counter() - misc_timer[5] - perf_timer[4]

    cv2.imwrite(cfg.DEC_OUT, img)
    overall_time = perf_counter() - overall_time
    perf = np.sum(perf_timer)
    misc = np.sum(misc_timer)
    unaccounted = overall_time - perf - misc

    # Print timing statistics
    if cfg.DEBUG_TIMER:
        print("\nTarget: {} ({}x{})".format(cfg.ENC_IN, dim[1], dim[0]))

        print("\nPERF. OPS: \t{0:9.7f}s ({1:5.2f}%)".format(perf, perf/overall_time*100))
        print("Shuffle Gen:   \t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[0], perf_timer[0]/overall_time*100))
        print("Perm. Kernel:  \t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[1], perf_timer[1]/overall_time*100))
        print("Fractal XOR:   \t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[2], perf_timer[2]/overall_time*100))
        print("LUT Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[3], perf_timer[3]/overall_time*100))
        print("Mapping Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[4], perf_timer[4]/overall_time*100))
        
        print("\nMISC. OPS: \t{0:9.7f}s ({1:5.2f}%)".format(misc, misc/overall_time*100))
        print("Input Read:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[0], misc_timer[0]/overall_time*100)) 
        print("Log Read:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[1], misc_timer[1]/overall_time*100))
        print("Permute PreP:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[2], misc_timer[2]/overall_time*100)) 
        print("FracXOR PreP:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[3], misc_timer[3]/overall_time*100)) 
        print("LUT PreP:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[4], misc_timer[4]/overall_time*100)) 
        print("Mapping PreP:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[5], misc_timer[5]/overall_time*100))
        
        print("\nUnnaccounted: \t{0:9.7f}s ({1:5.2f}%)".format(unaccounted, unaccounted/overall_time*100))

        print("NET TIME:\t{0:7.5f}s\n".format(overall_time))

Decrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()
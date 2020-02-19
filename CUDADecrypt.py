import cv2                  # OpenCV
import time                 # Timing Execution
import numpy as np          # See above
import CONFIG as cfg        # Debug flags and constants
import CoreFunctions as cf  # Common functions
from os import chdir        # Path-setting

#PyCUDA Import
import pycuda.driver as cuda
import pycuda.autoinit

chdir(cfg.PATH)

# Driver function
def Decrypt():
    #Initialize Timer
    timer = np.zeros(5)
    overall_time = time.perf_counter()

    #Open the image
    img = cv2.imread(cfg.ENC_OUT, 1)
    if img is None:
        print("File does not exist!")
        raise SystemExit(0)
    dim = img.shape

    # Read log file
    with open(cfg.LOG, "r") as f:
        width = int(f.readline())
        height = int(f.readline())
        rounds = int(f.readline())
        fracID = int(f.readline())
    
    # Inverse Permutation: Intra-row/column rotation
    timer[2] = time.perf_counter()
    U = cf.genRelocVec(dim[0],dim[1],cfg.P1LOG, ENC=False) # Col-rotation | len(U)=n, values from 0->m
    V = cf.genRelocVec(dim[1],dim[0],cfg.P2LOG, ENC=False) # Row-rotation | len(V)=m, values from 0->n
    timer[2] = time.perf_counter() - timer[2]
    
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
    timer[3] = time.perf_counter()
    for i in range(cfg.PERM_ROUNDS):
        func(gpuimgIn, gpuimgOut, gpuU, gpuV, grid=(dim[0],dim[1],1), block=(3,1,1))
        temp = gpuimgIn
        gpuimgIn = gpuimgOut
        gpuimgOut = temp
    timer[3] = time.perf_counter() - timer[3]

    cuda.memcpy_dtoh(imgArr, gpuimgIn)
    img = (np.reshape(imgArr,dim)).astype(np.uint8)

    if cfg.DEBUG_IMAGES:
        cv2.imwrite(cfg.UNPERM, img)
        
    # Inverse Fractal XOR Phase
    timer[4] = time.perf_counter()
    img = cf.FracXor(img, fracID)
    timer[4] = time.perf_counter() - timer[4]

    if cfg.DEBUG_IMAGES:
        cv2.imwrite(cfg.UNXOR, img)
    # Ar Phase: Cat-map Iterations
    dim = img.shape
    imgArr = np.asarray(img).reshape(-1)
    imgShuffle = np.arange(start=0, stop=len(imgArr)/3, dtype=np.uint32)
    gpuimgIn = cuda.mem_alloc(imgShuffle.nbytes)
    gpuimgOut = cuda.mem_alloc(imgShuffle.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgShuffle)
    func = cf.mod.get_function("ArMapTable")

    iteration = 0
    func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(1,1,1))
    temp = gpuimgOut
    gpuimgOut = gpuimgIn
    gpuimgIn = temp
    
    # Recalculate mapping to generate lookup table
    timer[0] = time.perf_counter()
    while iteration<rounds:
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(1,1,1))
        temp = gpuimgOut
        gpuimgOut = gpuimgIn
        gpuimgIn = temp
        iteration+=1
    timer[0] = time.perf_counter() - timer[0]

    # Apply mapping
    gpuShuffle = gpuimgIn
    gpuimgIn = cuda.mem_alloc(imgArr.nbytes)
    gpuimgOut = cuda.mem_alloc(imgArr.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgArr)
    func = cf.mod.get_function("ArMapTabletoImg")
    timer[1] = time.perf_counter()
    func(gpuimgIn, gpuimgOut, gpuShuffle, grid=(dim[0],dim[1],1), block=(3,1,1))
    timer[1] = time.perf_counter() - timer[1]
    cuda.memcpy_dtoh(imgArr, gpuimgOut)

    img = (np.reshape(imgArr,dim)).astype(np.uint8)

    # Resize image to OG dimensions if needed
    if height!=width:
        img = cv2.resize(img,(height,width),interpolation=cv2.INTER_CUBIC)
        dim = img.shape

    cv2.imwrite(cfg.DEC_OUT, img)
    overall_time = time.perf_counter() - overall_time
    misc = overall_time - np.sum(timer)

    # Print timing statistics
    if cfg.DEBUG_TIMER:
        print("Target: {} ({}x{})".format(cfg.ENC_IN, dim[1], dim[0]))
        print("LUT Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(timer[0], timer[0]/overall_time*100))
        print("Mapping Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(timer[1], timer[1]/overall_time*100))
        print("Shuffle Gen:   \t{0:9.7f}s ({1:5.2f}%)".format(timer[2], timer[2]/overall_time*100))
        print("Perm. Kernel:  \t{0:9.7f}s ({1:5.2f}%)".format(timer[3], timer[3]/overall_time*100))
        print("Fractal XOR:   \t{0:9.7f}s ({1:5.2f}%)".format(timer[4], timer[4]/overall_time*100))
        print("Misc. ops: \t{0:9.7f}s ({1:5.2f}%)".format(misc, misc/overall_time*100))
        print("NET TIME:\t{0:7.5f}s\n".format(overall_time))

Decrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()
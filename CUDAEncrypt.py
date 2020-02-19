import os                   # Path setting and file-retrieval
import cv2                  # OpenCV
import time                 # Timing Execution
import numpy as np          # See above
import CONFIG as cfg        # Debug flags and constants
import CoreFunctions as cf  # Common functions
from shutil import rmtree    # Directory removal

#PyCUDA Import
import pycuda.driver as cuda
import pycuda.autoinit

os.chdir(cfg.PATH)

def PreProcess():
    # Check if ./images directory exists
    if not os.path.exists(cfg.SRC):
        print("Input directory does not exist!")
        raise SystemExit(0)
    else:
        if os.path.isfile(cfg.ENC_OUT):
            os.remove(cfg.ENC_OUT)
        if os.path.isfile(cfg.DEC_OUT):
            os.remove(cfg.DEC_OUT)
        
    # Check if ./temp directory exists
    if os.path.exists(cfg.TEMP):
        rmtree(cfg.TEMP)
    os.makedirs(cfg.TEMP)

    # Open Image
    img = cv2.imread(cfg.ENC_IN, 1)
    if img is None:
        print("File does not exist!")
        raise SystemExit(0)
    dim = img.shape

    # Write original dimensions to file and resize to square image if neccessary
    with open(cfg.LOG, 'w+') as f:
        f.write(str(dim[0]) + "\n")
        f.write(str(dim[1]) + "\n")
    return img, dim 

# Driver function
def Encrypt():
    #Initialize Timer
    timer = np.zeros(4)
    overall_time = time.perf_counter()
    
    # Read image and clear temp directories
    img, dim = PreProcess()

    # Ar Phase: Cat-map Iterations
    if dim[0]!=dim[1]:
        N = max(dim[0], dim[1])
        img = cv2.resize(img,(N,N), interpolation=cv2.INTER_CUBIC)
        dim = img.shape
    
    imgArr = np.asarray(img).reshape(-1)
    gpuimgIn = cuda.mem_alloc(imgArr.nbytes)
    gpuimgOut = cuda.mem_alloc(imgArr.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgArr)
    func = cf.mod.get_function("ArMapImg")

    rounds = int(cf.ArMapLen(dim[0])/8)
    with open(cfg.LOG, 'a+') as f:
        f.write(str(rounds)+"\n")
    
    func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(dim[2],1,1))
    temp = gpuimgOut
    gpuimgOut = gpuimgIn
    gpuimgIn = temp
    timer[3] = time.perf_counter()
    for i in range (max(rounds,5)):
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(dim[2],1,1))
        temp = gpuimgOut
        gpuimgOut = gpuimgIn
        gpuimgIn = temp
    timer[3] = time.perf_counter() - timer[3]

    cuda.memcpy_dtoh(imgArr, gpuimgIn)
    img = (np.reshape(imgArr,dim)).astype(np.uint8)

    if cfg.DEBUG_IMAGES:
        cv2.imwrite(cfg.ARMAP, img)

    # Fractal XOR Phase
    timer[0] = time.perf_counter()
    img = cf.FracXor(img)
    timer[0] = time.perf_counter() - timer[0]
    if cfg.DEBUG_IMAGES:
        cv2.imwrite(cfg.XOR, img)

    # Permutation: ArMap-based intra-row/column rotation
    timer[1] = time.perf_counter()
    U = cf.genRelocVec(dim[0],dim[1],cfg.P1LOG, ENC=True) # Col-rotation | len(U)=n, values from 0->m
    V = cf.genRelocVec(dim[1],dim[0],cfg.P2LOG, ENC=True) # Row-rotation | len(V)=m, values from 0->n
    timer[1] = time.perf_counter() - timer[1]
    
    imgArr  = np.asarray(img).reshape(-1)
    gpuimgIn  = cuda.mem_alloc(imgArr.nbytes)
    gpuimgOut = cuda.mem_alloc(imgArr.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgArr)
    gpuU = cuda.mem_alloc(U.nbytes)
    gpuV = cuda.mem_alloc(V.nbytes)
    cuda.memcpy_htod(gpuU, U)
    cuda.memcpy_htod(gpuV, V)
    func = cf.mod.get_function("Enc_GenCatMap")

    func(gpuimgIn, gpuimgOut, gpuU, gpuV, grid=(dim[0],dim[1],1), block=(3,1,1))
    temp = gpuimgIn
    gpuimgIn = gpuimgOut
    gpuimgOut = temp
    timer[2] = time.perf_counter()
    for i in range(cfg.PERM_ROUNDS):
        func(gpuimgIn, gpuimgOut, gpuU, gpuV, grid=(dim[0],dim[1],1), block=(3,1,1))
        temp = gpuimgIn
        gpuimgIn = gpuimgOut
        gpuimgOut = temp
    timer[2] = time.perf_counter() - timer[2]

    cuda.memcpy_dtoh(imgArr, gpuimgIn)
    img = (np.reshape(imgArr,dim)).astype(np.uint8)

    if cfg.DEBUG_IMAGES:
        cv2.imwrite(cfg.PERM, img)

    cv2.imwrite(cfg.ENC_OUT, img)
    overall_time = time.perf_counter() - overall_time
    misc = overall_time - np.sum(timer)

    # Print timing statistics
    if cfg.DEBUG_TIMER:
        print("Target: {} ({}x{})".format(cfg.ENC_IN, dim[1], dim[0]))      
        print("Fractal XOR: \t{0:9.7f}s ({1:5.2f}%)".format(timer[0], timer[0]/overall_time*100))
        print("Shuffle Gen: \t{0:9.7f}s ({1:5.2f}%)".format(timer[1], timer[1]/overall_time*100))
        print("Perm. Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(timer[2], timer[2]/overall_time*100))
        print("ArMap Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(timer[3], timer[3]/overall_time*100)) 
        print("Misc. ops: \t{0:9.7f}s ({1:5.2f}%)".format(misc, misc/overall_time*100))
        print("NET TIME:\t{0:7.5f}s\n".format(overall_time))
    
Encrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()
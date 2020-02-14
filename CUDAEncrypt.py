import os                   # Path setting and file-retrieval
import cv2                  # OpenCV
import time                 # Timing Execution
import random               # Obviously neccessary
import numpy as np          # See above
import CONFIG as cfg        # Debug flags and constants
import CoreFunctions as cf  # Common functions
import shutil               # Directory removal

#PyCUDA Import
import pycuda.driver as cuda
import pycuda.autoinit

os.chdir(cfg.PATH)

# Function to equalize Luma channel
def HistEQ(img_in):
    # Convert to LAB Space
    img_lab = cv2.cvtColor(img_in, cv2.COLOR_BGR2LAB)

    # Equalize L(Luma) channel
    img_lab[:,:,0] = cv2.equalizeHist(img_lab[:,:,0])

    # Convert back to RGB Space
    img_out = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    if cfg.DEBUG_HISTEQ and cfg.DEBUG_IMAGES:
        cv2.imwrite(cfg.HISTEQ, imgEQ)
    return img_out

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
        shutil.rmtree(cfg.TEMP)
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
    if dim[0]!=dim[1]:
        N = max(dim[0], dim[1])
        img = cv2.resize(img,(N,N), interpolation=cv2.INTER_CUBIC)
        dim = img.shape

    return img, dim 

# Driver function
def Encrypt():
    #Initialize Timer
    timer = np.zeros(6)
    overall_time = time.perf_counter()
    
    timer[0] = overall_time
    # Perform histogram equalization
    imgEQ, dim = PreProcess()
    if cfg.DO_HISTEQ:
        imgEQ = HistEQ(imgEQ)
    timer[0] = time.perf_counter() - timer[0]
    imgAr = imgEQ

    timer[1] = time.perf_counter()
    # Compute hash of imgEQ and write to text file
    imghash = cf.sha2HashFile(cfg.ENC_IN)
    timer[1] = time.perf_counter() - timer[1]
    with open(cfg.LOG, 'a+') as f:
        f.write(str(imghash)+"\n")
    
    # Ar Phase: Cat-map Iterations
    imgAr_In = np.asarray(imgAr).reshape(-1)
    gpuimgIn = cuda.mem_alloc(imgAr_In.nbytes)
    gpuimgOut = cuda.mem_alloc(imgAr_In.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgAr_In)
    func = cf.mod.get_function("ArMapImg")

    rounds = int(cf.ArMapLen(dim[0])/2)
    with open(cfg.LOG, 'a+') as f:
        f.write(str(rounds)+"\n")
    
    func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(dim[2],1,1))
    timer[2] = time.perf_counter()
    for i in range (max(rounds,5)):
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(dim[2],1,1))
        temp = gpuimgOut
        gpuimgOut = gpuimgIn
        gpuimgIn = temp
    timer[2] = time.perf_counter() - timer[2]

    cuda.memcpy_dtoh(imgAr_In, gpuimgIn)
    imgAr = (np.reshape(imgAr_In,dim)).astype(np.uint8)
    if cfg.DEBUG_IMAGES:
        cv2.imwrite(cfg.ARMAP, imgAr)

    timer[3] = time.perf_counter()
    # Fractal XOR Phase
    imgFr = cf.FracXor(imgAr, imghash)
    timer[3] = time.perf_counter() - timer[3]
    if cfg.DEBUG_IMAGES:
        cv2.imwrite(cfg.XOR, imgFr)

    # Permutation: ArMap-based intra-row/column rotation
    timer[4] = time.perf_counter()
    U = cf.genRelocVec(dim[0],dim[1],cfg.P1LOG, ENC=True) # Col-rotation | len(U)=n, values from 0->m
    V = cf.genRelocVec(dim[1],dim[0],cfg.P2LOG, ENC=True) # Row-rotation | len(V)=m, values from 0->n
    timer[4] = time.perf_counter() - timer[4]
    
    imgArray  = np.asarray(imgFr).reshape(-1)
    gpuimgIn  = cuda.mem_alloc(imgArray.nbytes)
    gpuimgOut = cuda.mem_alloc(imgArray.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgArray)
    gpuU = cuda.mem_alloc(U.nbytes)
    gpuV = cuda.mem_alloc(V.nbytes)
    cuda.memcpy_htod(gpuU, U)
    cuda.memcpy_htod(gpuV, V)
    func = cf.mod.get_function("Enc_GenCatMap")

    func(gpuimgIn, gpuimgOut, gpuU, gpuV, grid=(dim[0],dim[1],1), block=(3,1,1))
    temp = gpuimgIn
    gpuimgIn = gpuimgOut
    gpuimgOut = temp
    timer[5] = time.perf_counter()
    for i in range(cfg.PERM_ROUNDS):
        func(gpuimgIn, gpuimgOut, gpuU, gpuV, grid=(dim[0],dim[1],1), block=(3,1,1))
        temp = gpuimgIn
        gpuimgIn = gpuimgOut
        gpuimgOut = temp
    timer[5] = time.perf_counter() - timer[5]

    cuda.memcpy_dtoh(imgArray, gpuimgIn)
    imgMT = (np.reshape(imgArray,dim)).astype(np.uint8)

    if cfg.DEBUG_IMAGES:
        cv2.imwrite(cfg.MT, imgMT)

    cv2.imwrite(cfg.ENC_OUT, imgMT)
    overall_time = time.perf_counter() - overall_time
    misc = overall_time - np.sum(timer)

    # Print timing statistics
    if cfg.DEBUG_TIMER:
        print("Target: {} ({}x{})".format(cfg.ENC_IN, dim[1], dim[0]))
        print("Pre-processing:\t{0:9.7f}s ({1:5.2f}%)".format(timer[0], timer[0]/overall_time*100))
        print("Hashing:\t{0:9.7f}s ({1:5.2f}%)".format(timer[1], timer[1]/overall_time*100))
        print("ArMap Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(timer[2], timer[2]/overall_time*100))       
        print("Fractal XOR:\t{0:9.7f}s ({1:5.2f}%)".format(timer[3], timer[3]/overall_time*100))
        print("Shuffle Gen:\t{0:9.7f}s ({1:5.2f}%)".format(timer[4], timer[4]/overall_time*100))
        print("Perm. Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(timer[5], timer[5]/overall_time*100))
        print("Misc. ops: \t{0:9.7f}s ({1:5.2f}%)".format(misc, misc/overall_time*100))
        print("Net Time:\t{0:7.5f}s\n".format(overall_time))
    
Encrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()
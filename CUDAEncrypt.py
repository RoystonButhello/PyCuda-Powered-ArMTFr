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
    if cfg.DEBUG_HISTEQ:
        cv2.imshow('Input image', img_in)
        cv2.imshow('Histogram equalized', img_out)
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
    timer = np.zeros(5)
    overall_time = time.perf_counter()
    
    timer[0] = overall_time
    # Perform histogram equalization
    imgEQ, dim = PreProcess()
    if cfg.DO_HISTEQ:
        imgEQ = HistEQ(imgEQ)
    timer[0] = time.perf_counter() - timer[0]
    cv2.imwrite(cfg.HISTEQ, imgEQ)
    imgAr = imgEQ

    timer[1] = time.perf_counter()
    # Compute hash of imgEQ and write to text file
    imghash = cf.sha2HashImage(imgEQ)
    timer[1] = time.perf_counter() - timer[1]
    with open(cfg.LOG, 'a+') as f:
        f.write(str(imghash)+"\n")
    
    timer[2] = time.perf_counter()
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
    kernel_time = time.perf_counter()
    for i in range (max(rounds,5)):
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(dim[2],1,1))
        temp = gpuimgOut
        gpuimgOut = gpuimgIn
        gpuimgIn = temp
    kernel_time = time.perf_counter() - kernel_time

    cuda.memcpy_dtoh(imgAr_In, gpuimgIn)
    imgAr = (np.reshape(imgAr_In,dim)).astype(np.uint8)
    timer[2] = time.perf_counter() - timer[2]
    cv2.imwrite(cfg.ARMAP, imgAr)

    timer[3] = time.perf_counter()
    # Fractal XOR Phase
    imgFr = cf.FracXor(imgAr, imghash)
    timer[3] = time.perf_counter() - timer[3]
    cv2.imwrite(cfg.XOR, imgFr)

    timer[4] = time.perf_counter()
    # MT Phase: Intra-column pixel shuffle
    imgMT = cf.MTShuffle(imgFr, imghash)
    timer[4] = time.perf_counter() - timer[4]
    cv2.imwrite(cfg.MT, imgMT)

    cv2.imwrite(cfg.ENC_OUT, imgMT)
    overall_time = time.perf_counter() - overall_time
    misc = overall_time - np.sum(timer)

    # Print timing statistics
    if cfg.DEBUG_TIMER:
        print("Target: {} ({}x{})".format(cfg.ENC_IN, dim[1], dim[0]))
        print("Pre-processing:\t{0:9.7f}s ({1:5.2f}%)".format(timer[0], timer[0]/overall_time*100))
        print("Hashing:\t{0:9.7f}s ({1:5.2f}%)".format(timer[1], timer[1]/overall_time*100))
        print("Arnold Mapping:\t{0:9.7f}s ({1:5.2f}%)".format(timer[2], timer[2]/overall_time*100))
        print("Kernel Exec.:\t{0:9.7f}s ({1:5.2f}%)".format(kernel_time, kernel_time/overall_time*100))        
        print("Fractal XOR:\t{0:9.7f}s ({1:5.2f}%)".format(timer[3], timer[3]/overall_time*100))
        print("PRNG Shuffle:\t{0:9.7f}s ({1:5.2f}%)".format(timer[4], timer[4]/overall_time*100))
        print("Misc. ops: \t{0:9.7f}s ({1:5.2f}%)".format(misc, misc/overall_time*100))
        print("Net Time:\t{0:7.5f}s\n".format(overall_time))
    
Encrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()
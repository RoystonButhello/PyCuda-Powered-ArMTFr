import os                   # Path setting and file-retrieval
import cv2                  # OpenCV
import time                 # Timing Execution
import random               # Obviously neccessary
import numpy as np          # See above
import CONFIG as cfg        # Debug flags and constants
import CoreFunctions as cf  # Common functions
import shutil           # Directory removal

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
    with open(cfg.DIM, 'w+') as f:
        f.write(str(dim[0]) + " " + str(dim[1]))
    if dim[0]!=dim[1]:
        N = max(dim[0], dim[1])
        img = cv2.resize(img,(N,N), interpolation=cv2.INTER_LANCZOS4)
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
    with open(cfg.HASH, 'w+') as f:
        f.write(str(imghash))
    
    #Clear ArMap debug files
    if cfg.DEBUG_ARMAP:
        os.makedirs(cfg.ARTEMP)
    
    timer[2] = time.perf_counter()
    # Ar Phase: Cat-map Iterations
    imgAr_In = np.asarray(imgAr).reshape(-1)
    gpuimgIn = cuda.mem_alloc(imgAr_In.nbytes)
    gpuimgOut = cuda.mem_alloc(imgAr_In.nbytes)
    func = cf.mod.get_function("ArCatMap")
    
    for i in range (3, int(cf.ArMapLen(dim[0])/2)):
        cuda.memcpy_htod(gpuimgIn, imgAr_In)
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(dim[2],1,1))
        cuda.memcpy_dtoh(imgAr_In, gpuimgOut)
        # Write intermediate files if debugging is enabled
        if cfg.DEBUG_ARMAP:
            imgAr = (np.reshape(imgAr_In,dim)).astype(np.uint8)
            cv2.imwrite(cfg.ARTEMP + str(i) + ".png", imgAr)

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

    # Print timing statistics
    if cfg.DEBUG_TIMER:
        print("Pre-processing completed in " + str(timer[0]) +"s")
        print("Hashing completed in " + str(timer[1]) +"s")
        print("Arnold Mapping completed in " + str(timer[2]) +"s")
        print("Fractal XOR completed in " + str(timer[3]) +"s")
        print("MT Shuffle completed in " + str(timer[4]) +"s")
        print("\nEncryption took " + str(np.sum(timer)) + "s out of " + str(overall_time) + "s of net execution time\n")
    
Encrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()
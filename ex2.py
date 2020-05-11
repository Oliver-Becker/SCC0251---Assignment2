#   Óliver Savastano Becker 10284890
#   Rafael Farias Roque 10295412    
#   SCC251 
#   Assignment 2
#   github.

import numpy as np
import imageio
import matplotlib.pyplot as plt

def padding(img, size):
    
    a = int((size-1)/2)
    m,n = img.shape
    
    img_padding = np.zeros((m+(a*2),n+(a*2)))
    
    img_padding[a:-a, a:-a] = img
    
    return img_padding 

def  unpadding(img, size):
    a = int((size-1)/2)
    img_up = img[a:-a, a:-a]
    return img_up

def convolution(img, w):
    N,M = img.shape
    m,n = w.shape
    new_img = np.zeros(img.shape, dtype= np.float64)

    w_flip = np.flip(np.flip(w,0),1)
    a = int((m-1)/2)
    b = int((n-1)/2)

    for x in range(a, N-a):
        for y in range(b, M-b):
            region_img = img[x-a:x+a+1, y-b:y+b+1]
            new_img[x,y] = np.sum(np.multiply(region_img, w_flip))
            
    
    return new_img

def normalize(img):
    img_new = ((img- np.amin(img))*255)/(np.amax(img) - np.amin(img))
    return img_new

def sq_error(img, img_f):
    error = np.sqrt(np.sum(np.power(np.subtract(img_f, img), 2)))
    return error

#def bilateral_filter():

def laplacian_filter(img, c, kernel):
        
    img_p = padding(img, 3)
    w = np.matrix([[0, -1, 0], [-1, 4, -1],[0, -1, 0]])
    if kernel == 2: 
        w =  np.matrix([[-1, -1, -1], [-1, 8, -1],[-1, -1, -1]])
    
    img_lf = convolution(img_p, w)

    img_lf = unpadding(img_lf, 3)
    
    img_lf = normalize(img_lf)
    
    img_lf = (img_lf*c) + img    
    
    img_lf = normalize(img_lf)  
    
    return img_lf

#def vignette_filter():
    
# reading the input
filename = str(input())
img = imageio.imread(filename)
img = img.astype(np.float64)
method = int(input())
S = int(input())

if method == 1:
    n =  int(input())
    sig_s = float(input())
    sig_r = float(input())
    #new_img = bilateral_filter()
    #print(sq_error(img, new_img))
    if S == 1:
        imageio.imwrite(filename, new_img.astype(np.uint8))
elif method == 2:
    c = float(input())
    kernel = int(input())
    new_img = laplacian_filter(img, c, kernel)
    print(sq_error(img, new_img))
    if S == 1:
        imageio.imwrite(filename, new_img.astype(np.uint8))
    
elif method == 3:
    sig_row = float(input())
    sig_col = float(input())
    #new_img = vignette_filter()
    #print(sq_error(img, new_img))
    if S == 1:
        imageio.imwrite(filename, new_img.astype(np.uint8))
    
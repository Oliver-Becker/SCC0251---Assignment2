#   Óliver Savastano Becker 10284890
#   Rafael Farias Roque 10295412    
#   SCC251 
#   Assignment 2
#   https://github.com/Oliver-Becker/SCC0251---Assignment2

import numpy as np
import imageio

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

def gaussian_kernel(x, sig):
    g = float((np.exp((x*x)/(-2*sig*sig)))/(2*np.pi* np.power(sig,2)))
    return g
    
def bilateral_filter(img, n, sig_s, sig_r):
    img_p = padding(img, n)
    
    M, N = img_p.shape
    a = int((n-1)/2)
    
    g_spatial = np.zeros((n,n))
    
    for x in range(0, n):
        for y in range(0, n):
            g_spatial[x][y] = gaussian_kernel(np.sqrt((x-a)*(x-a) + (y-a)*(y-a)), sig_s)
            
            
    new_img = np.zeros((M,N))
            
    for x in range (a, M-a):
        for y in range(a, N-a):
            g_range = np.zeros((n,n))
            for x2 in range(0, n):
                for y2 in range(0, n):
                    g_range[x2][y2] =  gaussian_kernel(img_p[x+(x2-a)][y+(y2-a)] - img_p[x][y], sig_r)
            wi = np.multiply(g_range, g_spatial)
            Wp = np.sum(wi)
            If = np.sum(np.multiply(wi,img_p[x-a:x+a+1, y-a:y+a+1]))
            new_img[x][y] = If/Wp
            
    new_img = unpadding(new_img, n)
    return new_img
    
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

def vignette_filter(img, sig_row, sig_col):
    M,N = img.shape
    a = np.ceil((M/2)-1)
    b = np.ceil((N/2)-1)
    g_row = np.zeros((1,M))
    g_col = np.zeros((1,N))
    for x in range(0, M):
        g_row[0][x] = gaussian_kernel(x-a, sig_row)
    for x in range(0, N):
        g_col[0][x] = gaussian_kernel(x-b, sig_col)
    
    g_row = g_row.T
    
    g_filter = np.matmul(g_row, g_col)

    img_vf = np.multiply(g_filter, img)
    
    img_vf = normalize(img_vf)

    return img_vf

# reading the input
filename = str(input()).rstrip()
img = imageio.imread(filename)
img = img.astype(np.float64)
method = int(input())
S = int(input())

if method == 1:
    n =  int(input())
    sig_s = float(input())
    sig_r = float(input())
    new_img = bilateral_filter(img, n, sig_s, sig_r)
    print(sq_error(img, new_img))
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
    new_img = vignette_filter(img, sig_row, sig_col)
    print(sq_error(img, new_img))
    if S == 1:
        imageio.imwrite(filename, new_img.astype(np.uint8))
    
#!/usr/bin/python2.7
import os
import struct
import sys
import math
import numpy as np
import argparse
import random

def calling(  ):
    import sys
    return sys._getframe(1).f_code.co_name

def apply_kernel(array, kernel):
    return np.sum(np.multiply(array, kernel)) 

def copy_part(input, startr, startc, endr, endc):
    nh = endr - startr + 1
    nw = endc - startc + 1
    out = np.zeros([nh, nw], dtype = float)
    nr = -1
    nc = -1
    for r in range(startr, endr + 1):
        nr = nr + 1
        nc = -1
        for c in range(startc, endc + 1):
            nc = nc + 1
            val = input[r][c]
            out[nr][nc] = val
    return out        

def add_pad(input, hpad = 0, vpad = 0):
    ih = input.shape[0]
    iw = input.shape[1]
    nh = ih + hpad
    nw = iw + vpad
    output = np.zeros([nh, nw], dtype = float)
    for r in range(ih):
        for c in range(iw):
            output[r][c] = input [r][c]
    return output
    
    

def conv_op_2d(input, kernel, hstride = 1, vstride = 1, stride = -1, pad = "NONE"):
    print "Input shape:", input.shape
    print "Input:\n", input 
    print "Kernel shape:", kernel.shape
    print "Kernel:\n", kernel
    print "Padding:", pad
    
    if stride != -1:
        hstride = stride
        vstride = stride
    print "Horizontal stride:", hstride
    print "Vertical stride:", vstride

    kh = kernel.shape[0]
    kw = kernel.shape[1]
    iho = input.shape[0]
    iwo = input.shape[1]
    if pad == "SAME":
        input = add_pad(input, hpad = kh-1, vpad = kw-1)
    ih = input.shape[0]
    iw = input.shape[1]
    oh = ((ih - kh)/hstride) + 1
    ow = ((iw - kw)/vstride) + 1
    name = calling() 
    print "(", name, ")", "Output size:", oh , "x" , ow
    output = np.zeros([oh, ow], dtype = float)
    outr = -1
    outc = -1		
    for r in range(0, iho, vstride):
	outr += 1
	outc = -1
        for c in range(0, iwo, hstride):
	    outc += 1
	    conv_val = 0.0
            if r+kh-1 < ih and c+kw-1 < iw:		
            	part = copy_part(input, r, c, r+kh-1, c+kw-1)
            	conv_val = apply_kernel(part, kernel)
            	output[outr][outc] = conv_val
    return output

def rand_pixel(array):
    h = array.shape[0]
    w = array.shape[1]
    for r in range(h):
	for c in range(w):
	    array[r][c] = random.randint(-2, 1)


def fillarray_img(array, filler, count):
    if filler == 'inc':
        array.fill(count + 1)
    elif filler == 'rand':
        array.fill(random.randint(1, 2))
	rand_pixel(array)
    else:
        print "Wrong filler type.."
        exit()
    
def fillarray_wt(array, filler):
    print "Shape:", array.shape
    val = 0
    if filler == 'mod':
        for i in range(array.shape[0]):
            f = val % array.shape[1]
            array[i].fill(f+1)
            val = val + 1
    elif filler == 'inc':
        for i in range(array.shape[0]):
            array[i].fill(val+1)
            val = val + 1
    elif filler == 'rand':
        for i in range(array.shape[0]):
            array[i].fill(random.randint(1, 2))
    else:
        print "Wrong filler type.."
        exit()

def conv_op_3d(input, kernel, hstride = 1, vstride = 1, stride = -1, pad = "NONE"):
    print "Input shape:", input.shape
    print "Input image:\n", input
    print "Kernel shape:", kernel.shape
    print "Weight kernels:\n", kernel
    print "Padding:", pad
    
    if stride != -1:
        hstride = stride
        vstride = stride
    print "Horizontal stride:", hstride
    print "Vertical stride:", vstride
    oc = kernel.shape[0]
    ic = input.shape[0]
    print "Input channels:", ic
    print "Output channels:", oc


    kh = kernel.shape[2]
    kw = kernel.shape[3]

    ih = input.shape[1]
    iw = input.shape[2]

    if pad == "SAME":
        ih = ih + 2 * ((kh - 1)/2)
        iw = iw + 2 * ((kw -1)/2)
    
    oh = ((ih - kh)/hstride) + 1
    ow = ((iw - kw)/vstride) + 1

    print "Output (Row X Column):(", oh, "X", ow, ")"
    conv_tmp = np.zeros([oc, ic, oh, ow], dtype = float)
    
    for o in range(oc):
        for i in range(ic):
            img = input[i]
            wt = kernel[o][i]
            cnv = conv_op_2d(img, wt, stride = stride, pad = pad)
            conv_tmp[o][i] = cnv
    print "Conv Tmp:\n", conv_tmp

    conv = np.zeros([oc, oh, ow], dtype = float)

    for o in range(oc):
        for h in range(oh):
            for w in range(ow):
                val = 0.0
                for i in range(ic):
                    val = val + conv_tmp[o][i][h][w]    
                conv[o][h][w] = val
    return conv

def max_pool_3d(input, kh = 2, kw = 2, kernel_size = -1, hstride = 2, vstride = 2, stride = -1):
    print "Max pool input shape:", input.shape
    print "Kernel:", kernel_size, "X", kernel_size
    if stride != -1:
	hstride = stride
	vstride = stride
    print "HStride:", hstride
    print "VStride:", vstride
    oc = input.shape[0]
    ih = input.shape[1]
    iw = input.shape[2]
    if kernel_size != -1:
	kh = kernel_size
	kw = kernel_size
    
    oh = ((ih - kh)/hstride) + 1
    ow = ((iw - kw)/vstride) + 1

    print ("Max pool output: %d x %d" % (oh, ow))	
    pool_out = np.zeros([oc, oh, ow], dtype = float)
    outr = -1
    outc = -1	
    for o in range(oc):
	outr = -1	
    	for r in range(0, ih, vstride):
	    outr += 1
	    outc = -1
    	    for c in range(0, iw, hstride):
		outc += 1
    	        if r+kh-1 < ih and c+kw-1 < iw:		
    	        	part = copy_part(input[o], r, c, r+kh-1, c+kw-1)
			max_val = np.amax(part)	
			#print part, "Max:", np.amax(part)
			#print ("o:%d, outr:%d, outc:%d" % (o, outr, outc))
			pool_out[o][outr][outc] = max_val
    return pool_out 
	
    	
def avg_pool_3d(input, kh = 2, kw = 2, kernel_size = -1, hstride = 2, vstride = 2, stride = -1):
    print "Avg pool input shape:", input.shape
    print "Kernel:", kernel_size, "X", kernel_size
    if stride != -1:
	hstride = stride
	vstride = stride
    print "HStride:", hstride
    print "VStride:", vstride
    oc = input.shape[0]
    ih = input.shape[1]
    iw = input.shape[2]
    if kernel_size != -1:
	kh = kernel_size
	kw = kernel_size
    
    oh = ((ih - kh)/hstride) + 1
    ow = ((iw - kw)/vstride) + 1

    print ("Avg pool output: %d x %d" % (oh, ow))	
    pool_out = np.zeros([oc, oh, ow], dtype = float)
    outr = -1
    outc = -1	
    for o in range(oc):
	outr = -1	
    	for r in range(0, ih, vstride):
	    outr += 1
	    outc = -1
    	    for c in range(0, iw, hstride):
		outc += 1
    	        if r+kh-1 < ih and c+kw-1 < iw:		
    	        	part = copy_part(input[o], r, c, r+kh-1, c+kw-1)
			mean_val = np.mean(part)	
			#print part, "Mean:", np.mean(part)
			#print ("o:%d, outr:%d, outc:%d" % (o, outr, outc))
			pool_out[o][outr][outc] = mean_val
    return pool_out 

def ReLu(input):
    oc = input.shape[0]
    ih = input.shape[1]
    iw = input.shape[2]
    for o in range(oc):
	for r in range(ih):
	    for c in range(iw):
                input[o][r][c] = max(0, input[o][r][c])		
    return input	

def Leaky_ReLu(input, alpha = 0.01):
    oc = input.shape[0]
    ih = input.shape[1]
    iw = input.shape[2]
    for o in range(oc):
	for r in range(ih):
	    for c in range(iw):
                input[o][r][c] = max(alpha * input[o][r][c], input[o][r][c])		
    return input	

def dLeaky_ReLu(input, alpha = 0.01):
    oc = input.shape[0]
    ih = input.shape[1]
    iw = input.shape[2]
    for o in range(oc):
	for r in range(ih):
	    for c in range(iw):
                input[o][r][c] = max(alpha, input[o][r][c])		
    return input	

def fully_connected_3d(input):
    print "Fully connected input shape:", input.shape
    print input
    (oc, h, w) = input.shape
    fc = input.reshape(oc * h * w, 1)
    print "After flatten: Shape:", fc.shape
    print fc

def main(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-oc','--op_channels', default=1, help='Number of output channels', required=False)
    parser.add_argument('-ic','--ip_channels', default=1, help='Number of output channels', required=False)
    parser.add_argument('-iw','--img_width', default=8, help='Input image width', required=False)
    parser.add_argument('-ih','--img_height', default=8, help='Input image height', required=False)
    parser.add_argument('-k','--kernel', default=5, help='Kernel size', required=False)
    parser.add_argument('-p','--padding', default='SAME', help='Padding(SAME/NONE)', required=False)
    parser.add_argument('-f','--filler', default='rand', help='Weight matrix filler type(mod/inc/rand)', required=False)
    args = parser.parse_args()
    
    oc = int(args.op_channels)
    ic = int(args.ip_channels)
    iw = int(args.img_width)
    ih = int(args.img_height)
    K  = int(args.kernel)
    filler = args.filler
    print "Input image size:(", ih ,"x", iw, ")"
    print "Number of output channels:", oc
    print "Number of input channels:", ic
    print "Kernel size:", K
    print "Filler type:", filler

    ker = np.array([[1, 0, 1],
                  [0, 1, 0],
                  [1, 0, 1]
                 ])
    inp = np.array([[1,1,1,0,0],
                  [0,1,1,1,0],
                  [0,0,1,1,1],
                  [0,0,1,1,0],
                  [0,1,1,0,0]      
                 ])
    
    image = np.zeros([ic, ih, iw], dtype = float)
    for i in range(ic):
        fillarray_img(image[i], filler, i)

    weights = np.zeros([oc, ic, K, K], dtype = float)
    for i in range(oc):
        fillarray_wt(weights[i], filler)
   
    print "Image:\n", image
    print "Weights:\n", weights 
    
    conv_out = conv_op_3d(image, weights, stride = 1, pad = args.padding)
    print "Convolution output shape:", conv_out.shape
    print "Convolve output:\n", conv_out
    
    max_pool_out = max_pool_3d(conv_out, kernel_size = 2, stride = 2)
    print "Max pool out shape:", max_pool_out.shape
    print max_pool_out	

    avg_pool_out = avg_pool_3d(conv_out, kernel_size = 2, stride = 2)
    print "Avg pool out shape:", avg_pool_out.shape
    print avg_pool_out

    relu_out = ReLu(max_pool_out)
    print "ReLu shape:", relu_out.shape
    print relu_out		
    
    fully_connected_3d(relu_out)	
     

if __name__ == "__main__":
    main(sys.argv)

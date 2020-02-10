#!/usr/bin/python2.7
import os
import struct
import sys
import math
import numpy as np
import argparse
import random
import cnn_layers as cl

def main(argv):
    filler = 'rand'
    ic = 1          #Input channel
    ih = 28         #Image height
    iw = 28         #Image width
    oc = 20         #Output channel
    K  = 5          #Kernel size
    image = np.zeros([ic, ih, iw], dtype = float)   #Image size 10 x 10 x 4
    for i in range(ic):
        cl.fillarray_2d(image[i], filler, i)

    weights_conv0 = np.zeros([oc, ic, K, K], dtype = float)     #Output channel = 1, Kernel size: 3 x 3, Input channel = 4
    for i in range(oc):
        cl.fillarray_3d(weights_conv0[i], filler)
    
    conv0_out = cl.conv_op_3d(image, weights_conv0, stride = 1, pad = 'SAME')
    print "Convolution0 output shape:", conv0_out.shape
    print "Convolve0 output:\n", conv0_out

    max_pool_out = cl.max_pool_3d(conv0_out, kernel_size = 2, stride = 2)   
    print "Max pool out shape:", max_pool_out.shape
    print max_pool_out	

    ic = 20
    K  = 5
    oc = 50
    weights_conv1 = np.zeros([oc, ic, K, K], dtype = float)     #Output channel = 2, Kernel size = 3 x 3, Input channel = 1
    for i in range(oc):
        cl.fillarray_3d(weights_conv1[i], filler)
    
    conv1_out = cl.conv_op_3d(max_pool_out, weights_conv1, stride = 1, pad = 'SAME')
    print "Convolution1 output shape:", conv1_out.shape
    print "Convolve1 output:\n", conv1_out
    
    max_pool_out = cl.max_pool_3d(conv1_out, kernel_size = 2, stride = 2)   
    print "Max pool out shape:", max_pool_out.shape
    print max_pool_out

    flat = cl.flatten(max_pool_out)     #Flatten ReLu output
    fc_out_n = 500
    fc0_weights = np.zeros([fc_out_n, flat.shape[0]], dtype = float)
    for i in range(fc_out_n):
        cl.fillarray_1d(fc0_weights[i], filler, i)
    fc0_out = cl.fully_connected_1d(flat, fc0_weights, fc_out_n)	
    print "FC0 output shape:", fc0_out.shape
    print fc0_out
    
    relu_out = cl.ReLu_1d(fc0_out)
    print "ReLu shape:", relu_out.shape
    print relu_out		
   
    
    fc_out_n = 10
    fc1_weights = np.zeros([fc_out_n, relu_out.shape[0]], dtype = float)
    for i in range(fc_out_n):
        cl.fillarray_1d(fc1_weights[i], filler, i)
    fc1_out = cl.fully_connected_1d(relu_out, fc1_weights, fc_out_n)	
    print "FC1 output shape:", fc1_out.shape
    print fc1_out
    
    soft_max_out = cl.softmax(fc1_out)
    print "Softmax shape:", soft_max_out.shape
    print soft_max_out 


if __name__ == "__main__":
    main(sys.argv)

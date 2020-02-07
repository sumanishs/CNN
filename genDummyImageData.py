#!/local/workspace/tools/anaconda2/bin/python2.7
import os
import struct
import sys
import math
import numpy as np
import argparse
import random

def genImageData(sysarray, in_list, tf, w, h):
    print "Number of weight matrices set(s):", len(in_list)   
    for mat in in_list:
        print mat
    IPch = len(in_list)    
    print "Systolic array structure:"
    print sysarray
    M = sysarray.shape[0]
    N = sysarray.shape[1]
    print "Systolic array size (M x N):", M, " x ", N
    print "Number of input channel(s):", IPch
    chunks = IPch/N
    if IPch % N > 0:
        chunks = chunks + 1
    print "Required input channel chunks:", chunks
    for ic in range(chunks):
        for x in range(w):
            for y in range(h):
                for n in range(N):
                    icdx = ic * N + n
                    val = 0
                    if icdx < len(in_list):
                        mat = in_list[icdx]
                        val = mat[x][y]
                    print val,
                    tf.write("%d\n" % val)
                print ""        

def fillarray(array, filler, count):
    if filler == 'inc':
        array.fill(count + 1)
    elif filler == 'rand':
        array.fill(random.randint(0, 3))
    else:
        print "Wrong filler type.."
        exit()

def main(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m','--row', default=4, help='Systolic array rows', required=False)
    parser.add_argument('-n','--col', default=4, help='Systolic array columns', required=False)
    parser.add_argument('-ic','--ip_channels', default=1, help='Number of output channels', required=False)
    parser.add_argument('-f','--filler', default='inc', help='Weight matrix filler type(inc/rand)', required=False)
    parser.add_argument('-of','--out_file', default='dummy_input_image.txt', help='Output file name', required=False)
    parser.add_argument('-iw','--width', default=32, help='Image width', required=False)
    parser.add_argument('-ih','--height', default=32, help='Image height', required=False)
    args = parser.parse_args()

    M = int(args.row)
    N = int(args.col)
    IPch = int(args.ip_channels)
    filler = args.filler
    w = int(args.width)
    h = int(args.height)
    print "Systolic array size (M x N):", M, " x ", N
    print "Weight filler:", filler
    print "Number of input channels:", IPch
    print "Image size:(W x H):" , w , "x", h  
    in_list = []
    
    np.random.seed(0)
    for i in range(IPch):
        image = np.random.randint(1, size=(w, h))
        fillarray(image, filler, len(in_list))
        in_list.append(image)
    
    sysarray = np.zeros([M, N], dtype = int)    
    tf = open(args.out_file, "w")
    genImageData(sysarray, in_list, tf, w, h)
    print "Dumped dummy weights in <", args.out_file, ">"
    tf.close()

if __name__ == "__main__":
    main(sys.argv)

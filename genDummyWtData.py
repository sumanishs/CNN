#!/local/workspace/tools/anaconda2/bin/python2.7
import os
import struct
import sys
import math
import numpy as np
import argparse
import random


def genWtData(sysarray, wt_list, tf):
    print "Number of weight matrices set(s):", len(wt_list)   
    for mat in wt_list:
        print mat
    OPch = len(wt_list)    
    print "Systolic array structure:"
    print sysarray
    M = sysarray.shape[0]
    N = sysarray.shape[1]
    K = wt_list[0].shape[1]
    IPch = wt_list[0].shape[0]
    print "Systolic array size (M x N):", M, " x ", N
    print "Kernel size:", K
    print "Number of output channel(s):", OPch
    print "Number of input channel(s):", IPch
    outseti = OPch/M
    if OPch % M > 0:
        outseti = outseti + 1
    #print "Required Output Chunks:", outseti
    inseti = IPch/N
    if IPch % M > 0:
        inseti = inseti + 1
    #print "Required Input Chunks:", inseti
    for oc in range(outseti):
        for ic in range(inseti):
            for x in range(K):
                for y in range(K):
                    for m in range(M):
                        ocdx = oc*M + m
                        #print "ocdx:", ocdx
                        for n in range(N):
                            icdx = ic*N + n
                            #print "icdx:", icdx
                            if ocdx < len(wt_list):
                                mat = wt_list[ocdx]
                                #print "mat:", mat
                                if icdx < mat.shape[0]:
                                    layermat = mat[icdx]
                                    #print "layermat:", layermat
                                    val = layermat[x][y]
                                    sysarray[m][n] = val
                    print "Systolic array:"
                    print sysarray
                    t = np.transpose(sysarray)
                    #print "Trans:\n", t
                    c = np.reshape(t, M * N)
                    #print "Reshaped:", c
                    print "Writing: ",
                    for l in range(M * N):
                        tf.write("%d\n" % c[l])
                        print c[l],
                    print "" 
                

def fillarray(array, filler):
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
            array[i].fill(random.randint(0, 3))
    else:
        print "Wrong filler type.."
        exit()
        
def main(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m','--row', default=4, help='Systolic array rows', required=False)
    parser.add_argument('-n','--col', default=4, help='Systolic array columns', required=False)
    parser.add_argument('-oc','--op_channels', default=1, help='Number of output channels', required=False)
    parser.add_argument('-ic','--ip_channels', default=1, help='Number of output channels', required=False)
    parser.add_argument('-k','--kernel', default=5, help='Kernel size', required=False)
    parser.add_argument('-f','--filler', default='inc', help='Weight matrix filler type(mod/inc/rand)', required=False)
    parser.add_argument('-of','--out_file', default='dummy_weights.txt', help='Output file name', required=False)

    args = parser.parse_args()

    M = int(args.row)
    N = int(args.col)
    OPch = int(args.op_channels)
    IPch = int(args.ip_channels)
    K = int(args.kernel)
    filler = args.filler
    print "Systolic array size (M x N):", M, " x ", N
    print "Number of output channel(s):", OPch
    print "Kernel size:", K
    print "Weight filler:", filler
    wt_list = []
    
    np.random.seed(0)
    for i in range(OPch):
        wt = np.random.randint(1, size=(IPch, K, K))
        fillarray(wt, filler)
        wt_list.append(wt)
   
    opcount = 0
    for mat in wt_list:
        print "Set of weight matrices for Output channel:", opcount
        opcount = opcount + 1
        for i in range(mat.shape[0]):
            print mat[i]
    sysarray = np.zeros([M, N], dtype = int)    
    tf = open(args.out_file, "w")
    genWtData(sysarray, wt_list, tf)
    print "Dumped dummy weights in <", args.out_file, ">"
    tf.close()

if __name__ == "__main__":
    main(sys.argv)

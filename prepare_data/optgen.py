#!/usr/bin/python3

'''
	OPT / Belady's algorithm for cache eviction.

        Hawkeye, Micro'16

	This algorithm simulates the OPT cache eviction policy.
	Input: block trace file containing a column with multiple different Blocks.
	Output: Cache Miss count and cache configuration.

	Description

	We first generate a dictionary of block numbers and their relative access sequence.
	We call this dictionary the OPT dictionary.

	[Block Number1] [Position1, Postion2 ...]
	[Block Number2] [Position1, Postion2 ...]
	[Block Number3] [Position1, Postion2 ...]

	We create an array of Cached blocks "C" which is empty in the beginning.

	for each block request:
		if the block is present in the C, remove the position from the OPT dictionary.
		hit_count++
		if the block is not present in C, find max_distance
			max_distance = max(OPT[Bnumber1][0], OPT[BNumber2][0], OPT[BNumber3][0])
			where {Bnumber1, Bnumber2...} E C
		remove max_distance block from C, remove position of requested block from OPT dictionary.
		miss_count++

	hit_count+miss_count = sizeof block_request list

'''
import torch
import numpy as np
from collections import defaultdict, Counter
from functools import partial
from tqdm import tqdm
import pandas as pd
from io import StringIO
import os
import argparse

block_trace = []

'''
get the furthest accessed block. Scans OPT dictionary and selects the maximum positioned element
'''

def getFurthestAccessBlock():
    global OPT
    global C
    maxAccessPosition = -1
    maxAccessBlock = -1
    for cached_block in C:
        if len(OPT[cached_block]) == 0:
            #print ( "Not Acccessing block anymore " + str(cached_block))
            return cached_block
        if OPT[cached_block][0] > maxAccessPosition:
            maxAccessPosition = OPT[cached_block][0]
            maxAccessBlock = cached_block
    #print ( "chose to evict " + str(maxAccessBlock) + " since its position of access is " + str(maxAccessPosition))
    return maxAccessBlock

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OPT.\n')
    parser.add_argument('cache_percent', type=float,  help='relative cache size, e.g., 0.2 stands for 20\% of total trace length\n')
    #parser.add_argument('ratio', type=float,  help='sample ratio\n')
    #parser.add_argument('idx', type=int,  help='column number of blocks. type 0 if only 1 column containing block trace is present\n')
    parser.add_argument('traceFile', type=str,  help='trace file name\n')
    args = parser.parse_args()

    cache_size = args.cache_percent
    #idx = args.idx
    #ratio = args.ratio
    traceFile = args.traceFile

    folder_name = traceFile[0:traceFile.rfind("/")] + f"_cache_{int(cache_size*100)}"

    print(folder_name)

    isExist = os.path.exists(folder_name)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(folder_name)
        print("The {folder_name} is created!")


    #sampled_trace = traceFile[0:traceFile.rfind(".pt")] + f"_sampled_{int(ratio*100)}.txt"
    sampled_trace = traceFile
    file = open(sampled_trace,mode='r')

    # read all lines at once
    all_of_it = file.read()

    # close the file
    file.close()
    d = StringIO(all_of_it)
    trace = np.loadtxt(d, dtype=float)
    block_trace = trace[:,1]

    #items = np.unique(block_trace)
    #print(f"num of unique indices is {len(items)}")
    #blockTraceLength = len(block_trace)
    #cache_size = int(cache_size * len(items))

    #block_trace =  block_trace.numpy()


    items = np.unique(block_trace)
    print(f"num of unique indices is {len(items)}")
    blockTraceLength = len(block_trace)
    cache_size = int(cache_size * len(items))

    print(f"created block trace list, cache size is {cache_size}")


    # build OPT

    OPT = defaultdict(list)


    seq_number = 0

    for b in tqdm(block_trace):
        #OPT[b] = np.append(OPT[b],seq_number)
        OPT[b].append(seq_number)
        seq_number+=1

    print ("created OPT dictionary")

    # run algorithm

    hit_count = 0
    miss_count = 0

    C = set()

    cached_trace = folder_name  + traceFile[traceFile.rfind("/"):traceFile.rfind(".txt")] + "_cached_trace_opt.txt"
    dataset_trace = folder_name + traceFile[traceFile.rfind("/"):traceFile.rfind(".txt")] + "_dataset_cache_miss_trace.txt"
    print(cached_trace)
    print(dataset_trace)

    f_hit = open(cached_trace,"w")
    f_miss = open(dataset_trace,"w")


    seq_number = 0
    for b in tqdm(block_trace):
        seq_number+=1
        if(seq_number % (blockTraceLength / 10) == 0):
            print("Completed "+str(( seq_number * 100 / blockTraceLength)) + " %")
        if b in C:
            #print ("HIT " + str(b))
            #np.delete(OPT[b],[0])
            #OPT[b] = OPT[b][1:]
            OPT[b] = np.delete(OPT[b],0)
            hit_count+=1
            #cache_hit[seq_number-1] = 1
            f_hit.write("1\n")
            f_miss.write("0\n")
        else:
            #print ("MISS " + str(b))
            miss_count+=1
            #cache_miss[seq_number-1] = b
            f_hit.write("0\n")
            f_miss.write(str(b)+'\n')
            if len(C) == cache_size:
                fblock = getFurthestAccessBlock()
                assert(fblock != -1)
                C.remove(fblock)
            C.add(b)
            #np.delete(OPT[b],[0])
            #OPT[b] = OPT[b][1:]
            OPT[b] = np.delete(OPT[b],0)
            #print ("CACHE ")
            #print (C)

    print ("hit count" + str(hit_count))
    print ("miss count" + str(miss_count))
    f_miss.close()
    f_hit.close()
    #cached_trace = traceFile[0:traceFile.rfind(".pt")] + "_cached_trace_opt.csv"
    #df = pd.DataFrame(cache_hit)
    #df.to_csv(cached_trace)

    #dataset_trace = traceFile[0:traceFile.rfind(".pt")] + "_dataset_cache_miss_trace.csv"
    #df = pd.DataFrame(cache_miss)
    #df.to_csv(dataset_trace)

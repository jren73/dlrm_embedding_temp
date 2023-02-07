from functools import lru_cache
import sys
import random
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from collections import Counter, deque, defaultdict
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import torch
from io import StringIO


def optcache(gt, blocktrace, frame):
    cache = set()
    recency = deque()
    hit, miss = 0, 0

    for idx in tqdm(range(len(blocktrace))):
        block = gt[idx]
        if block in cache:
            hit = hit+1
            if blocktrace[idx]==1:
                recency.remove(block)
                recency.append(block)
        else:
            miss = miss +1
            if blocktrace[idx]==1:
                if len(cache) < frame:
                    cache.add(block)
                    recency.append(block)
                else:
                    cache.remove(recency[0])
                    recency.popleft()
                    cache.add(block)
                    recency.append(block)

    hitrate = hit / (hit + miss)
    print(hitrate)
    print(hit)
    print(hit+miss)


def optcache_new(gt, blocktrace, frame):
    cache1 = set()
    recency1 = deque()
    cache2 = set()
    recency2 = deque()
    hit, miss = 0, 0

    for idx in tqdm(range(len(blocktrace))):
        block = gt[idx]
        stay = blocktrace[idx]
        if block in cache1:
            hit = hit+1
            recency1.remove(block)
            recency1.append(block)
        elif block in cache2:
            hit = hit+1
            recency2.remove(block)
            recency2.append(block)
        else:
            miss = miss +1
            if len(cache1)+len(cache2) < frame:
                if stay==1:
                    cache1.add(block)
                    recency1.append(block)
                else:
                    cache2.add(block)
                    recency2.append(block)
            else:
                if len(cache2)>0:
                    cache2.remove(recency2[0])
                    recency2.popleft()
                elif len(cache2)==0 and stay:
                    cache1.remove(recency1[0])
                    recency1.popleft()
                if stay:
                    cache1.add(block)
                    recency1.append(block)
                elif len(cache1)<frame-1:
                    cache2.add(block)
                    recency2.append(block)


    hitrate = hit / (hit + miss)
    print(hitrate)
    print(hit)
    print(hit+miss)


def optcache_fifo(gt, blocktrace, frame):
    cache = deque()
    hit, miss = 0, 0

    for idx in tqdm(range(len(blocktrace))):
        block = gt[idx]
        if block in cache:
            hit = hit+1

        else:
            miss = miss +1
            if blocktrace[idx]==1:
                if len(cache) < frame:
                    cache.append(block)

                else:
                    cache.popleft()
                    cache.append(block)

    hitrate = hit / (hit + miss)
    print(hitrate)
    print(hit)
    print(hit+miss)



if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='caching algorithm.\n')
        parser.add_argument('cache_percent', type=float,  help='relative cache size, e.g., 0.2 stands for 20\% of total trace length\n')
        #parser.add_argument('sample_ratio', type=float,  help='sample ratio\n')
        #parser.add_argument('idx', type=int,  help='column number of blocks. type 0 if only 1 column containing block trace is present\n')
        parser.add_argument('traceFile', type=str,  help='trace file name\n')
        parser.add_argument('gtFile', type=str,  help='trace file name\n')
        args = parser.parse_args()

        cache_size = args.cache_percent
        #idx = args.idx
        #ratio = args.sample_ratio
        traceFile = args.traceFile
        gtFile = args.gtFile


        #sampled_trace = traceFile[0:traceFile.rfind(".pt")] + f"_sampled_{int(ratio*100)}.txt"
        sampled_trace = traceFile
        file = open(sampled_trace,mode='r')


        # read all lines at once
        all_of_it = file.read()

        # close the file
        file.close()
        d = StringIO(all_of_it)
        trace = np.loadtxt(d, dtype=float)
        #block_trace = trace[:,1]
        block_trace = trace[:1000000]


        print("read gt")
        file = open(gtFile,mode='r')
        # read all lines at once
        all_of_it = file.read()

        # close the file
        file.close()
        d = StringIO(all_of_it)
        trace = np.loadtxt(d, dtype=float)
        gt_trace = trace[:len(block_trace),1]

        items = np.unique(gt_trace)
        print(f"num of unique indices is {len(items)}")
        cache_size = int(cache_size * len(items))
        print("processed!")

        optcache_new(gt_trace, block_trace,cache_size)

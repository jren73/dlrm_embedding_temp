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

def LFU(blocktrace, frame):
        cache = set()
        cache_frequency = defaultdict(int)
        frequency = defaultdict(int)
        
        hit, miss = 0, 0
        lfu = np.zeros(len(blocktrace))
        lfu_miss = np.zeros(len(blocktrace))

        i=0
        for block in tqdm(blocktrace, leave=False):
                frequency[block] += 1
                if block in cache:
                        hit += 1
                        cache_frequency[block] += 1
                        lfu[i] = 1
                
                elif len(cache) < frame:
                        cache.add(block)
                        cache_frequency[block] += 1
                        miss += 1
                        lfu_miss[i] = block

                else:
                        e, f = min(cache_frequency.items(), key=lambda a: a[1])
                        cache_frequency.pop(e)
                        cache.remove(e)
                        cache.add(block)
                        cache_frequency[block] = frequency[block]
                        miss += 1
                        lfu_miss[i] = block
                i = i+1
        
        hitrate = hit / ( hit + miss )
        print(hitrate)

        return lfu,lfu_miss

def LRU(blocktrace, frame):
        
        cache = set()
        recency = deque()
        hit, miss = 0, 0
        
        i=0
        for block in tqdm(blocktrace, leave=False):
                
                if block in cache:
                        recency.remove(block)
                        recency.append(block)
                        hit += 1
                      
                
                elif len(cache) < frame:
                        cache.add(block)
                        recency.append(block)
                        miss += 1
                       
                
                else:
                        cache.remove(recency[0])
                        recency.popleft()
                        cache.add(block)
                        recency.append(block)
                        miss += 1
                     
        i=i+1
        hitrate = hit / (hit + miss)
        print(hitrate)
        print(hit)
        print(hit+miss)
        return lru,lru_miss

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='caching algorithm.\n')
        parser.add_argument('cache_percent', type=float,  help='relative cache size, e.g., 0.2 stands for 20\% of total trace length\n')
        #parser.add_argument('sample_ratio', type=float,  help='sample ratio\n')
        #parser.add_argument('idx', type=int,  help='column number of blocks. type 0 if only 1 column containing block trace is present\n')
        parser.add_argument('traceFile', type=str,  help='trace file name\n')
        args = parser.parse_args() 

        cache_size = args.cache_percent
        #idx = args.idx
        #ratio = args.sample_ratio
        traceFile = args.traceFile
    
    
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
        block_trace = trace[:,1]
        items = np.unique(block_trace)
        print(f"num of unique indices is {len(items)}")
        cache_size = int(cache_size * len(items))
        print("processed!")

        # build LRU
        lru_cache,lru_miss = LRU(block_trace, cache_size)
        #lru_cache, lru_miss = LRU(block_tmp, cache_size)
        '''
        cached_trace = traceFile[0:traceFile.rfind(".pt")] + "_lru_cache.csv"
        df = pd.DataFrame(lru_cache)
        df.to_csv(cached_trace)
        cached_trace = traceFile[0:traceFile.rfind(".pt")] + "_lru_miss.csv"
        df = pd.DataFrame(lru_miss)
        df.to_csv(cached_trace)

        # build LFU
        lfu_cache, lfu_miss = LFU(block_trace, cache_size)
        #lfu_cache, lfu_miss = LFU(block_tmp, cache_size)
        cached_trace = traceFile[0:traceFile.rfind(".pt")] + "_lfu_cache.csv"
        df = pd.DataFrame(lfu_cache)
        df.to_csv(cached_trace)
        cached_trace = traceFile[0:traceFile.rfind(".pt")] + "_lfu_miss.csv"
        df = pd.DataFrame(lfu_miss)
        df.to_csv(cached_trace)
        '''

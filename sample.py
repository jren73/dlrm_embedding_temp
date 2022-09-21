import torch
import numpy as np
import random
import argparse
from tqdm import tqdm
random.seed(0)

def get_table_ID(lengths):
    rows_num=len(lengths)
    column_num=len(lengths[0])
    ID_list=[]
    for i in range(column_num):
        for j in range(rows_num):
            candi_num = lengths[j,i]
            for it in range(candi_num):
               ID_list.append(j)
    return np.array(ID_list)


# def dataset_sample2(lengths,offsets,indices,ratio):
#     new_lengths=[]
#     new_offsets=[0]
#     new_indices=[]
#     # columns=round(len(lengths[0])*ratio)
#     # # sample_col_ID=sorted(random.sample(range(len(lengths[0])),columns))
#     sample_col_ID = [x for x in range(len(lengths[0]))]
#     for row in range(len(lengths)):
#         new_row=[]
#         for col in sample_col_ID:
#             new_row.append(lengths[row,col])
#             new_offsets.append(row*65536+col+1)
#             indices_start=offsets[row*65536+col]
#             span=lengths[row,col]
#             for i in range(indices_start,indices_start+span):
#                 new_indices.append(indices[i])
#         new_lengths.append(new_row)
#     return np.array(new_lengths),np.array(new_offsets),np.array(new_indices)

def dataset_sample2(lengths,offsets,indices,ratio):
    new_lengths=[]
    new_offsets=[0]
    new_indices=[]
    columns=round(len(lengths[0])*ratio)
    sample_col_ID=sorted(random.sample(range(len(lengths[0])),columns))
    #sample_col_ID = [x for x in range(len(lengths[0]))]
    for row in tqdm(range(len(lengths))):
        new_row=[]
        for col in sample_col_ID:
            new_row.append(lengths[row,col])
            new_offsets.append(row*65536+col+1)
            indices_start=offsets[row*65536+col]
            span=lengths[row,col]
            for i in range(indices_start,indices_start+span):
                new_indices.append(indices[i])
        new_lengths.append(new_row)
    return np.array(new_lengths),np.array(new_offsets),np.array(new_indices)


def get_unique_id(lengths):
    uni=0
    new_lengths=[]
    for i in range(len(lengths)):
        if(lengths[i]==uni):
            new_lengths.append(uni);
        else:
            uni=lengths[i]
            new_lengths.append(uni)
    return np.array(new_lengths)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='sample.\n')
    parser.add_argument('sample_ratio', type=float,  help='relative cache size, e.g., 0.2 stands for 20\% of total trace length\n')
    parser.add_argument('traceFile', type=str,  help='trace file name\n')
    args = parser.parse_args() 
    
    ratio = args.sample_ratio
    traceFile = args.traceFile
    sampled_trace = traceFile[0:traceFile.rfind(".pt")] + f"_sampled_{int(ratio*100)}.txt"
    print(sampled_trace)
    #indices, offsets, lengths = torch.load("~/dlrm_datasets/embedding_bag/fbgemm_t856_bs65536_15.pt")
    
    indices, offsets, lengths = torch.load(traceFile)
    lengths,new_offsets,new_indices=dataset_sample2(lengths,offsets,indices,ratio)
    
    new_lengths=get_table_ID(lengths)

    n_lengths=get_unique_id(new_lengths)
    matrix = np.vstack((n_lengths, new_indices))
    matrix = matrix.T
    np.savetxt(sampled_trace, matrix, fmt='%d', delimiter=' ')
    

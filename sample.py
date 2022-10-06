import torch
import numpy as np
import random
import argparse
import os
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
    #sample_col_ID=sorted(random.sample(range(len(lengths[0])),columns))
    sample_col_ID=random.sample(range(len(lengths[0])),columns)
    print(sample_col_ID)
    print(len(np.unique(sample_col_ID)))
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
    #sampled_trace = traceFile[0:traceFile.rfind(".pt")] + f"_sampled_{int(ratio*100)}.txt"
    #print(sampled_trace)
    #indices, offsets, lengths = torch.load("~/dlrm_datasets/embedding_bag/fbgemm_t856_bs65536_15.pt")
    folder_name = traceFile[0:traceFile.rfind("/")]+ f"/sample_" + traceFile[traceFile.rfind("_")+1:traceFile.rfind(".pt")]
    print(folder_name)

    isExist = os.path.exists(folder_name)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(folder_name)
        print("The new directory is created!")
    indices, offsets, lengths = torch.load(traceFile)
    print(offsets[0], offsets[2], lengths[0])

    n_zeros = np.count_nonzero(indices==0)
    print(n_zeros)
    n_zeros = np.count_nonzero(offsets==0)
    print(n_zeros)
    n_zeros = np.count_nonzero(lengths==0)
    print(n_zeros)
    print(indices[offsets[656]])
    print(lengths[657,0])

    lengths,new_offsets,new_indices=dataset_sample2(lengths,offsets,indices,ratio)

    new_lengths=get_table_ID(lengths)

    n_lengths=get_unique_id(new_lengths)
    matrix = np.vstack((n_lengths, new_indices*1000+n_lengths))
    matrix = matrix.T
    print(len(matrix))

    idx=0
    for i in range(0,len(matrix), 5000000):
            sampled_trace = folder_name + f"/dataset_" + traceFile[traceFile.rfind("_")+1:traceFile.rfind(".pt")] + f"_sampled_{int(ratio*100)}_{idx}.txt"
            np.savetxt(sampled_trace, matrix[i:i+5000000,], fmt='%d', delimiter=' ')
            idx = idx+1
            print(sampled_trace)
    #np.savetxt(sampled_trace, matrix, fmt='%d', delimiter=' ')

import argparse
import torch
import json
import os
import editdistance
import matplotlib.pyplot as plt
import tqdm
import torch.nn.functional as F
from seq2seq_prefetching import seq2seq_prefetch
from seq2seq_caching import Seq2Seq_cache
from torch.utils.data import DataLoader
from utils import prepare_data, MyDataset_cache, MyDataset_prefetch
import pandas as pd
import numpy as np
import glob
from io import StringIO
import torch.nn as nn
from torch.autograd import Variable
import math

def initial():
    global history 
    global device 
    global processing_file
    global chk_path 

    history = dict(train=[], val=[])
    device = device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    processing_file = ""
    chk_path = 'models/cache_model.pt'

def grub_datafile(datafolder, inputsfolder, model_type=1):
    cache_trainingdata = datafolder+"/*cached_trace_opt.txt"
    prefetcher_trainingdata  = datafolder+"/*dataset_cache_miss_trace.txt"
    inputs = inputsfolder
    res = []
    if model_type==0:
        res = glob.glob(cache_trainingdata)
    else:
        res = glob.glob(prefetcher_trainingdata)

    inputsfile = [f for f in glob.glob(inputs+f"/*.txt")]
    print(res)
    print(inputsfile)
    assert(len(res) == len(inputsfile))
    return inputsfile, res

def init_weights(m):
    print("Initializing weights...")
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
'''
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
'''
def predict(x,y):
    acc = 0
    sigmoid = nn.Sigmoid()
    r = sigmoid(x)
    r =  (r>0.5).float()
    #need to update here!!!
    mask = r==y
    acc = torch.count_nonzero(mask).item()
    return acc/len(x)




def train_model(model, train_set,eval_set,seq_length, n_epochs):
    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_loss = 10000.0
    #mb = master_bar(range(1, n_epochs + 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 5e-3, eta_min=1e-8, last_epoch=-1)
    f = open("training_result.txt", "a")
    for epoch in range(n_epochs):
        model = model.train()

        train_losses = []
        index = np.arange(seq_length, len(train_set), seq_length)
        for i in index:
            data,j =  train_set[i]
            trainX = Variable(torch.Tensor(data)).to(device)
            trainy = Variable(torch.Tensor(j)).to(device)
            optimizer.zero_grad()
            y_pred = model(trainX, trainX[-1])
            #print(y_pred)
            #print(trainy.unsqueeze(1))
            loss = criterion(y_pred, trainy.unsqueeze(1))
            #print(y_pred)
            #print(trainy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            train_losses.append(loss.item())
     
            if i%100000 == 0:
                out_tmp = str(int(i/seq_length))+"th iteration loss: "+str(loss.item())+ ", avg loss: "+ str(np.mean(train_losses))
                f.write(out_tmp +"\n")
                print(out_tmp)
        val_losses = []
        val_accuracy = []
        model = model.eval()
        index = np.arange(seq_length, len(eval_set), seq_length)
        with torch.no_grad():
            for i in index:
                data,j =  eval_set[i]
                evalX = Variable(torch.Tensor(data)).to(device)
                evaly = Variable(torch.Tensor(j)).to(device)
                y_pred = model(evalX, evalX[-1])
                loss = criterion(y_pred, evaly.unsqueeze(1))
                val_losses.append(loss.item())
                accuracy = predict(y_pred, evaly.unsqueeze(1))
                val_accuracy.append(accuracy)

                if i%4000 == 0:
                    out_tmp = "Evaluation "+ str(int(i/seq_length))+ "th iteration - avg loss: "+ str(np.mean(val_losses))+ "avg accuracy: " + str(np.mean(val_accuracy))
                    f.write(out_tmp +"\n")
                    print(out_tmp)
                    

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print("Finish processing ", processing_file)
        scheduler.step(val_loss)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        f.write(str(train_loss)+ "\n")
        f.write(str(val_loss)+"\n")
        f.write("~~~~~~~~~~~~~~~~~~~~\n\n\n")
        f.close()
        return model.eval()

def run(traceFile, model_type):
    #USE_CUDA = 0
    config_path = FLAGS.config

    if not os.path.exists(config_path):
        raise FileNotFoundError

    with open(config_path, "r") as f:
        config = json.load(f)

    config["gpu"] = torch.cuda.is_available()
    input_sequence_length = config["n_channels"]
    evaluation_windown_length = config["evaluation_window"]
    BATCHSIZE = config["batch_size"]

    model = Seq2Seq_cache(input_sequence_length, 1, 512, input_sequence_length)
    model = model.to(device)
    print(model)

    if model_type == 1:
        model.load_state_dict(torch.load('models/cache_model.pt'))
    else:
        model.apply(init_weights)

    inputsfolder = traceFile
    datafolder = traceFile+"_cache_10"

    if not os.path.exists(inputsfolder):
        raise FileNotFoundError
    if not os.path.exists(datafolder):
        raise FileNotFoundError
    trace, res = grub_datafile(datafolder, inputsfolder, model_type)
    trace.sort()
    print(res)
    print(trace)
    
    
    input_trace = ""
    output_trace = ""
    for f in trace:
        output_trace = f
        index1 = f.find("sampled_")
        index2 = f.find(".txt")
        dataset_id = f[index1:index2]
        for ff in res:
            print("\n")
            if dataset_id in ff:
                input_trace = ff
        assert(input_trace != "")
        processing_file = output_trace
        print("Processing "+ input_trace +" and " +output_trace)
        print("================================================\n")
        file = open(input_trace,mode='r')

        # read all lines at once
        all_of_it = file.read()

        # close the file
        file.close()
        d = StringIO(all_of_it)
        trace = np.loadtxt(d, dtype=float)
        block_trace = trace[:]

        file = open(output_trace,mode='r')

        # read all lines at once
        all_of_it = file.read()

        # close the file
        file.close()
        d = StringIO(all_of_it)
        trace = np.loadtxt(d, dtype=float)
        gt_trace = trace[:len(block_trace),1]

        FLAGS.train_size = len(gt_trace)
        assert(len(gt_trace) == len(block_trace))

        if model_type==1:
            train_set = MyDataset_prefetch(gt_trace[:],block_trace[:],input_sequence_length,evaluation_windown_length) 
            train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=False, collate_fn=None, drop_last=True)
            
   
        else:
            data_len = len(gt_trace)
            #data_len = 2500
            data_boundary = int(data_len*0.8)
            train_set = MyDataset_cache(gt_trace[:data_boundary],block_trace[:data_boundary],input_sequence_length)
            eval_set = MyDataset_cache(gt_trace[data_boundary:data_len],block_trace[data_boundary:data_len],input_sequence_length)
            #print(train_set[20])
            #train_loader = DataLoader(train_set, batch_size=3, shuffle=False, collate_fn=None, drop_last=True)
            
            # Train
            print("==> Start training caching model ... with datafile " + output_trace)
            model = train_model(model, train_set, eval_set, input_sequence_length, 1)
    ## need to add loss figure draw
    ## need to make sure the chk has better quality?
    torch.save(model.state_dict(), 'cache_model.pt')
        
    
def inference(trace_file, model_type):
    inf_seq_length = 500
    print("Loading model...")
    model = Seq2Seq_cache(inf_seq_length, 1, 512, inf_seq_length)                        
    model = model.to(device)
    model.load_state_dict(torch.load('models/cache_model.pt'))
        
    model.eval()
    file = open(trace_file,mode='r')

    # read all lines at once
    all_of_it = file.read()

    # close the file
    file.close()
    d = StringIO(all_of_it)
    trace = np.loadtxt(d, dtype=float)
    block_trace = trace[:,1]
    inference_set = MyDataset_cache(block_trace[:],block_trace[:],inf_seq_length)
    print("inferening input ", trace_file)
    #f = open("inference_result.txt", "w")
    
    index = np.arange(inf_seq_length, len(inference_set), inf_seq_length)
    res=[]
    with torch.no_grad():    
        for i in index:
            data,j =  inference_set[i]
            evalX = Variable(torch.Tensor(data)).to(device)
            y_pred = model(evalX, evalX[-1])
            '''
            x = y_pred.cpu().numpy()
            for i in range(len(x)):
                t=0
                if sigmoid(x[i])>=0.5:
                    t=1
                f.write(str(t)+"\n")
            '''
            sigmoid = nn.Sigmoid()
            r = sigmoid(y_pred)
            r =  (r>0.5).float()
            res.append(r)
    #print(res)
    #f.close()
    torch.save(res,"inference.pt")
    print("inference restults are in inference_result.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--traceFile', type=str,  help='trace file name\n')
    parser.add_argument('--model_type', default=0, type=int,  help='0 for training from stratch, 1 for loading saved model\n')    
    parser.add_argument('--infFile', type=str,  help='inference file name\n')

    FLAGS, _ = parser.parse_known_args()
    traceFile = FLAGS.traceFile
    model_type = FLAGS.model_type
    inferenceFile = FLAGS.infFile
    initial()

    if model_type == 1:
        check_file = os.path.isfile(chk_path)
        if check_file is False:
                print("Can not load caching model from checkpoint.")
                model_type = 0

    model = "stratch" if model_type==0 else "checkpoint"
    print("training caching model with " + traceFile + " from " + model)

    run(traceFile, model_type)
    #inference(inferenceFile, model_type)

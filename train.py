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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("device is:",device)
processing_file=""
history = dict(train=[], val=[])

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
def predict(x,y):
    acc = 0
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    assert(len(x) == len(y))
    for i in range(len(x)):
        t=0
        if x[i]>=0.5:
            t=1
        if t == y[i]:
            acc = acc+1
    return acc



def train_model(model, train_set,eval_set,seq_length, n_epochs):
    #

    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_loss = 10000.0
    #mb = master_bar(range(1, n_epochs + 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 5e-3, eta_min=1e-8, last_epoch=-1)
    
    for epoch in range(n_epochs):
        model = model.train()

        train_losses = []
        np.arange(seq_length, len(train_set), seq_length)
        for i in range(len(train_set)):
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
     
            if i%seq_length == 0:
                print(i,"th iteration avg loss: ",np.mean(train_losses))
        
        val_losses = []
        val_accuracy = []
        model = model.eval()
        with torch.no_grad():
            for i in range(len(eval_set)):
                data,j =  eval_set[i]
                evalX = Variable(torch.Tensor(data)).to(device)
                evaly = Variable(torch.Tensor(j)).to(device)
                y_pred = model(evalX, evalX[-1])
                loss = criterion(y_pred, evaly.unsqueeze(1))
                val_losses.append(loss.item())
                accuracy = predict(y_pred, evaly.unsqueeze(1))
                val_accuracy.append(accuracy)

                if i%1 == 0:
                    print("Evaluation - avg loss: ",np.mean(val_losses), "avg accuracy: ", np.mean(val_accuracy)/seq_length)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print("Finish processing ", processing_file)
        scheduler.step(val_loss)

        return model.eval()

def train(model, optimizer, train_loader, state):
    epoch, n_epochs, train_steps = state

    losses = []
    cers = []

    # t = tqdm.tqdm(total=min(len(train_loader), train_steps))
    t = tqdm.tqdm(train_loader)
    model.train()

    for batch in t:
        t.set_description("Epoch {:.0f}/{:.0f} (train={})".format(epoch, n_epochs, model.training))
        loss, _, _, _ = model.loss(batch)
        losses.append(loss.item())
        # Reset gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()
        t.set_postfix(loss='{:05.3f}'.format(loss.item()), avg_loss='{:05.3f}'.format(np.mean(losses)))
        t.update()

    return model, optimizer
    # print(" End of training:  loss={:05.3f} , cer={:03.1f}".format(np.mean(losses), np.mean(cers)*100))


'''
def evaluate(model, eval_loader):

    losses = []
    accs = []

    t = tqdm.tqdm(eval_loader)
    model.eval()

    with torch.no_grad():
        for batch in t:
            t.set_description(" Evaluating... (train={})".format(model.training))
            loss, logits, labels, alignments = model.loss(batch)
            preds = logits.detach().cpu().numpy()
            # acc = np.sum(np.argmax(preds, -1) == labels.detach().cpu().numpy()) / len(preds)
            acc = 100 * editdistance.eval(np.argmax(preds, -1), labels.detach().cpu().numpy()) / len(preds)
            losses.append(loss.item())
            accs.append(acc)
            t.set_postfix(avg_acc='{:05.3f}'.format(np.mean(accs)), avg_loss='{:05.3f}'.format(np.mean(losses)))
            t.update()
        align = alignments.detach().cpu().numpy()[:, :, 0]

    # Uncomment if you want to visualise weights
    # fig, ax = plt.subplots(1, 1)
    # ax.pcolormesh(align)
    # fig.savefig("data/att.png")
    print("  End of evaluation : loss {:05.3f} , acc {:03.1f}".format(np.mean(losses), np.mean(accs)))
    # return {'loss': np.mean(losses), 'cer': np.mean(accs)*100}
'''

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

    if model_type == 1:
        model = seq2seq_prefetch(config)
    else:
        model = Seq2Seq_cache(input_sequence_length, 1, 512, input_sequence_length)
                               
        model = model.to(device)
        print(model)
        model.apply(init_weights)

    inputsfolder = traceFile
    datafolder = traceFile+"_cache_10"

    if not os.path.exists(inputsfolder):
        raise FileNotFoundError
    if not os.path.exists(datafolder):
        raise FileNotFoundError
    trace, res = grub_datafile(datafolder, inputsfolder, model_type)
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
            #data_len = len(gt_trace)
            data_len = 1000
            data_boundary = int(data_len*0.8)
            train_set = MyDataset_cache(gt_trace[:data_boundary],block_trace[:data_boundary],input_sequence_length)
            eval_set = MyDataset_cache(gt_trace[data_boundary:data_len],block_trace[data_boundary:data_len],input_sequence_length)
            #print(train_set[20])
            #train_loader = DataLoader(train_set, batch_size=3, shuffle=False, collate_fn=None, drop_last=True)
            
            # Train
            print("==> Start training caching model ... with datafile " + output_trace)
            model = train_model(model, train_set, eval_set, input_sequence_length, 1)
    

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--traceFile', type=str,  help='trace file name\n')
    parser.add_argument('--model_type', default=1, type=int,  help='0 for caching model, 1 for prefetcing model\n')
    parser.add_argument('--gpu_id', default=0, type=int,  help='idicate which gpu used for training\n')
    #parser.add_argument('--epochs', default=1200, type=int)
    #parser.add_argument('--train_size', default=4000000, type=int)
    #parser.add_argument('--eval_size', default=2600, type=int)
    #args = parser.parse_args() 

    FLAGS, _ = parser.parse_known_args()
    traceFile = FLAGS.traceFile
    model_type = FLAGS.model_type
    gpu_id = FLAGS.model_type
    model = "cache" if model_type==0 else "prefetch"

    print("training " + model + " model with " + traceFile)

    run(traceFile, model_type)

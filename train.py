import argparse
import torch
import json
import os
import editdistance
import matplotlib.pyplot as plt
import tqdm
import torch.nn.functional as F
from seq2seq_prefetching import seq2seq_prefetch
from seq2seq_caching import seq2seq_cache
from torch.utils.data import DataLoader
from utils import prepare_data, MyDataset_cache, MyDataset_prefetch
import pandas as pd
import numpy as np
import glob
from io import StringIO


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


def run(traceFile, model_type):
    USE_CUDA = torch.cuda.is_available()

    config_path = FLAGS.config

    if not os.path.exists(config_path):
        raise FileNotFoundError

    with open(config_path, "r") as f:
        config = json.load(f)
    config["gpu"] = torch.cuda.is_available()
    input_sequence_length = config["n_channels"]
    evaluation_windown_length = config["evaluation_window"]



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

        
        assert(len(gt_trace) == len(block_trace))

        if model_type==1:
            train_set = MyDataset_prefetch(gt_trace[:],block_trace[:],input_sequence_length,evaluation_windown_length)
            model = seq2seq_prefetch(config, train_set)
        else:
            train_set = MyDataset_cache(gt_trace[:],block_trace[:],input_sequence_length,evaluation_windown_length)
            model = seq2seq_cache(config, train_set)
        # Train
        print("==> Start training ...")
        model.git()

        # Prediction
        y_pred = model.test()

        
        if USE_CUDA:
            model = model.cuda()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", .001))

        print("=" * 60)
        print(model)
        print("=" * 60)
        for k, v in sorted(config.items(), key=lambda i: i[0]):
            print(" (" + k + ") : " + str(v))
        print()
        print("=" * 60)

        print("\nInitializing weights...")
        for name, param in model.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_normal_(param)

        for epoch in range(FLAGS.epochs):
            run_state = (epoch, FLAGS.epochs, FLAGS.train_size)

            # Train needs to return model and optimizer, otherwise the model keeps restarting from zero at every epoch
            model, optimizer = train(model, optimizer, train_loader, run_state)
            #evaluate(model, eval_loader)


    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--traceFile', type=str,  help='trace file name\n')
    parser.add_argument('--model_type', default=1, type=int,  help='0 for caching model, 1 for prefetcing model\n')
    #parser.add_argument('--epochs', default=1200, type=int)
    #parser.add_argument('--train_size', default=4000000, type=int)
    #parser.add_argument('--eval_size', default=2600, type=int)
    #args = parser.parse_args() 

    FLAGS, _ = parser.parse_known_args()
    traceFile = FLAGS.traceFile
    model_type = FLAGS.model_type
    model = "cache" if model_type==0 else "prefetch"

    print("training " + model + " model with " + traceFile)

    run(traceFile, model_type)

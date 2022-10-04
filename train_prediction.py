import argparse
import torch
import json
import os
import editdistance
import matplotlib.pyplot as plt
import tqdm
import torch.nn.functional as F
from Seq2Seq_prefecthing import Seq2Seq
from torch.utils.data import DataLoader
from utils import prepare_data, MyDataset
import pandas as pd
import numpy as np




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

def run(mydataset, gt):
    USE_CUDA = torch.cuda.is_available()

    config_path = FLAGS.config

    if not os.path.exists(config_path):
        raise FileNotFoundError

    with open(config_path, "r") as f:
        config = json.load(f)

    config["gpu"] = torch.cuda.is_available()
    input_sequence_length = config["n_channels"]
    evaluation_windown_length = config["evaluation_window"]
    
    
    train_set = MyDataset(mydataset[:],gt[:],input_sequence_length,evaluation_windown_length)
    #eval_dataset = MyDataset(mydataset[FLAGS.train_size:len(gt)],gt[FLAGS.train_size:len(gt)],input_sequence_length,evaluation_windown_length)

      
    
    #train_set = ToyDataset(5, 15)
    #eval_dataset = ToyDataset(5, 15, type='eval')
    #train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=False, collate_fn=None, drop_last=True)
    #eval_loader = DataLoader(eval_dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=None,
    #                              drop_last=True)
    '''
    t = tqdm.tqdm(train_loader)
    idx = 0
    for batch in t:
        print(idx)
        print(batch)
        idx = idx+1
        if idx==1:
            break

    return
    '''
    #print(train_loader.shape)
    #print(eval_loader.shape)
    # Models
    #model = Seq2Seq(config)

    model = Seq2Seq(config,
        train_set
    )

    # Train
    print("==> Start training ...")
    model.train()

    # Prediction
    y_pred = model.test()

    '''
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

        # TODO implement save models function
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--sample_ratio', default=0.02, type=int)
    parser.add_argument('--traceFile', type=str,  help='trace file name\n')
    #parser.add_argument('--epochs', default=1200, type=int)
    #parser.add_argument('--train_size', default=4000000, type=int)
    #parser.add_argument('--eval_size', default=2600, type=int)
    #args = parser.parse_args() 

    FLAGS, _ = parser.parse_known_args()
    traceFile = FLAGS.traceFile
    sample_ratio = FLAGS.sample_ratio
    dataset, gt = prepare_data(traceFile,sample_ratio,1) 
    
    #ensure the training and groudtruth has the same size. When we processing groundtruth, we cutout some data
    dataset = dataset[:len(gt)]
    FLAGS.train_size = int(len(dataset)*0.8)
    FLAGS.eval_size = len(gt) - FLAGS.train_size

    run(dataset, gt)

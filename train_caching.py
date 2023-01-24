import random
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle as pkl
from tensorflow import keras 
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, \
    multiply, concatenate, Flatten, Activation, dot
from tensorflow.keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
import pydot as pyd
from tensorflow.python.keras.utils.vis_utils import plot_model, model_to_dot
import torch
import collections 
import argparse
import pandas as pd
from utils import get_logger

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
_logger = get_logger(__name__)

#processing data
def data(input_file):
        indices, offsets, lengths = torch.load(input_file)
        print(f"Data file indices = {indices.size()}", f"offsets = {offsets.size()}, lengths = {lengths.size()}) ")
        return indices

def truncate(x, feature_cols=range(3), target_cols=range(3), label_col=3, train_len=100, test_len=20):
        in_, out_, lbl = [], [], []
        for i in range(len(x)-train_len-test_len+1):
                in_.append(x[i:(i+train_len), feature_cols].tolist())
                out_.append(x[(i+train_len):(i+train_len+test_len), target_cols].tolist())
                lbl.append(x[i+train_len, label_col])
        return np.array(in_), np.array(out_), np.array(lbl)

def merge(list1, list2):
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

def build(N, M, X_input_train, X_output_train):
        n_hidden = N

        #input layer
        input_train = Input(shape=(X_input_train.shape[1], X_input_train.shape[2]-1))
        output_train = Input(shape=(X_output_train.shape[1], X_output_train.shape[2]-1))

        # encoder
        encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
        n_hidden, activation='sigmoid', dropout=0.2, recurrent_dropout=0.2, 
        return_state=True, return_sequences=True)(input_train)
        print(encoder_stack_h)
        print(encoder_last_h)
        print(encoder_last_c)

        #batch_norm
        encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
        encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

        #decoder
        decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
        print(decoder_input)

        decoder_stack_h = LSTM(n_hidden, activation='sigmoid', dropout=0.2, recurrent_dropout=0.2,
        return_state=False, return_sequences=True)(
        decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        print(decoder_stack_h)

        #attention layer: Luong attention
        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
        attention = Activation('softmax')(attention)
        print(attention)

        context = dot([attention, encoder_stack_h], axes=[2,1])
        context = BatchNormalization(momentum=0.6)(context)
        print(context)
        decoder_combined_context = concatenate([context, decoder_stack_h])
        print(decoder_combined_context)
        out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)
        print(out)

        model = Model(inputs=input_train, outputs=out)
        opt = Adam(lr=0.001, clipnorm=1)
        #model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=opt, metrics=[keras.metrics.CategoricalAccuracy()])
        #model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


        model.summary()
        #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        return model



def test(model, X_input_train, X_input_test, X_output_train, X_output_test, x_train_max):
        train_pred_detrend = model.predict(X_input_train[:, :, :2])*x_train_max[:2]
        test_pred_detrend = model.predict(X_input_test[:, :, :2])*x_train_max[:2]
        print(train_pred_detrend.shape, test_pred_detrend.shape)
        train_true_detrend = X_output_train[:, :, :2]*x_train_max[:2]
        test_true_detrend = X_output_test[:, :, :2]*x_train_max[:2]
        print(train_true_detrend.shape, test_true_detrend.shape)

        train_pred_detrend = np.concatenate([train_pred_detrend, np.expand_dims(X_output_train[:, :, 2], axis=2)], axis=2)
        test_pred_detrend = np.concatenate([test_pred_detrend, np.expand_dims(X_output_test[:, :, 2], axis=2)], axis=2)
        print(train_pred_detrend.shape, test_pred_detrend.shape)
        train_true_detrend = np.concatenate([train_true_detrend, np.expand_dims(X_output_train[:, :, 2], axis=2)], axis=2)
        test_true_detrend = np.concatenate([test_true_detrend, np.expand_dims(X_output_test[:, :, 2], axis=2)], axis=2)
        print(train_pred_detrend.shape, test_pred_detrend.shape)
        print(test_true_detrend)
        print(test_pred_detrend)

        return train_pred_detrend, train_true_detrend, test_pred_detrend, test_true_detrend


def main():
        parser = argparse.ArgumentParser(description='caching model.\n')
        parser.add_argument('traceFile', type=str,  help='trace file name\n')
        parser.add_argument('n', type=int,  help='input sequence length N\n')
        parser.add_argument('m', type=int,  help='output sequence length\n')
        args = parser.parse_args() 

        traceFile = args.traceFile
        M = args.m
        N = args.n
        gt_trace = traceFile[0:traceFile.rfind(".pt")] + "_cached_trace_opt.txt"

        #dataset = data("dlrm_datasets/embedding_bag/fbgemm_t856_bs65536_9.pt")
        dataset = data(traceFile)
        #csvdata = pd.read_csv(gt_trace)
        #gt = csvdata[1].tolist()
        gt_file = open(gt_trace, "r")
        gt_tmp = gt_file.readlines()
        gt =  [float(x) for x in gt_tmp]
        #ensure the training and groudtruth has the same size. When we processing groundtruth, we cutout some data
        #gt = gt[:50000]
        #gt = gt[:10000000]
        dataset = dataset[:len(gt)]
         #input sequence length
        #N = 150
        #output sequence length
        #M = 10
        # evalutaion window size
        #W = 150

        '''
        plt.figure(figsize=(50, 4))
        plt.plot(range(len(dataset)), dataset, label='dataset')
        #plt.plot(range(len(x1_trend)), x1_trend, linestyle='--', label='x1_trend')
        plt.plot(range(len(gt)), gt, label='gt')
        #plt.plot(range(len(x2_trend)), x2_trend, linestyle='--', label='x2_trend')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
        #plt.show()
        plt.savefig('example.png')
        '''
      
        x_index = np.array(range(len(gt)))
        train_ratio = 0.8
        train_len = int(train_ratio * len(gt))
        print(train_len)

        x_lbl = np.column_stack([dataset, gt, x_index, [1]*train_len+[0]*(len(x_index)-train_len)])
        print(x_lbl.shape)
        print(x_lbl)

        '''
        plt.figure(figsize=(50, 4))
        plt.plot(range(train_len), x_lbl[:train_len, 0], label='x1_train')
        plt.plot(range(train_len), x_lbl[:train_len, 1], label='x2_train')
        plt.plot(range(train_len, len(x_lbl)), x_lbl[train_len:, 0], label='x1')
        plt.plot(range(train_len, len(x_lbl)), x_lbl[train_len:, 1], label='x2')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
        plt.savefig('example1.png')
        '''

        x_train_max = x_lbl[x_lbl[:, 3]==1, :2].max(axis=0)
        x_train_max = x_train_max.tolist()+[1]*2  # only normalize for the first 2 columns
        print(x_train_max)
        x_normalize = np.divide(x_lbl, x_train_max)
        print(x_normalize)

        '''
        plt.figure(figsize=(50, 4))
        plt.plot(range(train_len), x_normalize[:train_len, 0], label='x1_train_normalized')
        plt.plot(range(train_len), x_normalize[:train_len, 1], label='x2_train_normalized')
        plt.plot(range(train_len, len(x_normalize)), x_normalize[train_len:, 0], label='x1_test_normalized')
        plt.plot(range(train_len, len(x_normalize)), x_normalize[train_len:, 1], label='x2_test_normalized')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
        plt.savefig('example2.png')
        '''
        
        X_in, X_out, lbl = truncate(x_normalize, feature_cols=range(3), target_cols=range(3), 
                            label_col=3, train_len=N, test_len=M)
        print(X_in.shape, X_out.shape, lbl.shape)
        
        X_input_train = X_in[np.where(lbl==1)]
        X_output_train = X_out[np.where(lbl==1)]
        X_input_test = X_in[np.where(lbl==0)]
        X_output_test = X_out[np.where(lbl==0)]
        
        print(X_input_train.shape, X_output_train.shape)
        print(X_input_test.shape, X_output_test.shape)
        print("Finish data loadding")
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,:-1], dataset[:,-1].astype(int), test_size=0.2, random_state=None, shuffle=True)
        '''
       
        
        model = build(N, M, X_input_train, X_output_train)
       
        
        epc=10
        es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
        history = model.fit(X_input_train[:, :, :2], X_output_train[:, :, :2], validation_split=0.2, 
                        epochs=epc, verbose=1, callbacks=[es], 
                        batch_size=50000)
        #train_acc = history.history['categorical_accuracy']
        #valid_acc = history.history['val_categorical_accuracy']
        
        train_acc = history.history['accuracy']
        valid_acc = history.history['val_accuracy']

        
        model.save('model_caching_seq2seq.h5')
        plt.figure(figsize=(15, 4))
        plt.plot(train_acc, label='train acc'), 
        plt.plot(valid_acc, label='validation acc')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.title('train vs. validation accuracy')
        plt.legend( bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
        plt.savefig('training_acc_curve.png')

        plt.figure(figsize=(15, 4))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.savefig('training_loss_curve.png')
        
        #model = keras.models.load_model('model_caching_seq2seq.h5')
        
        print("Evaluate on test data")
        
        results = model.evaluate(np.delete(X_input_test, 2, axis=2), np.delete(X_output_test, 2, axis=2), batch_size=4)
        print("test loss, test acc:", results)

        
        
        #train_pred_detrend, train_true_detrend, test_pred_detrend, test_true_detrend = test(model, X_input_train, X_input_test, X_output_train, X_output_test,x_train_max)
        
        
        

if __name__ == "__main__":
    main()

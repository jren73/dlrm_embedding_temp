# dlrm_embedding_temp
    .
    ├── models                                               # model chk (train cache model and prefetch model pt file)
    ├── results                                              # training process intermediate result
    ├── datasets                                             # processed sythtic dataset
    ├── prepare_data                                         # Generate dataset from scratch
    ├── simulator                                            # simulate different caching behavior
    ├── run.sh                                               # train the model
    └── README.md

## Train cache model from scratch
 ./run.sh -c

## Train cache model from last cache model checkpoint
 ./run.sh -c -e

## Train prefetch model from scratch
 ./run.sh -p

## Train prefetch model from last cache model checkpoint
 ./run.sh -p -e

## Generate dataset from scratch

- sample.py is used to shuffle data based on batch and stores in 5M indices files.
  command `python3 sample.py 0.8 embedding_bag/fbgemm_t856_bs65536_x.pt` will genetate several files in folder sample_x. The sample rate is configrable( In this cacse is 0.8, which means 80% of 65536 bacthes in sythetic dataset) 
  
- optgen.py is used to generate training data for cache model and prefetch model
  command `python optgen.py 0.1 embedding_bag/sample_X/dataset_x_sampled_80_N` will generate several files in forder sample_x_cached_10. The cache size of Belady's algorithm is configurable. (In this case is 0.1, which means 10% of unique indices number is configed as cache size )
  For fast data generation, we can use `bash optgen.sh` to generate tens of training data files simultaneously. 
  
## ~~Cache model~~
To train cache model, simplyly use command `python3 train.py --config=example_caching.json --traceFile=dataset/sample_0 --model_type=0 --infFile=dataset/dataset_0_sampled_80_4.txt` with example dataset. We use CorssEntropyLoss for binary classification

Change # of indices for training: line 258 and 259
change report frequency: line 92 and 99


## ~~Prefetch model~~
To train prefetch model, use command `python3 train.py --config=example_prefetching.json --traceFile=dataset/sample_0 --model_type=1` with example dataset. We use customized Chamfer1DLoss in iou_loss.py


## Example dataset
A processed data can be found in https://drive.google.com/drive/folders/1S-oFOWEYzDTpkXgxXnDnu2BaiwfzgHOc?usp=share_link

To train the prefetching model, please run with `python3 train_prediction.py --config example_prefetching.json --traceFile fbgemm_t856_bs65536_15.pt`
Please download the sampled dataset *_sample.txt, cached_trace_opt.txt and *_cache_miss_trace.txt from google drive into the folder. We do not need to download fbgemm_t856_bs65536_15.pt in this training (We train with rewrited sampled data). 



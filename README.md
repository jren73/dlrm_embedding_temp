# dlrm_embedding_temp
dataset sampled with sampling rate as 0.1 can be found in https://drive.google.com/drive/folders/140HGV4TZ2IPK1dK2BdYrreCPFeVlmaGq?usp=sharing

To train the prefetching model, please run with `python3 train_prediction.py --config example_prefetching.json --traceFile fbgemm_t856_bs65536_15.pt`
Please download the sampled dataset *_sample.txt, cached_trace_opt.txt and *_cache_miss_trace.txt from google drive into the folder. We do not need to download fbgemm_t856_bs65536_15.pt in this training (We train with rewrited sampled data). 



import numpy as np
import torch
from random import randrange
import nltk.tokenize
import codecs
import logging
from torch.utils import data
from io import StringIO
_tokenizer = nltk.tokenize.RegexpTokenizer(pattern=r'[\w\$]+|[^\w\s]')


def get_logger(file_name):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(file_name)

    return logger


def get_formatted_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    formatted_time = '%d:%02d:%02d' % (h, m, s)

    return formatted_time


def tokenize(text):
    tokens = _tokenizer.tokenize(text.lower())
    return tokens


class IterableSentences(object):
    def __init__(self, filename):
        self._filename = filename

    def __iter__(self):
        for line in codecs.open(self._filename, 'r', 'utf-8'):
            yield line.strip()

def batch(iterable, n=1):
    args = [iter(iterable)] * n
    return zip_longest(*args)


def pad_tensor(vec, pad, value=0, dim=0):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = pad - vec.shape[0]

    if len(vec.shape) == 2:
        zeros = torch.ones((pad_size, vec.shape[-1])) * value
    elif len(vec.shape) == 1:
        zeros = torch.ones((pad_size,)) * value
    else:
        raise NotImplementedError
    return torch.cat([torch.Tensor(vec), zeros], dim=dim)


def pad_collate(batch, values=(0, 0), dim=0):
    """
    args:
        batch - list of (tensor, label)
    reutrn:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
        ws - a tensor of sequence lengths
    """
    
    #print(len(batch))
    ''''''
    sequence_lengths = torch.Tensor([int(x[0].shape[dim]) for x in batch])
    sequence_lengths, xids = sequence_lengths.sort(descending=True)
    target_lengths = torch.Tensor([int(x[1].shape[dim]) for x in batch])
    target_lengths, yids = target_lengths.sort(descending=True)
    # find longest sequence
    src_max_len = max(map(lambda x: x[0].shape[dim], batch))
    tgt_max_len = max(map(lambda x: x[1].shape[dim], batch))
    # pad according to max_len
    batch = [(pad_tensor(x, pad=src_max_len, dim=dim), pad_tensor(y, pad=tgt_max_len, dim=dim)) for (x, y) in batch]

    # stack all
    xs = torch.stack([x[0] for x in batch], dim=0)
    ys = torch.stack([x[1] for x in batch]).int()
    xs = xs[xids]
    ys = ys[yids]
    
    return xs, ys, sequence_lengths.int(), target_lengths.int()
    '''
    print(len(batch))
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xs = torch.stack([x for x in torch.tensor(xx)])
    ys = torch.stack([x for x in torch.tensor(yy)])
    #xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_id)
    #yy_pad = pad_sequence(yy, batch_first=True, padding_value=dict_pos['igner'])
    
    return xs, ys, x_lens, y_lens
    '''

def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    _, labels, lengths = zip(*data)
    max_len = max(lengths)
    n_ftrs = data[0][0].size(1)
    features = torch.zeros((len(data), max_len, n_ftrs))
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j, k = data[i][0].size(0), data[i][0].size(1)
        features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

    return features.float(), labels.long(), lengths.long()


#processing data: chunkization, chunksize is 1
def prepare_data(traceFile, ratio=0.1, model_type=1):
        sampled_trace = traceFile[0:traceFile.rfind(".pt")] + f"_sampled_{int(ratio*100)}.txt"

        file = open(sampled_trace,mode='r')

        # read all lines at once
        all_of_it = file.read()

        # close the file
        file.close()
        d1 = StringIO(all_of_it)
        trace = np.loadtxt(d1, dtype=float)
        dataset = trace[:,1]
        data_trace = ""
        # caching model
        if model_type == 0:
                data_trace = traceFile[0:traceFile.rfind(".pt")] + f"_cached_trace_opt.txt"
        elif model_type == 1:
                data_trace = traceFile[0:traceFile.rfind(".pt")] + f"_dataset_cache_miss_trace.txt"
        file = open(data_trace,mode='r')
        all_of_it = file.read()
        file.close()
        d2 = StringIO(all_of_it)
        gt = np.loadtxt(d2, dtype=float)

        return dataset,gt
        

class MyDataset(data.Dataset):
    """
    https://talbaumel.github.io/blog/attention/
    """
    def __init__(self,  sample, groundtruth, input_sequence=10,evaluation_window=10, min_length=19, max_length=20, type='train'):
        self.SOS = "<s>"  # all strings will end with the End Of String token
        self.EOS = "</s>"  # all strings will end with the End Of String token
        self.characters = list("abcd")
        self.int2char = list(self.characters)
        self.char2int = {c: i+3 for i, c in enumerate(self.characters)}
        #print(self.char2int)
        self.VOCAB_SIZE = len(self.characters)
        self.min_length = min_length
        self.max_length = max_length
        if type == 'train':
            self.set = [self._sample() for _ in range(3000)]
        else:
            self.set = [self._sample() for _ in range(300)]

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):
        return self.set[item]

    def _sample(self):
        random_length = randrange(self.min_length, self.max_length)  # Pick a random length
        random_char_list = [np.random.choice(self.characters[:-1]) for _ in range(random_length)]  # Pick random chars
        random_string = ''.join(random_char_list)
        a = np.array([self.char2int.get(x) for x in random_string])
        b = np.array([self.char2int.get(x) for x in random_string[::-1]] + [2]) # Return the random string and its reverse
        x = np.zeros((random_length, self.VOCAB_SIZE))

        x[np.arange(random_length), a-3] = 1

        return x, b

class MyDataset_prefetch(data.Dataset):

    def __init__(self, sample, groundtruth, input_sequence=10,evaluation_window=10):
        self.sample = sample
        self.groundtruth = groundtruth
        self.max = max(np.max(sample),np.max(groundtruth))
        self.set = [self._sample(idx,input_sequence,evaluation_window) for idx in range(0, len(self.sample), input_sequence)]
        #self.set = [self._test(idx,input_sequence,evaluation_window) for idx in range(0, len(self.sample), input_sequence)]
        

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):
        return self.set[item]

    def _sample(self, idx, sample_length, gt_length):
        #data_X = torch.tensor(self.sample[idx])
        #target_Y = torch.tensor(self.groundtruth[idx])
        #print(idx)
        if len(self.sample) < idx+sample_length:
            return
        x = self.sample[idx:idx+sample_length]
        
        y = self.groundtruth[idx+sample_length:min(idx+sample_length+gt_length, len(self.groundtruth))]
        if len(y)<gt_length:
            y = np.append(y, torch.zeros(gt_length-len(y)))
        x = np.array(x,dtype='float64')/500000
        y = np.array(y,dtype='float64')/500000
        return x,y

    def _test(self, offset, sample_length, gt_length):
        x, y = [], []
        #offset = 0
        y = self.groundtruth[offset:gt_length+sample_length+offset]

        for i in range(sample_length):
            if offset < len(self.sample)-sample_length-gt_length:
                x.append(self.sample[offset:sample_length+offset])
                offset += 1
        
        x = np.array(x,dtype='f')/self.max
        y = np.array(y,dtype='f')/self.max
        
        a = x.shape
        if(a[0]<sample_length):
            return [],[]
        return x,y


class MyDataset_cache(data.Dataset):

    def __init__(self, sample, groundtruth, seq_length=20):
        mean = sample.mean()
        stdev = sample.std()

        self.sample = torch.tensor((sample - mean) / stdev).float()
        self.groundtruth = torch.tensor(groundtruth).float()
        self.max = max(np.max(sample),np.max(groundtruth))
        assert(len(sample) == len(groundtruth))
        self.set = [self.sample, self.groundtruth]
        self.sequence_length = seq_length
        
        

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, item):
        if item >= self.sequence_length - 1:
            i_start = item - self.sequence_length + 1
            x = self.sample[i_start:(item + 1)]
            y = self.groundtruth[i_start:(item + 1)]
        else:
            padding = torch.zeros(self.sequence_length - item - 1)
            x = self.sample[0:(item + 1)]
            x = torch.cat((padding, x), 0)
            y = self.groundtruth[0:(item + 1)]
            y = torch.cat((padding, y), 0)

        return x, y

  


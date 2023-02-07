
import torch
import numpy as np 
import torch.nn as nn
import sys


### todo meaningless in python version. Add accuracy count here.

# PyTroch version
class iou_accuracy(nn.Module):
    def __init__(self):
        super(iou_accuracy, self).__init__()
        self.SMOOTH = 1e-6

    def iou_pytorch(self, outputs: torch.Tensor, labels: torch.Tensor):
        # You can comment out this line if you are passing tensors of equal shape
        # But if you are passing output from UNet or something it will most probably
        # be with the BATCH x 1 x H x W shape
        outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
        results = outputs.reshape(-1).detach().numpy()
        #results = np.append(results, results*2) scale things back
        results = np.unique(results)
        results = results.astype(int)
        label = labels.reshape(-1).detach().numpy()
        label = np.unique(labels)

        #for i in results:
        #    results = np.append(results, prefetch_neighbour(i))
        #results = np.unique(results)

        #intersection = (outputs.reshape(-1) & labels.reshape(-1)).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        #intersection = (results & label).sum((1, 2))
        intersection = len(np.intersect1d(results, label))
        #print(intersection)
        #union = (results | label).sum((1, 2))
        union = min(len(label), len(results)) # output size
        iou = 1 - (intersection + self.SMOOTH) / (union + self.SMOOTH)  # We smooth our devision to avoid 0/0
        accuracy = (intersection + self.SMOOTH) / (union + self.SMOOTH)
        #union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
        
        #iou = (intersection + SMOOTH) / (union + SMOOTH) 
        
        thresholded = torch.tensor(iou, requires_grad=True)
        #print(thresholded)
        return accuracy, thresholded  # Or thresholded.mean() if you are interested in average across the batch
        
        
    # Numpy version
    # Well, it's the same function, so I'm going to omit the comments

    def iou_numpy(self, outputs: np.array, labels: np.array):
        outputs = outputs.squeeze(1)
        
        intersection = (outputs & labels).sum((1, 2))
        union = (outputs | labels).sum((1, 2))
        
        iou = (intersection + self.SMOOTH) / (union + self.SMOOTH)
        
        thresholded = 1- np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
        
        return thresholded  # Or thresholded.mean()

    def prefetch_neighbour(res):
        ret=[]
        for i in range(5):
            ret.append(i+res)



class Chamfer1DLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Chamfer1DLoss, self).__init__()

    def chamfer_distance(self, x, y):
        assert(x.dim()== y.dim())
        assert(x.dim()== 1)
        cd = torch.tensor(0.0).requires_grad_()
        for i in x:
            dist = sys.maxsize
            for j in y:
                tmp = (torch.abs(torch.tensor(i-j))).clone().detach().requires_grad_(True)
                if dist > tmp:
                    dist = tmp
            cd = cd + dist
        return cd

    def forward(self, inputs, targets, alpha=0.5):        
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        inputs_lengths = inputs.size(0)
        targets_lengths = targets.size(0)

        l = alpha * 1/inputs_lengths * self.chamfer_distance(inputs, targets) + (1-alpha) * 1/targets_lengths * self.chamfer_distance(targets, inputs)
        #intersection = (inputs * targets).sum()                            
        #dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return l



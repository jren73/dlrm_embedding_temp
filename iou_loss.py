# Stolen from pytorch3d, and tweaked (very little).

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import warnings
from collections import namedtuple
from typing import Optional, Union

import torch

# Throws an error without this import
from chamferdist import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable




class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(
        self,
        source_cloud: torch.Tensor,
        target_cloud: torch.Tensor,
        bidirectional: Optional[bool] = False,
        reverse: Optional[bool] = False,
        reduction: Optional[str] = "mean",
    ):

        if not isinstance(source_cloud, torch.Tensor):
            raise TypeError(
                "Expected input type torch.Tensor. Got {} instead".format(type(pts))
            )
        if not isinstance(target_cloud, torch.Tensor):
            raise TypeError(
                "Expected input type torch.Tensor. Got {} instead".format(type(pts))
            )
        if source_cloud.device != target_cloud.device:
            raise ValueError(
                "Source and target clouds must be on the same device. "
                f"Got {source_cloud.device} and {target_cloud.device}."
            )

        batchsize_source, lengths_source, dim_source = source_cloud.shape
        batchsize_target, lengths_target, dim_target = target_cloud.shape

        lengths_source = (
            torch.ones(batchsize_source, dtype=torch.long, device=source_cloud.device)
            * lengths_source
        )
        lengths_target = (
            torch.ones(batchsize_target, dtype=torch.long, device=target_cloud.device)
            * lengths_target
        )

        chamfer_dist = None

        if batchsize_source != batchsize_target:
            raise ValueError(
                "Source and target pointclouds must have the same batchsize."
            )
        if dim_source != dim_target:
            raise ValueError(
                "Source and target pointclouds must have the same dimensionality."
            )
        if bidirectional and reverse:
            warnings.warn(
                "Both bidirectional and reverse set to True. "
                "bidirectional behavior takes precedence."
            )
        if reduction != "sum" and reduction != "mean":
            raise ValueError('Reduction must either be "sum" or "mean".')

        source_nn = knn_points(
            source_cloud,
            target_cloud,
            lengths1=lengths_source,
            lengths2=lengths_target,
            K=1,
        )

        target_nn = None
        if reverse or bidirectional:
            target_nn = knn_points(
                target_cloud,
                source_cloud,
                lengths1=lengths_target,
                lengths2=lengths_source,
                K=1,
            )

        # Forward Chamfer distance (batchsize_source, lengths_source)
        chamfer_forward = source_nn.dists[..., 0]
        chamfer_backward = None
        if reverse or bidirectional:
            # Backward Chamfer distance (batchsize_source, lengths_source)
            chamfer_backward = target_nn.dists[..., 0]

        chamfer_forward = chamfer_forward.sum(1)  # (batchsize_source,)
        if reverse or bidirectional:
            chamfer_backward = chamfer_backward.sum(1)  # (batchsize_target,)

        if reduction == "sum":
            chamfer_forward = chamfer_forward.sum()  # (1,)
            if reverse or bidirectional:
                chamfer_backward = chamfer_backward.sum()  # (1,)
        elif reduction == "mean":
            chamfer_forward = chamfer_forward.mean()  # (1,)
            if reverse or bidirectional:
                chamfer_backward = chamfer_backward.mean()  # (1,)

        if bidirectional:
            return chamfer_forward + chamfer_backward
        if reverse:
            return chamfer_backward

        return chamfer_forward
      
  def iou_statistic(outputs: torch.Tensor, labels: torch.Tensor):
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
    union1 = min(len(label), len(results)) # output size
    union2 = max(len(label), len(results))
    accuracy = (intersection + SMOOTH) / (union1 + SMOOTH)
    coverage = (intersection + SMOOTH) / (union2 + SMOOTH)
    #union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    #iou = (intersection + SMOOTH) / (union + SMOOTH) 
    
    thresholded = torch.tensor(iou, requires_grad=True)
    #print(thresholded)
    return accuracy, coverage
  
  

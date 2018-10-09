import bbox_helper
import torch

a = torch.tensor([[2.,2.,2.,2.],[3.,3.,2.,2.]])
b = torch.tensor([[3.,3.,2.,2.],[4.,4.,2.,2.]])

out = bbox_helper.iou(a,b)
print('iou',out)
import bbox_helper
import torch

# a = torch.tensor([[2.,2.,2.,2.],[3.,3.,2.,2.]])
# b = torch.tensor([[3.,3.,2.,2.],[4.,4.,2.,2.]])

# out = bbox_helper.iou(a,b)
# print('iou',out)

# a = torch.tensor([2,1,3,4])
# b = torch.cat((a.view(1,4),a.view(1,4)),0)
# print ('b',b)

# a = torch.tensor([[-1],[0],[2]])
# print('a.shape',a.size())
# #a = torch.tensor([[1,2,0,4,5]])
# print('a.nonzero',a.nonzero())
# a = a[a.nonzero()[:,0]]
# print('a',a)

a = torch.tensor([[-1,-2,3,4],[-1,3,4,-2]],dtype=torch.float)
a = torch.where(a > 0,a,torch.zeros(a.size()))
print ('a',a)


from util.module_util import summary_layers
import torch.nn as nn
import torch.nn.functional as F
from mobilenet import MobileNet
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')
m = MobileNet()
summary_layers(m, (3, 300,300))
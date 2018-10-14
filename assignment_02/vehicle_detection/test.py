import numpy as np
import torch.nn
import torch.optim as optim
import cityscape_dataset as csd
from torch.utils.data import Dataset
from torch.autograd import Variable
from bbox_helper import generate_prior_bboxes, match_priors,nms_bbox,loc2bbox
from data_loader import get_list
from ssd_net import SSD
from bbox_loss import MultiboxLoss







prior_box = train_dataset.get_prior_boxes()
boxes = loc2bbox(loc_preds, prior_box, center_var=0.1, size_var=0.2)
select_box = nms_bbox(boxes, conf_preds, overlap_threshold=0.5, prob_threshold=0.6)


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
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import os


use_gpu = True
img_dir = '../cityscapes_samples/'
label_dir = '../cityscapes_samples_labels/'
learning_rate = 0.001
max_epochs = 20

test_list = get_list(img_dir, label_dir)
print('list',test_list)
# valid_list = test_list[-20: ]
# test_list = test_list[0:-20]
test_dataset = csd.CityScapeDataset(test_list, train=False, show=False)
test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=16,
                                                shuffle=False,
                                                num_workers=0)
print('test items:', len(test_dataset))

file_name = 'SSD'
test_net_state = torch.load(os.path.join('.', file_name+'.pth'))

net = SSD(3)
net.cuda()
net.load_state_dict(test_net_state)
itr = 0

net.eval()
for test_batch_idx,(loc_targets, conf_targets,imgs) in enumerate(test_data_loader):
  itr += 1
  imgs = imgs.permute(0, 3, 1, 2).contiguous()
  if use_gpu:
    imgs = imgs.cuda()
  imgs = Variable(imgs)
  conf, loc = net.forward(imgs)
  conf = conf[0,...]
  loc = loc[0,...].cpu()
  
  prior =  test_dataset.get_prior_bbox()
  prior = torch.unsqueeze(prior, 0)
  # prior = prior.cuda()
  real_bounding_box = loc2bbox(loc,prior,center_var=0.1,size_var=0.2)
  real_bounding_box = torch.squeeze(real_bounding_box,0)
  sel_box = nms_bbox(real_bounding_box, conf, overlap_threshold=0.5, prob_threshold=0.6)

  # img = Image.open(os.path.join(img_dir, 'bad-honnef', 'bad-honnef_000000_000000_leftImg8bit.png'))
  img = imgs[0].permute(1,2,0).contiguous()
  true_loc = loc_targets[0,conf_targets[0,...].nonzero(),:].squeeze()
  img = Image.fromarray(np.uint8(img*128 + 127))
  draw = ImageDraw.Draw(img)
  sel_box = np.array(sel_box)
  print('sel_box.shape',sel_box)
  for box in range(0,sel_box.shape[0]):
    sel_cx = sel_box[box,0]
    sel_cy = sel_box[box,1]
    sel_w = sel_box[box,2]
    sel_h = sel_box[box,3]
    sel_bbox = np.array([sel_cx-sel_w/2,sel_cy-sel_h/2,sel_cx+sel_w/2,sel_cy+sel_h/2])
    sel_bbox = sel_bbox*128+127
    draw.rectangle(list(sel_bbox), outline='red')
  for b in range(0,true_loc.shape[0]):
    cx = true_loc[b,0]
    cy = true_loc[b,1]
    w = true_loc[b,2]
    h = true_loc[b,3]
    true_bbox = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
    true_bbox = np.array(true_bbox)*128 +127
    draw.rectangle(list(true_bbox), outline='green')

    # print(box.shape)
    print('tls', true_loc[b,:]*128 + 127)

  img.show()
  break

  print('conf',conf.shape)
  print('loc',loc.shape)



# prior =  test_dataset.get_prior_bbox()
# real_bounding_box = loc2bbox(loc,prior,center_var=0.1,size_var=0.2)
# sel_box = nms_bbox(real_bounding_box, conf, overlap_threshold=0.5, prob_threshold=0.6)

# draw = ImageDraw.Draw(img)
# for box in boxes:
#     box[::2] *= img.width
#     box[1::2] *= img.height
#     draw.rectangle(list(box), outline='red')
# img.show()

import numpy as np
import torch.nn
import torch.optim as optim
import cityscape_dataset as csd
from torch.utils.data import Dataset
from torch.autograd import Variable
from bbox_helper import generate_prior_bboxes, match_priors,nms_bbox,loc2bbox
from data_loader import get_list
from ssd_net import SSD
# from bbox_loss import MultiboxLoss
from PIL import Image, ImageDraw
# import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


use_gpu = True
img_dir = '../cityscapes_samples/'
label_dir = '../cityscapes_samples_labels/'
learning_rate = 0.001
max_epochs = 20

test_list = get_list(img_dir, label_dir)
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
if use_gpu:
  net = net.cuda()
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
  class_list,sel_box = nms_bbox(real_bounding_box, conf, overlap_threshold=0.5, prob_threshold=0.6)
  for c in class_list:
    if c == 1:


  # img = Image.open(os.path.join(img_dir, 'bad-honnef', 'bad-honnef_000000_000000_leftImg8bit.png'))
  img = imgs[0].permute(1,2,0).contiguous()
  true_loc = loc_targets[0,conf_targets[0,...].nonzero(),:].squeeze()
  img = Image.fromarray(np.uint8(img*128 + 127))
  # draw = ImageDraw.Draw(img)
  sel_box = np.array(sel_box[:,1:])
  print('sel_box.shape',sel_box)

  fig, ax = plt.subplots(1)
  ax.imshow(img)
  for idx in range(len(sel_box)):
      cx, cy, w, h = sel_box[idx]*300
      if class_list[idx] == 1:
        rect = patches.Rectangle((cx-w/2,cy-h/2),w,h, linewidth=2, edgecolor='r', facecolor='none')
      if class_list[idx] == 2:
        rect = patches.Rectangle((cx-w/2,cy-h/2),w,h, linewidth=2, edgecolor='g', facecolor='none')
      ax.add_patch(rect)
  plt.show()


  
  # print('conf',conf.shape)
  # print('loc',loc.shape)

# draw = ImageDraw.Draw(img)
# for box in boxes:
#     box[::2] *= img.width
#     box[1::2] *= img.height
#     draw.rectangle(list(box), outline='red')
# img.show()

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

train_list = get_list(img_dir, label_dir)
print('list',train_list)
# valid_list = train_list[-20: ]
# train_list = train_list[0:-20]
train_dataset = csd.CityScapeDataset(train_list, train=True, show=False)
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=32,
                                                shuffle=True,
                                                num_workers=0)
print('train items:', len(train_dataset))

idx, (bbox, label, img) = next(enumerate(train_data_loader))


# valid_dataset = csd.CityScapeDataset(train_list, False, False)
# valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
#                                                 batch_size=4,
#                                                 shuffle=False,
#                                                  num_workers=0)
# print('validation items:', len(valid_dataset))

net = SSD(3)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
criterion = MultiboxLoss([0.1,0.1,0.2,0.2])

if use_gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net.cuda()
    criterion.cuda()

train_losses = []
valid_losses = []
itr = 0
for epoch_idx in range(0, max_epochs):
    for train_batch_idx, (loc_targets, conf_targets, imgs) in enumerate(train_data_loader):
       
        itr += 1
        net.train()

        imgs = imgs.permute(0, 3, 1, 2).contiguous() # [batch_size, W, H, CH] -> [batch_size, CH, W, H]
        if use_gpu:
            imgs = imgs.cuda()
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()

        imgs = Variable(imgs)
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)

        optimizer.zero_grad()
        conf_preds, loc_preds = net.forward(imgs)
        conf_loss, loc_huber_loss, loss = criterion.forward(conf_preds, loc_preds, conf_targets, loc_targets)
        
        loss.backward()
        optimizer.step()
        train_losses.append((itr, conf_loss.item(), loc_huber_loss.item(), loss.item()))
        
        # if train_batch_idx % 2 == 0:
        print('Epoch: %d Itr: %d Conf_Loss: %f Loc_Loss: %f Loss: %f' 
                % (epoch_idx, itr, conf_loss.item(), loc_huber_loss.item(), loss.item()))

net_state = net.state_dict()
file_name = 'SSD'
torch.save(net_state,  os.path.join('.', file_name+'.pth'))



'''
            net.eval() 
            valid_loss_set = []
            valid_itr = 0

            for valid_batch_idx, (valid_input, valid_lm) in enumerate(valid_data_loader):
                net.eval()
                valid_input = torch.transpose(valid_input, 1, 3)
                valid_input = Variable(valid_input.cuda())
                valid_out = net.forward(valid_input)

                valid_lm = Variable(valid_lm.cuda())
                valid_loss = criterion(valid_out.view((-1, 2, 7)), valid_lm)
                valid_loss_set.append(valid_loss.item())

                valid_itr += 1
                if valid_itr > 5:
                    break
            
            avg_valid_loss = np.mean(np.asarray(valid_loss_set))
            print('Valid Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, avg_valid_loss))
            valid_losses.append((itr, avg_valid_loss))

train_losses = np.asarray(train_losses)
valid_losses = np.asarray(valid_losses)
plt.plot(train_losses[:, 0],
         train_losses[:, 1])
plt.plot(valid_losses[:, 0],
         valid_losses[:, 1])
plt.show()
plt.savefig(file_name+'.jpg')

net_state = net.state_dict()
torch.save(net_state, os.path.join(lfw_dir, file_name+'.pth'))
'''
import torch
import numpy as np
import data_process as dp
import os
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
from PIL import Image

test_set_path = 'LFW_annotation_test.txt'
# lfw_dir = '../../../../Courses_data'
# lfw_dataset_dir = '../../../../Courses_data/lfw'
# test_set_path = os.path.join(lfw_dir, 'LFW_annotation_test.txt')
max_epochs = 180
learning_rate = 0.0005
pretrained = True
str_pre = 'pre'
file_name = 'lfw_alexnet_'+str(learning_rate)+'_'+str(max_epochs)+'_'+str_pre

test_list = dp.get_list(test_set_path)
test_dataset = dp.LFWDataSet(test_list)
test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=64,
                                                shuffle=False,
                                                num_workers=0)
print('test items: ', len(test_dataset))

net = models.alexnet(pretrained=pretrained)
net.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 14),
        )
net.cuda()

test_net_state = torch.load(os.path.join('.', file_name+'.pth'))
net.load_state_dict(test_net_state)
net.eval()

## test
result = []
test_lm_list = []
for test_batch_idx, (test_input, test_lm) in enumerate(test_data_loader):
    test_input = torch.transpose(test_input, 1, 3)
    test_out = net.forward(test_input.cuda())
    test_out = test_out.view((-1, 2, 7)).cpu().detach().numpy()
    for i in range(0, len(test_out)):
        test_lm_list.append(test_lm[i].cpu().numpy().T)
        result.append(test_out[i].T)

## radius
dist_test = []
test_lm_list = np.array(test_lm_list)
result = np.array(result)
print('result_size', len(result))
dist_test = np.linalg.norm(test_lm_list - result,axis=(1,2))
# dist_test = np.sqrt(np.square(test_lm_list - result))
percent = []
for i in range(0,20):
    percent.append(len(np.where(dist_test < i/10.0)[0])*1.0/len(test_lm_list)) 
   
step = np.arange(0,2,0.1)

plt.plot(step, percent)
plt.show()

## write result file
## format: name '\t' original lm[x1,y1] ' ' ... '\t' output lm [x1',y1'] ' '
file = open(file_name+'.txt', 'w')
for idx in range(0, len(test_list)):
    file.write(test_list[idx]['name'] + '\t')
    for item in test_lm_list[idx]:
        file.write(str(item.squeeze()) + ' ')
    file.write('\t')
    for item in result[idx]:
        file.write(str(item.squeeze()) + ' ')
    file.write('\n')
file.close()
print('finish writing')

## show some result
show_list = test_list[random.randint(0, len(test_list)-4):-1]
show_dataset = dp.LFWDataSet(show_list)
show_data_loader = torch.utils.data.DataLoader(show_dataset,
                                                batch_size=4,
                                                shuffle=False,
                                                num_workers=0)

idx, (img, lm) = next(enumerate(show_data_loader))
test_input = torch.transpose(img, 1, 3)
test_lm = net.forward(test_input.cuda())
test_lm = test_lm.view((-1, 2, 7)).cpu().detach().numpy()

nd_img = img.cpu().numpy()
nd_lm = lm.cpu().numpy()
# dp.show_landmarks(nd_img, nd_lm) # ground truth
dp.show_landmarks(nd_img, test_lm)
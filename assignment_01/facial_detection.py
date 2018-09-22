#!usr/bin/python3
import numpy as np
import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import matplotlib.pyplot as plt
import resnet
import data_process as dp

# use ResNet18
net = resnet.resnet18(pretrained=True,num_classes=18)

## lock the layer before fc layer
#ct = 0
#for child in net.children() :
#    ct += 1
#    if ct < 9:
#        for param in child.parameters():
#            param.requires_grad = False

torch.set_default_tensor_type('torch.cuda.FloatTensor')


# Begin training
train_set_path = 'LFW_annotation_train.txt'
max_epochs = 180
learning_rate = 0.0005
str_pre = 'pre'
file_name = 'lfw_resnet_'+str(learning_rate)+'_'+str(max_epochs)+'_'+str_pre

train_list = dp.get_list(train_set_path)
valid_list = train_list[-2000: ]
train_list = train_list[: -2000]

transform = ['flip','rcrop']
train_dataset = dp.LFWDataSet(train_list,transform=transform)
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=64,
                                                shuffle=False,
                                                num_workers=0)
valid_dataset = dp.LFWDataSet(valid_list, transform=transform)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=0)
net.cuda()

criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

train_losses = []
valid_losses = []
itr = 0
for epoch_idx in range(0, max_epochs):
    for train_batch_idx, (train_input, train_lm) in enumerate(train_data_loader):
        itr += 1
        net.train()
        optimizer.zero_grad()
        print(train_input.shape)
        train_input = torch.transpose(train_input,1,3)
        print(train_input.shape)
        train_input = Variable(train_input.cuda())
        train_out = net.forward(train_input)

        train_lm = Variable(train_lm.cuda())
        loss = criterion(train_out.view(-1,2,7),train_lm)

        loss.backward()
        optimizer.step()
        train_losses.append((itr, loss.item()))
 # add validation while training
        if train_batch_idx % 200 == 0:
            print('Epoch: %d Itr: %d Loss: %f' %(epoch_idx, itr, loss.item()))
            net.eval()
            valid_loss_set = []
            valid_itr = 0

            for valid_batch_idx, (valid_input, valid_lm) in enumerate(valid_data_loader):
                net.eval()
                valid_input = torch.transpose(valid_input,1,3)
                valid_out = net.forward(valid_input)
                
                valid_lm = Variable(valid_lm.cuda())
                valid_loss = criterion(valid_out.view((-1,2,7)),valid_lm)
                valid_loss_set.append(valid_loss.item())

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
# plt.show()
plt.savefig(file_name+'.jpg')

net_state = net.state_dict()
torch.save(net_state,  os.path.join('.', file_name+'.pth'))







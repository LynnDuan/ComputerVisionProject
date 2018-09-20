#!usr/bin/python3
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.model_zoo as model_zoo
import resnet
model = resnet.resnet18(pretrained=True,num_classes=18)

## lock the layer before fc layer
#ct = 0
#for child in model.children() :
#    ct += 1
#    if ct < 9:
#        for param in child.parameters():
#            param.requires_grad = False


## load and split data for training/validation/testing
import os
label_file_path = 'LFW_annotation_train.txt'


data_list = []
with open(label_file_path,"r") as f:
    for line in f:
        tokens = line.split('\t') #split the line with ' '
        print(tokens)
        file_path = tokens[0].strip()

        face_bbox = []
        for i in range(1, 5):
            face_bbox.append(int(tokens[i]))
        face_bbox = np.array(face_bbox, dtype=np.int)

        canthus_rr = []
        for i in range(5, 7):
            canthus_rr.append(float(tokens[i]))
        canthus_rr = np.array(canthus_rr, dtype=np.float32)

        canthus_rl = []
        for i in range(7, 9):
            canthus_rl.append(float(tokens[i]))
        canthus_rl = np.array(canthus_rl, dtype=np.float32)

        canthus_lr = []
        for i in range(9, 11):
            canthus_lr.append(float(tokens[i]))
        canthus_lr = np.array(canthus_lr, dtype=np.float32)

        canthus_ll = []
        for i in range(11, 13):
            canthus_ll.append(float(tokens[i]))
        canthus_ll = np.array(canthus_ll, dtype=np.float32)

        mouth_r = []
        for i in range(13,15):
            mouth_r.append(float(tokens[i]))
        mouth_r = np.array(mouth_r,dtype=np.float32)

        mouth_l = []
        for i in range(15,17):
            mouth_l.append(float(tokens[i]))
        mouth_l = np.array(mouth_l,dtype=np.float32)

        nose = []
        for i in range(17,19):
            nose.append(float(tokens[i]))
        nose = np.array(nose,dtype=np.float32)



        data_list.append({'file_path': file_path, 'face_bbox': face_bbox, 'canthus_rr':canthus_rr, 'canthus_rl':canthus_rl, 'canthus_lr':canthus_lr,'canthus_ll':canthus_ll, 'mouth_r':mouth_r, 'mouth_l':mouth_l, 'nose':nose})






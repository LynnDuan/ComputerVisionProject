import os
import glob
import numpy as np
import json
import pandas as pd

'''
input: img_dir, label_dir
return: data_list
format: dict {'img': img_path, 'h': float, 'w': float, 
        'bboxes': [[left_top, right_bottom],...],
        'labels': [label, ...]}
'''

def get_list(img_dir, label_dir):
    label_dict_file = 'label_dict.csv'
    img_list = glob.glob(os.path.join(img_dir, '*/*'))
    label_list = glob.glob(os.path.join(label_dir, '*/*.json'))
    label_dict = {'bicycle': 1, 'bicyclegroup': 1, 'bus': 1, 'busgroup': 1, 'caravan': 1,
                'car': 1, 'cargroup': 1, 'motorcycle': 1, 'motorcyclegroup': 1, 
                'on rails': 1, 'person': 2, 'persongroup': 2, 'rider': 2, 
                'ridergroup': 2, 'trailer': 1, 'train': 1, 'truck': 1}

    # put items into data_list
    data_list = []
    for idx in range(len(label_list)):
        with open(label_list[idx], 'r') as read_file:
            label = json.load(read_file)
            sample_bboxes = []
            sample_labels = []
            for obj in label['objects']:
                if obj['label'] in label_dict:
                    label_idx = label_dict[obj['label']]
                    left_top = np.amin(np.asarray(obj['polygon']), axis=0)
                    right_bottom = np.amax(np.asarray(obj['polygon']), axis=0)
                    sample_bboxes.append([left_top, right_bottom])
                    sample_labels.append(label_idx)

            data_list.append({'img': img_list[idx], 'h': label['imgHeight'], 'w': label['imgWidth'],
                            'bboxes': np.asarray(sample_bboxes), 'labels': np.asarray(sample_labels)})

    return data_list






    # label_no = []
    # get label, write label file
    # if os.path.isfile(label_dict_file) == False:
    #     for idx in range(len(label_list)):
    #         with open(label_list[idx], 'r') as read_file:
    #             label = json.load(read_file)
    #             for obj in label['objects']:
    #                 if obj['label'] in label_no:
    #                     continue
    #                 else:
    #                     label_no.append(obj['label'])
    #     label_no.sort()
    #     df = pd.DataFrame(label_no)
    #     df.to_csv(label_dict_file, index=False)

    # # read label file to label dict
    # # eg. {'bicycle': 1, 'bicyclegroup': 2, 'building': 3,...}
    # df = pd.read_csv(label_dict_file)
    # d = df.T.to_dict('index')['0']
    # for k, v in d.items():
    #     label_dict.setdefault(v, k+1)
    # # print(label_dict)
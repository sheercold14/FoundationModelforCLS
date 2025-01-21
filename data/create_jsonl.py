path = '/data/lishichao/data/hospital_organ/organ/Thyroid/'
import os
import os.path as osp
import random
import json
json_path = '/data/lishichao/project/Foundation-Medical/data/Thyroid_json_label.jsonl'
img_folder_list = os.listdir(path)
# random.shuffle(img_folder_list)
# half = len(img_folder_list) // 2
# test_data = img_folder_list[:half]  # 前一半作为测试数据
# train_data = img_folder_list[half:]  # 后一半作为训练数据
label_dict = {'N':'benign','C':'malignant'}
data_list = []
for img in os.listdir(path):
    data_dict = {}
    img_path = osp.join(path,img)
    data_dict['image'] = img_path 
    data_dict['label'] = label_dict[img_path[-5]]
    data_dict['split'] = '7'
    data_list.append(data_dict)
print(data_dict)
with open(json_path,'w') as json_file:
    for data_dict in data_list:
        json_line = json.dumps(data_dict) + '\n'
        json_file.write(json_line)
print('all data have been written')
    # print(img_folder)
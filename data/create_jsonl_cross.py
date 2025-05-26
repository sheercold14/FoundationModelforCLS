path = '/data/lishichao/data/hospital_organ/Thyroid_RGB_labeled'
# path2 = '/data/lishichao/data/hospital_organ/hospital3_organ_merged'
# path2 = '/data/lishichao/data/hospital_organ/organ_202411/Thyroid'
import os
import os.path as osp
import random
import json
json_path = '/data/lishichao/project/Foundation-Medical/data/v1/RGB_thyroid.jsonl'
img_list = os.listdir(path)
random.shuffle(img_list)
# random.shuffle(img_folder_list)
# half = len(img_folder_list) // 2
# test_data = img_folder_list[:half]  # 前一半作为测试数据
# train_data = img_folder_list[half:]  # 后一半作为训练数据
label_dict = {'N':'benign','C':'malignant','n':'benign','c':'malignant'}
data_list = []
# 将数据分为 5 折
folds = 3
fold_size = len(img_list) // folds
data_list = []

for i, img in enumerate(img_list):
    data_dict = {}
    img_path = osp.join(path, img)
    
    # 构建数据字典
    data_dict['image'] = img
    if img_path[-5] not in label_dict.keys() and img_path[-8] not in label_dict.keys():
        continue
    if img_path[-5] in label_dict.keys(): 
        data_dict['label'] = label_dict[img_path[-5]]
    else: 
        data_dict['label'] = label_dict[img_path[-8]]
    data_dict['split'] = str(i % folds)  # 按顺序将数据分配到不同的折
    
    data_list.append(data_dict)

# 打印最后一个数据字典（示例）
print(data_list[-1])

# 将数据写入 .jsonl 文件
with open(json_path, 'w') as json_file:
    for data_dict in data_list:
        json_line = json.dumps(data_dict) + '\n'
        json_file.write(json_line)

print('All data have been written and divided into 5 folds.')
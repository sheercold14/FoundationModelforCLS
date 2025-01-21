import json
import shutil
import os

def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每行的JSON字符串为字典
            data = json.loads(line.strip())
            data_list.append(data['image'])
    return data_list

# 使用示例
file_path = '/data/lishichao/project/Foundation-Medical/data/v2/medical_json_label.jsonl'  # 替换为你的jsonl文件路径
jsonl_data = read_jsonl(file_path)
target_root = '/data/lishichao/project/Foundation-Medical/SAM/data/hospital_3'
for path in jsonl_data:
    target_name = path.split('/')[-1]
    target_path = os.path.join(target_root,target_name)
    shutil.copy(path,target_path)
    
print(jsonl_data)
import os
from PIL import Image
import numpy as np
import json
def calculate_mean_and_std(folder_path,name_list):
    # 初始化用于累加的变量
    total_pixels = None
    pixel_sum = np.zeros(3, dtype=np.float64)  # RGB通道的像素值总和
    pixel_squared_sum = np.zeros(3, dtype=np.float64)  # RGB通道的像素值平方总和

    # 遍历文件夹中的所有图像文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and filename in name_list:
            file_path = os.path.join(folder_path, filename)
            with Image.open(file_path) as img:
                # 将图像转换为RGB（如果需要的话）
                # img = img.convert('RGB')
                # 将图像转换为numpy数组
                pixels = np.asarray(img).astype(np.float64)

                # 更新累加变量
                pixel_count = pixels.shape[0] * pixels.shape[1]
                if total_pixels is None:
                    total_pixels = pixel_count
                else:
                    total_pixels += pixel_count

                pixel_sum += pixels.sum(axis=(0, 1))
                pixel_squared_sum += (pixels ** 2).sum(axis=(0, 1))

    # 计算均值
    mean = pixel_sum / total_pixels

    # 计算方差
    variance = pixel_squared_sum / total_pixels - mean ** 2

    return mean, variance
def get_name(x):
    img_file = x.split('/')[-1]
    return img_file
    
datainfo = [json.loads(q) for q in open(os.path.expanduser('/data/lishichao/project/Foundation-Medical/data/medical_json_label.jsonl'), "r")]
name_list = [get_name(datainfo[i]['image']) for i in range(len(datainfo)) if datainfo[i]['split'] in ['1','2','3']]

# 指定包含图像的文件夹路径
folder_path = '/data/lishichao/project/LLaVA-Med/data/train_images'
mean, variance = calculate_mean_and_std(folder_path,name_list)

print(f"Mean (per channel): {mean/255}")
print(f"Variance (per channel): {np.sqrt(variance)/255}")


# Mean (per channel): [0.71576676 0.64997176 0.62305216]
# Variance (per channel): [0.20383366 0.27344528 0.28959818]
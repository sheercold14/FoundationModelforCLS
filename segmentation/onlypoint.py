import os
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.use('tkagg')
import json

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 使用matplotlib进行前景和背景点标注
def get_two_points(image):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 使用 ginput 选择两个点，第一个点为前景，第二个点为背景
    print("请点击图片中的两个点，依次是前景点和背景点")
    points = plt.ginput(2, timeout=360)
    plt.close(fig)

    # 解析选中的两个点
    (x1, y1), (x2, y2) = points
    x1, x2 = int(x1), int(x2)
    y1, y2 = int(y1), int(y2)

    return [(x1, y1), (x2, y2)]

# 加载 SAM 模型
sam = sam_model_registry["vit_b"](checkpoint="/data/lishichao/project/Foundation-Medical/SAM/sam_vit_b_01ec64_pre.pth")
predictor = SamPredictor(sam)

# 处理单张图片并保存 mask
def process_image(image_path, output_dir):
    # 载入图像
    image = np.array(Image.open(image_path).convert('RGB'))
    points = get_two_points(image)
    print(f"选中的前景和背景点坐标: {points}")

    # 设置图像
    predictor.set_image(image)

    # 将选中的前景点和背景点作为 prompt 进行预测
    input_points = np.array(points)
    input_labels = np.array([1, 0])  # 标签 1 表示前景，标签 0 表示背景

    # 预测 mask
    masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels)

    # 保存生成的 mask
    mask_img = Image.fromarray((masks[0] * 255).astype(np.uint8))

    # 构建输出文件路径
    image_name = os.path.basename(image_path)
    mask_name = os.path.splitext(image_name)[0] + ".png"
    mask_path = os.path.join(output_dir, mask_name)

    mask_img.save(mask_path)
    print(f"Mask saved for {image_name} at {mask_path}")

# 批量处理文件夹中的所有图片
def process_images_in_folder(input_dir, output_dir, num_images=100, load_random_list=True):
    # 如果输出文件夹不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if load_random_list:
        random_images = load_json_file('/data/lishichao/project/Foundation-Medical/SAM/random_images_3_hos.json')
    else:
        # 获取输入文件夹中的所有图片文件
        all_images = [f for f in os.listdir(input_dir) if f.endswith(".png") or f.endswith(".jpg")]
        
        # 确保不超过文件夹内图片数量
        num_images = min(num_images, len(all_images))

        # 随机选择100张图片
        random_images = random.sample(all_images, num_images)

        # 保存 random_images 列表为 json 文件
        with open("./random_images_v2.json", "w") as f:
            json.dump(random_images, f, indent=4)

    # 遍历输入文件夹中的所有图片
    for filename in random_images:
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            check_path_jpg = os.path.join(output_dir, filename.split('.')[0] + '.jpg')
            check_path_png = os.path.join(output_dir, filename.split('.')[0] + '.png')
            if not os.path.exists(check_path_jpg) and not os.path.exists(check_path_png):
                process_image(image_path, output_dir)

# 设置文件夹路径和单点坐标
input_dir = "/data/lishichao/project/Foundation-Medical/SAM/data/hospital_3"
output_dir = "/data/lishichao/project/Foundation-Medical/SAM/data/hospital_point_mask"

# 批量处理文件夹中的图片
process_images_in_folder(input_dir, output_dir, num_images=100)

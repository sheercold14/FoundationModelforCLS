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
        data = json.load(file)  # 加载 JSON 文件为 Python 对象（列表）
    return data

# 使用matplotlib进行五点标注（前两个点为box，后五个点为point）
def get_box_and_points(image):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 使用 ginput 选择两个点作为box，后五个点作为point
    print("请依次点击图片：前两个点作为矩形框（box），后五个点作为提示（point）")
    points = plt.ginput(7, timeout=360)
    plt.close(fig)

    # 确保选择了七个点（2个用于box，5个用于point）
    if len(points) != 7:
        raise ValueError(f"选择的点数不正确，必须选择7个点，但得到 {len(points)} 个点")

    # 前两个点用于定义box，后五个点作为point
    box = [int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1])]
    point_coords = [(int(x), int(y)) for (x, y) in points[2:]]
    
    return box, point_coords

# 加载 SAM 模型
sam = sam_model_registry["vit_b"](checkpoint="/data/lishichao/project/Foundation-Medical/SAM/sam_vit_b_01ec64_pre.pth")
predictor = SamPredictor(sam)

# 处理单张图片并保存 mask
def process_image(image_path, output_dir):
    # 载入图像
    image = np.array(Image.open(image_path).convert('RGB'))

    # 获取矩形框和五个点
    box, points = get_box_and_points(image)
    print(f"选中的矩形框: {box}")
    print(f"选中的五个点坐标: {points}")

    # 设置图像
    predictor.set_image(image)

    # 将选中的矩形框和五个点作为 prompt 进行预测
    input_box = np.array(box).reshape(1, 4)
    input_points = np.array(points)
    input_labels = np.ones(input_points.shape[0])  # 为所有点设置标签 1，表示前景

    # 同时传入 box 和 point 进行预测
    masks, _, _ = predictor.predict(box=input_box, point_coords=input_points, point_labels=input_labels)

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
        with open("./random_images_3_hos.json", "w") as f:
            json.dump(random_images, f, indent=4)

    # 遍历输入文件夹中的所有图片
    for filename in random_images:
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            check_path_jpg = os.path.join(output_dir, filename.split('.')[0] + '.jpg')
            check_path_png = os.path.join(output_dir, filename.split('.')[0] + '.png')
            if not os.path.exists(check_path_jpg) and not os.path.exists(check_path_png):
                process_image(image_path, output_dir)

# 设置文件夹路径
input_dir = "/data/lishichao/project/Foundation-Medical/SAM/data/hospital_3"
output_dir = "/data/lishichao/project/Foundation-Medical/SAM/data/hospital_3_box_point_mask"

# 批量处理文件夹中的图片
process_images_in_folder(input_dir, output_dir, num_images=100,load_random_list=True)
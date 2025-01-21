
import numpy as np
import matplotlib 
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.patches as patches

# 使用matplotlib进行矩形框选
def get_bounding_box(image):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 使用 ginput 选择矩形框的两个角点
    print("请点击图片两个角点，依次是左上角和右下角")
    points = plt.ginput(2)
    plt.close(fig)

    # 解析选中的两个点，作为矩形框
    (x1, y1), (x2, y2) = points
    x1, x2 = int(x1), int(x2)
    y1, y2 = int(y1), int(y2)
    
    return [x1, y1, x2, y2]

# 载入图像
image_path = "/data/lishichao/project/Foundation-Medical/results/im1.png"
image = np.array(Image.open(image_path))

# 获取用户选择的矩形框坐标

# bounding_box = get_bounding_box(image)
# print(f"选中的矩形框坐标: {bounding_box}")

# 加载 SAM 模型
sam = sam_model_registry["vit_b"](checkpoint="/data/lishichao/project/Foundation-Medical/SAM/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)

# 设置图像
predictor.set_image(image)

bounding_box = [24, 7, 194, 202]
# 将选中的矩形框作为 prompt 进行预测
input_box = np.array(bounding_box).reshape(1, 4)  # 转换为形状 (1, 4)
masks, _, _ = predictor.predict(box=input_box)

# 可视化结果
fig, ax = plt.subplots(1)
ax.imshow(image)
ax.imshow(masks[0], alpha=0.5)  # 在原图上叠加半透明 mask
plt.show()

# 保存生成的 mask
mask_img = Image.fromarray((masks[0] * 255).astype(np.uint8))
mask_img.save("output_mask.png")
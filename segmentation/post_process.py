import cv2
import numpy as np
from matplotlib import pyplot as plt

# 你的图像路径和输出路径
image_path = "/data/lishichao/project/Foundation-Medical/SAM/data/box_point_mask/Breast_2422304A_N_mask.png"
output_path = "/data/lishichao/project/Foundation-Medical/SAM/data/processed_mask.png"

# 读取图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 1: 先进行形态学开运算（去除小的噪声）
kernel = np.ones((3, 3), np.uint8)  # 设置较小的核大小
opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Step 2: 再进行形态学闭运算（填充小的黑色孔洞）
closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

# Step 3: 检测并填充小面积的黑色区域
# 复制图像进行填充操作
filled_image = closed_image.copy()

# 查找所有黑色区域的轮廓并根据面积判断是否进行填充
contours, _ = cv2.findContours(255 - filled_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area < 500:  # 仅填充面积小于500的黑色区域，阈值可根据图像调整
        cv2.drawContours(filled_image, [contour], 0, 255, -1)

# Step 4: 对图像进行轻微的膨胀运算
kernel_dilate = np.ones((3, 3), np.uint8)  # 较小的膨胀核，3x3
dilated_image = cv2.dilate(filled_image, kernel_dilate, iterations=1)  # 膨胀一次

# 显示和保存处理后的图像
plt.figure(figsize=(10, 5))

# 原始图像
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

# 填充后处理图像
plt.subplot(1, 3, 2)
plt.title('Processed Image')
plt.imshow(filled_image, cmap='gray')

# 膨胀后的图像
plt.subplot(1, 3, 3)
plt.title('Dilated Image')
plt.imshow(dilated_image, cmap='gray')

plt.show()

# 保存结果到指定路径
cv2.imwrite(output_path, dilated_image)
print(f"Processed image saved as {output_path}")
#!/bin.bash
params_path="/data/lishichao/project/Foundation-Medical/config/hospital_3/thyroid.json"
val_every=1
gpu=3
cross_folds=5  # 设定交叉验证的折数
experiment_name="hospitals_3_Thyroid_quality1_img_lr5e4_augment"  # 设定实验名称

# 设置要更改的工作目录
TARGET_DIR="/data/lishichao/project/Foundation-Medical"

# 使用 cd 命令更改工作目录
cd "$TARGET_DIR" || { echo "Failed to change directory to $TARGET_DIR"; exit 1; }
# 循环进行交叉验证
for (( i=0; i<cross_folds; i++ ))
do
    echo "Running fold $i..."
    SHELL="./shell/hospital_3/augment/${experiment_name}_fold${i}.txt"
    python -m classification --params_path "$params_path" --val_every "$val_every" --gpu "$gpu" --cross_folder "$i" > $SHELL 2>&1
done
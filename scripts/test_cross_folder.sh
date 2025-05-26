#!/bin.bash
params_path="/data/lishichao/project/Foundation-Medical/config/hospital_3/202502/hospital3_all_thyroid_img.json"
val_every=1
gpu=2
cross_folds=5  # 设定交叉验证的折数

root="/data/lishichao/project/Foundation-Medical/shell/hospital_3/all_thyroid"
experiment_name="test_all_thyroid_img_augment_20250218"  # 设定实验名称
folder_name="all_thyroid_img_aug"

# 设置要更改的工作目录
TARGET_DIR="/data/lishichao/project/Foundation-Medical"

# 使用 cd 命令更改工作目录
cd "$TARGET_DIR" || { echo "Failed to change directory to $TARGET_DIR"; exit 1; }

if [ ! -d "$root/$folder_name" ]; then
    echo "Directory does not exist, creating: $root/$folder_name"
    mkdir -p "$root/$folder_name"
else
    echo "Directory already exists: $root/$folder_name"
fi

# 循环进行交叉验证
for (( i=0; i<2; i++ ))
do
    echo "Running fold $i..."
    SHELL="${root}/${folder_name}/${experiment_name}_fold${i}.txt"
    python -m classification --params_path "$params_path" --test --val_every "$val_every" --gpu "$gpu" --cross_folder "$i" > $SHELL 2>&1
done
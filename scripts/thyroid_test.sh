#!/bin.bash
params_path="/data/lishichao/project/Foundation-Medical/config/hospital_3/202411/text.json"
val_every=1
gpu=0
cross_folds=5  # 设定交叉验证的折数
experiment_name="hospitals_3_Thyroid_text_test_lr5e4_testroc_teston202411"  # 设定实验名称

# 设置要更改的工作目录
TARGET_DIR="/data/lishichao/project/Foundation-Medical"

# 使用 cd 命令更改工作目录
cd "$TARGET_DIR" || { echo "Failed to change directory to $TARGET_DIR"; exit 1; }
# 循环进行交叉验证


SHELL="./shell/hospital_3/test/${experiment_name}_fold${i}.txt"
python -m classification --params_path "$params_path" --test --val_every "$val_every" --gpu "$gpu" --cross_folder [0,1,2,3,4] > $SHELL 2>&1

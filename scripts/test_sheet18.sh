#!/bin.bash
params_path="/data/lishichao/project/Foundation-Medical/config/hospital_3/cross_thyroid_clean/test.json"
val_every=1
gpu=1
cross_folds=5  # 设定交叉验证的折数
experiment_name="Thyroid_clean_test_rocauc"  # 设定实验名称

# 设置要更改的工作目录
TARGET_DIR="/data/lishichao/project/Foundation-Medical"

# 使用 cd 命令更改工作目录
cd "$TARGET_DIR" || { echo "Failed to change directory to $TARGET_DIR"; exit 1; }
# 循环进行交叉验证
for (( i=0; i<cross_folds; i++ ))
do
    echo "Running fold $i..."
    SHELL="/data/lishichao/project/Foundation-Medical/shell/hospital_3/${experiment_name}_fold${i}.txt"
    python -m classification --params_path "$params_path" --test --val_every "$val_every" --gpu "$gpu" --cross_folder "$i" > $SHELL 2>&1
done
# do
#     echo "Running fold $i..."
#     SHELL="./shell/test_1002/${experiment_name}_fold${i}.txt"
#     python -m debugpy --listen 5678 --wait-for-client /data/lishichao/project/Foundation-Medical/classification.py --params_path "$params_path" --test --val_every "$val_every" --gpu "$gpu" --cross_folder "$i" > $SHELL 2>&1
# done
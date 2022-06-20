#!/usr/bin/bash

# 一键获取数据

# 数据压缩包
zip_file="road_fighter_car.zip"
# aistudio 数据目录
ais_dir="data148354"
# 解压后的数据目录
sub_data_dir="road_fighter_car"
# 数据目录
data_dir="./dataset/$sub_data_dir"

# 分类标签
labels_txt="car"

# 子目录对应的分类标签
dataset_list=""

# 分类标签文件
labels_file="$data_dir/labels.txt"

# 获取数据
if [ ! -d "$data_dir" ]; then
    bash get_data.sh "$zip_file" "$ais_dir"
fi

# 数据划分 图像分类
# paddlex --split_dataset --format ImageNet --dataset_dir "$data_dir" --val_value 0.2 --test_value 0.1
# 数据划分 目标检测
# paddlex --split_dataset --format VOC --dataset_dir "$data_dir" --val_value 0.2 --test_value 0.1
# 数据划分 实例分割
# paddlex --split_dataset --format COCO --dataset_dir "$data_dir" --val_value 0.2 --test_value 0.1
# 数据划分 语义分割
# paddlex --split_dataset --format SEG --dataset_dir "$data_dir" --val_value 0.2 --test_value 0.1

# 生成分类标签
# echo "$labels_txt">"$labels_file"

# 检查数据
bash check_data.sh "$sub_data_dir"

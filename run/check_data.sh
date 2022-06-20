#!/usr/bin/bash

# 检查 dataset 目录下的数据是否存在
# 参数1：数据目录，例如：train

dataset_dir="./dataset"

# 检查文件
check_files() {
    paths=$@
    for i in $paths; do
        if [ -f "$i" ]; then
            echo "检查文件: $i  --  存在"
        else
            echo "检查文件: $i  --  不存在"
        fi
    done
}

# 检查文件夹
check_dirs() {
    paths=$@
    for i in $paths; do
        if [ -d "$i" ]; then
            echo "检查文件夹: $i  --  存在"
        else
            echo "检查文件夹: $i  --  不存在"
        fi
    done
}

dataset_dir=$dataset_dir/$1

data_files="
$dataset_dir/train_list.txt
$dataset_dir/val_list.txt
$dataset_dir/test_list.txt
$dataset_dir/labels.txt
"

check_files ${data_files[@]}
# check_dirs ${data_dirs[@]}

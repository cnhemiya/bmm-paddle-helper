#!/usr/bin/bash

# 获取数据到 dataset 目录下
# 参数1：数据文件，例如：road_fighter_car.zip
# 参数1：aistudio 数据目录，例如：data148354

dir_list="./dataset ./output ./result"
dataset_dir="./dataset"
ais_dataset_dir="../data"
zip_files=""

# 创建目录
make_dir_list() {
    for i in $dir_list; do
        if [ ! -d "$i" ]; then
            mkdir "$i"
            touch "$i/.keep"
        fi
    done
}

# 解压缩文件
unzip_file() {
    file="$1"
    dir="$2"
    ext="${file##*.}"
    if [ -f "$file" ]; then
        echo "解压文件: $file"
        if [ $ext == "zip" ]; then
            unzip -oq "$file" -d "$dir"
        elif [ $ext == "gz" ]; then
            gzip -dqkfN "$file"
        fi
    fi
}

# 获取数据
get_data() {
    file="$1"
    if [ -f "$dataset_dir/$file" ]; then
        echo "找到文件: $dataset_dir/$file"
        if [ "$2" == "zip" ]; then
            unzip_file "$dataset_dir/$file" "$dataset_dir"
        fi
    elif [ -f "$ais_dataset_dir/$file" ]; then
        echo "找到文件: $ais_dataset_dir/$file"
        if [ "$2" == "zip" ]; then
            unzip_file "$ais_dataset_dir/$file" "$dataset_dir"
        else
            echo "复制文件到: $dataset_dir/$file"
            cp "$ais_dataset_dir/$file" "$dataset_dir/$file"
        fi
    fi
}

# 获取全部压缩文件数据
get_all_zip_data() {
    for i in $zip_files; do
        get_data "$i" "zip"
    done
}

zip_files=$1
ais_dataset_dir=$ais_dataset_dir/$2

make_dir_list
get_all_zip_data


#!/usr/bin/bash

# 目标检测，训练，量化，一键任务脚本

# 模型名称
MODEL="PicoDet"
# backbone 主干模型
BACKBONE="MobileNetV3"
# 数据集目录
DATASET="./dataset/road_fighter_car"
# 保存的目录
BASE_SAVE_DIR="./output/${MODEL}_${BACKBONE}"
# 导出模型的输入大小，默认 None，或者修改[n,c,w,h] --fixed_input_shape=[-1,3,224,224]
# 图像分割，没有 --fixed_input_shape 参数
FIXED_INPUT_SHAPE="--fixed_input_shape=[-1,3,608,608]"

# 程序目录
APP_DIR=run
# python 程序
PYTHON_APP=python3
# 训练程序
TRAIN_APP="$APP_DIR/train.py"
# 量化程序
QUANT_APP="$APP_DIR/quant.py"
# 裁剪程序
PRUNE_APP="$APP_DIR/prune.py"

# 训练轮数
TRAIN_EPOCHS=32
# 训练单批次数量
TRAIN_BATCH_SIZE=1
# 训练学习率
TRAIN_LEARNING_RATE=0.01
# 训练保存间隔轮数
TRAIN_SAVE_INTERVAL_EPOCHS=1
# 训练预加载权重 IMAGENET COCO
TRAIN_PRETRAIN_WEIGHTS=""
# 训练模型保存的目录
TRAIN_SAVE_DIR="$BASE_SAVE_DIR/normal"
# 训练最佳模型保存的目录
TRAIN_BSET_SAVE_DIR="$TRAIN_SAVE_DIR/best_model"

# 量化训练轮数
QUANT_EPOCHS=16
# 量化训练单批次数量
QUANT_BATCH_SIZE=1
# 量化训练学习率
QUANT_LEARNING_RATE=0.001
# 量化训练保存间隔轮数
QUANT_SAVE_INTERVAL_EPOCHS=1
# 量化训练模型读取的目录
QUANT_MODEL_DIR="$TRAIN_BSET_SAVE_DIR"
# 量化训练模型保存的目录
QUANT_SAVE_DIR="$BASE_SAVE_DIR/quant"
# 量化训练最佳模型保存的目录
QUANT_BSET_SAVE_DIR="$QUANT_SAVE_DIR/best_model"

# 裁剪训练轮数
PRUNE_EPOCHS=16
# 裁剪训练单批次数量
PRUNE_BATCH_SIZE=1
# 裁剪训练学习率
PRUNE_LEARNING_RATE=0.001
# 裁剪训练保存间隔轮数
PRUNE_SAVE_INTERVAL_EPOCHS=1
# 每秒浮点数运算次数（FLOPs）的剪裁比例
PRUNE_PRUNED_FLOPS=0.2
# 裁剪训练模型读取的目录
PRUNE_MODEL_DIR="$TRAIN_BSET_SAVE_DIR"
# 裁剪训练模型保存的目录
PRUNE_SAVE_DIR="$BASE_SAVE_DIR/prune"
# 裁剪训练最佳模型保存的目录
PRUNE_BSET_SAVE_DIR="$PRUNE_SAVE_DIR/best_model"

# 裁剪后量化训练轮数
P_Q_EPOCHS=16
# 裁剪后量化训练单批次数量
P_Q_BATCH_SIZE=1
# 裁剪后量化训练学习率
P_Q_LEARNING_RATE=0.001
# 裁剪后量化训练保存间隔轮数
P_Q_SAVE_INTERVAL_EPOCHS=1
# 裁剪后量化训练模型读取的目录
P_Q_MODEL_DIR="$PRUNE_BSET_SAVE_DIR"
# 裁剪后量化训练模型保存的目录
P_Q_SAVE_DIR="$BASE_SAVE_DIR/prune_quant"
# 裁剪后量化训练最佳模型保存的目录
P_Q_BSET_SAVE_DIR="$P_Q_SAVE_DIR/best_model"

# 训练模型压缩文档
TRAIN_ZIP_FILE="${MODEL}_${BACKBONE}_${TRAIN_EPOCHS}e_${TRAIN_LEARNING_RATE}.tar.gz"
# 量化模型压缩文档
QUANT_ZIP_FILE="${MODEL}_${BACKBONE}_${QUANT_EPOCHS}e_${QUANT_LEARNING_RATE}_quant.tar.gz"
# 裁剪模型压缩文档
PRUNE_ZIP_FILE="${MODEL}_${BACKBONE}_${PRUNE_EPOCHS}e_${PRUNE_LEARNING_RATE}_prune.tar.gz"
# 裁剪后量化模型压缩文档
P_Q_ZIP_FILE="${MODEL}_${BACKBONE}_${P_Q_EPOCHS}e_${P_Q_LEARNING_RATE}_prune_quant.tar.gz"

# 训练导出模型目录
TRAIN_INFER_SAVE_DIR="$BASE_SAVE_DIR/normal_infer"
# 量化导出模型目录
QUANT_INFER_SAVE_DIR="$BASE_SAVE_DIR/quant_infer"
# 裁剪导出模型目录
PRUNE_INFER_SAVE_DIR="$BASE_SAVE_DIR/prune_infer"
# 裁剪后量化导出模型目录
P_Q_INFER_SAVE_DIR="$BASE_SAVE_DIR/prune_quant_infer"

# 训练导出模型压缩文档
TRAIN_INFER_ZIP_FILE="${MODEL}_${BACKBONE}_${TRAIN_EPOCHS}e_${TRAIN_LEARNING_RATE}_infer.tar.gz"
# 量化导出模型压缩文档
QUANT_INFER_ZIP_FILE="${MODEL}_${BACKBONE}_${QUANT_EPOCHS}e_${QUANT_LEARNING_RATE}_quant_infer.tar.gz"
# 裁剪导出模型压缩文档
PRUNE_INFER_ZIP_FILE="${MODEL}_${BACKBONE}_${PRUNE_EPOCHS}e_${PRUNE_LEARNING_RATE}_prune_infer.tar.gz"
# 裁剪后量化导出模型压缩文档
P_Q_INFER_ZIP_FILE="${MODEL}_${BACKBONE}_${P_Q_EPOCHS}e_${P_Q_LEARNING_RATE}_prune_quant_infer.tar.gz"

echo "=====  开始训练  ====="
# 训练
$PYTHON_APP $TRAIN_APP --dataset "$DATASET" \
    --epochs $TRAIN_EPOCHS \
    --batch_size $TRAIN_BATCH_SIZE \
    --learning_rate $TRAIN_LEARNING_RATE \
    --model $MODEL \
    --backbone $BACKBONE \
    --save_interval_epochs $TRAIN_SAVE_INTERVAL_EPOCHS \
    --pretrain_weights "$TRAIN_PRETRAIN_WEIGHTS" \
    --save_dir "$TRAIN_SAVE_DIR"

echo "保存并压缩训练模型"
tar -caf "$BASE_SAVE_DIR/$TRAIN_ZIP_FILE" "$TRAIN_BSET_SAVE_DIR"

echo "导出训练模型并压缩"
paddlex --export_inference --model_dir="$TRAIN_BSET_SAVE_DIR" --save_dir="$TRAIN_INFER_SAVE_DIR" $FIXED_INPUT_SHAPE
tar -caf "$BASE_SAVE_DIR/$TRAIN_INFER_ZIP_FILE" "$TRAIN_INFER_SAVE_DIR"

echo "=====  开始量化  ====="
# 量化
$PYTHON_APP $QUANT_APP --dataset "$DATASET" \
    --epochs $QUANT_EPOCHS \
    --batch_size $QUANT_BATCH_SIZE \
    --learning_rate $QUANT_LEARNING_RATE \
    --save_interval_epochs $QUANT_SAVE_INTERVAL_EPOCHS \
    --model_dir "$QUANT_MODEL_DIR" \
    --save_dir "$QUANT_SAVE_DIR"

echo "保存并压缩量化模型"
tar -caf "$BASE_SAVE_DIR/$QUANT_ZIP_FILE" "$QUANT_BSET_SAVE_DIR"

echo "导出量化模型并压缩"
paddlex --export_inference --model_dir="$QUANT_BSET_SAVE_DIR" --save_dir="$QUANT_INFER_SAVE_DIR" $FIXED_INPUT_SHAPE
tar -caf "$BASE_SAVE_DIR/$QUANT_INFER_ZIP_FILE" "$QUANT_INFER_SAVE_DIR"

# echo "=====  开始裁剪  ====="
# # 裁剪
# $PYTHON_APP $PRUNE_APP --dataset "$DATASET" \
#     --epochs $PRUNE_EPOCHS \
#     --batch_size $PRUNE_BATCH_SIZE \
#     --learning_rate $PRUNE_LEARNING_RATE \
#     --save_interval_epochs $PRUNE_SAVE_INTERVAL_EPOCHS \
#     --model_dir "$PRUNE_MODEL_DIR" \
#     --save_dir "$PRUNE_SAVE_DIR" \
#     --pruned_flops $PRUNE_PRUNED_FLOPS

# echo "保存并压缩裁剪模型"
# tar -caf "$BASE_SAVE_DIR/$PRUNE_ZIP_FILE" "$PRUNE_BSET_SAVE_DIR"

# echo "导出裁剪模型并压缩"
# paddlex --export_inference --model_dir="$PRUNE_BSET_SAVE_DIR" --save_dir="$PRUNE_INFER_SAVE_DIR" $FIXED_INPUT_SHAPE
# tar -caf "$BASE_SAVE_DIR/$PRUNE_INFER_ZIP_FILE" "$PRUNE_INFER_SAVE_DIR"

# echo "=====  开始裁剪后量化  ====="
# # 裁剪后量化
# $PYTHON_APP $QUANT_APP --dataset "$DATASET" \
#     --epochs $P_Q_EPOCHS \
#     --batch_size $P_Q_BATCH_SIZE \
#     --learning_rate $P_Q_LEARNING_RATE \
#     --save_interval_epochs $P_Q_SAVE_INTERVAL_EPOCHS \
#     --model_dir "$P_Q_MODEL_DIR" \
#     --save_dir "$P_Q_SAVE_DIR"

# echo "保存并压缩裁剪后量化模型"
# tar -caf "$BASE_SAVE_DIR/$P_Q_ZIP_FILE" "$P_Q_BSET_SAVE_DIR"

# echo "导出裁剪后量化模型并压缩"
# paddlex --export_inference --model_dir="$P_Q_BSET_SAVE_DIR" --save_dir="$P_Q_INFER_SAVE_DIR" $FIXED_INPUT_SHAPE
# tar -caf "$BASE_SAVE_DIR/$P_Q_INFER_ZIP_FILE" "$P_Q_INFER_SAVE_DIR"

echo "=====  结束任务  ====="

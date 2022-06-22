# 苞米面 Paddle 助手

- 自己用的百度飞桨 Paddle，PaddleX 项目模板和小工具

## 适用系统

- 一些脚本使用 shell 编写，所以目前适用 Linux 和 百度 AI Studio

## 如何安装

- 从 gitee 获取源码

```bash
git clone git@gitee.com:cnhemiya/bmm-paddle-helper.git
```

- 从 github 获取源码

```bash
git clone git@github.com:cnhemiya/bmm-paddle-helper.git
```

## 程序参数

- 项目生成器 mkbmmph.py

```bash
cd bmm-paddle-helper
python3 tools/mkbmmph.py -h
usage: mkbmmph.py [-h] [--project] [--to_dir]

苞米面 Paddle 项目生成器

optional arguments:
  -h, --help  show this help message and exit
  --project   项目类型，可选：paddlex_cls, paddlex_det, paddlex_seg
  --to_dir    生成的项目模板保存目录，默认 './run'
```

- 生成不同的模板

```bash
cd bmm-paddle-helper
# PaddleX 图像分类
python3 tools/mkbmmph.py --project paddlex_cls --to_dir ./run
# PaddleX 目标检测
python3 tools/mkbmmph.py --project paddlex_det --to_dir ./run
# PaddleX 图像分割
python3 tools/mkbmmph.py --project paddlex_seg --to_dir ./run
```

- Linux 小技巧

可以使用软连接把 mkbmmph.py 连接到 $HOME/.local/bin 目录，方便使用。

## 使用示例

### 生成模板

- 生成 PaddleX 目标检测项目模板

```bash
cd bmm-paddle-helper
python3 tools/mkbmmph.py --project paddlex_det --to_dir ./run
```

### 模板目录结构

```bash
run
├── aismain.ipynb
├── check_data.sh
├── get_data.sh
├── infer.py
├── mod
│   ├── args.py
│   ├── config.py
│   ├── __init__.py
│   ├── pdxconfig.py
│   ├── report.py
│   └── utils.py
├── onekey.sh
├── onetasks.sh
├── paddlex_det_doc.md
├── prune.py
├── quant.py
└── train.py
```

### 文件说明

|文件|说明|
|:--|:--|
|aismain.ipynb|Jupyter notebook 适用百度 AI Studio，放到项目根目录，或者根据目录修改|
|check_data.sh|检查数据|
|get_data.sh|获取数据|
|infer.py|预测程序|
|mod|python 模块|
|onekey.sh|一键获取数据脚本模板，需要按照自己需求，修改路径|
|onetasks.sh|一键训练，量化脚本模板，需要按照自己需求，修改参数|
|paddlex_det_doc.md|参数说明|
|prune.py|裁剪程序|
|quant.py|量化程序|
|train.py|训练程序|

### aismain.ipynb 目录结构示例

```bash
├── aismain.ipynb
├── data
└── work
    └── run
```

### 训练示例

- train.py 加 -h 查看参数

```bash
python3 run/train.py \
    --dataset ./dataset/road_fighter_car \
    --epochs 32 \
    --batch_size 1 \
    --learning_rate 0.01 \
    --model PicoDet \
    --backbone ESNet_m \
    --pretrain_weights ""
```

### 裁剪示例

- prune.py 加 -h 查看参数

```bash
python3 run/prune.py \
    --dataset ./dataset/road_fighter_car \
    --epochs 16 \
    --batch_size 1 \
    --learning_rate 0.001 \
    --model_dir ./output/best_model \
    --save_dir ./output/prune \
    --pruned_flops 0.2
```

### 在线量化示例

- quant.py 加 -h 查看参数

```bash
python3 run/quant.py \
    --dataset ./dataset/road_fighter_car \
    --epochs 16 \
    --batch_size 1 \
    --learning_rate 0.001 \
    --model_dir ./output/best_model \
    --save_dir ./output/quant
```

## 需要修改什么

### aismain.ipynb

- 放到百度 AI Studio 项目根目录，或者根据目录修改

### check_data.sh

- dataset_dir：需要检查的文件所在的目录
- data_files：需要检查的文件

### onekey.sh

- app_dir：程序目录，如果不是 run，根据自己的设定修改
- zip_file：数据压缩包
- ais_dir：aistudio 数据目录
- sub_data_dir：解压后的数据目录
- data_dir：数据目录

### onetasks.sh

- MODEL：模型名称
- BACKBONE：主干模型
- DATASET：数据集目录
- BASE_SAVE_DIR：保存的目录
- FIXED_INPUT_SHAPE：导出模型的输入大小
- APP_DIR：程序目录
- PYTHON_APP：python 程序

### transforms

- 可以自己修改 train.py prune.py quant.py 的 transforms 非必须。

### dataset

- 可以自己修改 train.py prune.py quant.py 的 dataset 非必须。

## 开源协议

[**MulanPSL-2.0**](http://license.coscl.org.cn/MulanPSL2)

## 项目地址

**GITEE**&nbsp;&nbsp;&nbsp;&nbsp;**https://gitee.com/cnhemiya/bmm-paddle-helper**

**GITHUB**&nbsp;&nbsp;&nbsp;&nbsp;**https://github.com/cnhemiya/bmm-paddle-helper**

#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-14 15:26
文档说明: 图像分类预测
"""


import os
import paddlex as pdx
from paddlex import transforms as T
import mod.config as config
import mod.utils
import mod.args


# 训练 transforms 图像大小
TRAIN_IMAGE_SIZE = 224

# 评估 transforms 图像大小
EVAL_IMAGE_SIZE = 256

# 测试 transforms 图像大小
TEST_IMAGE_SIZE = 224

# 分类 ID Key
CATEGORY_ID_KEY = "category_id"


def main():
    # 解析命令行参数
    args = mod.args.PredictX()
    # 检查文件或目录是否存在
    args.check()
    # 使用 cuda gpu 还是 cpu 运算
    config.user_cude(not args.cpu)

    # 定义训练和验证时的 transforms
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
    infer_transforms = T.Compose([
        T.ResizeByShort(short_size=TRAIN_IMAGE_SIZE),
        T.CenterCrop(crop_size=TRAIN_IMAGE_SIZE),
        T.Normalize()])

    # 数据集解析
    image_paths, _ = mod.utils.parse_dataset(
        args.dataset, dataset_list_path=args.infer_list, shuffle=False)
    # 读取模型
    model = pdx.load_model(args.model_dir)
    # 样本数量
    sample_num = len(image_paths)
    print("预测样本数量：{}".format(sample_num))

    # 开始预测
    print("开始预测 。。。")
    result_lines = []
    for i in range(sample_num):
        image_file = image_paths[i]
        # 分类模型预测接口
        result = model.predict(img_file=image_file,
                               transforms=infer_transforms)
        data = result[0]
        if args.result_info:
            print(image_file)
            print(data)
        result_lines.append("{}{}{}\n".format(
            os.path.basename(image_file), args.split, data[CATEGORY_ID_KEY]))
    with open(args.result_path, "w") as f:
        f.writelines(result_lines)
    print("结束预测 。。。")


if __name__ == '__main__':
    main()

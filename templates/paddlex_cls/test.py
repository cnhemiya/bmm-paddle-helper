#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-12 02:04
文档说明: 测试
"""


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
    args = mod.args.TestX()
    # 检查文件或目录是否存在
    args.check()
    # 使用 cuda gpu 还是 cpu 运算
    config.user_cude(not args.cpu)

    # 定义训练和验证时的 transforms
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
    test_transforms = T.Compose([
        T.ResizeByShort(short_size=TRAIN_IMAGE_SIZE),
        T.CenterCrop(crop_size=TRAIN_IMAGE_SIZE),
        T.Normalize()])

    # 数据集解析
    image_paths, labels = mod.utils.parse_dataset(
        args.dataset, dataset_list_path=args.test_list, inc_label=True, shuffle=False)
    # 读取模型
    model = pdx.load_model(args.model_dir)
    # 样本数量
    sample_num = len(labels)
    # 测试几轮
    test_epochs = args.epochs

    for i in range(test_epochs):
        # 正确数量
        ok_num = 0
        # 错误数量
        err_num = 0
        print("开始测试 。。。第 {} 轮".format(i + 1))
        # 计算结果
        for i in range(len(image_paths)):
            # 分类模型预测接口
            result = model.predict(img_file=image_paths[i],
                                   transforms=test_transforms)
            data = result[0]
            if data[CATEGORY_ID_KEY] == labels[i]:
                ok_num += 1
            else:
                err_num += 1
        print("样本数量: {},  正确率: {:<.5f},  正确样本: {},  错误样本: {}".format(
            sample_num, ok_num/sample_num, ok_num, err_num))
    print("结束测试 。。。")


if __name__ == '__main__':
    main()

#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-06-19 13:52
文档说明: 图像分割预测
"""


import os
import sys
import argparse
import paddlex as pdx


# 可视化颜色，EISeg 颜色通道顺序为 RGB，paddlex.seg.visualize 颜色通道顺序为 BGR
# VISUALIZE_COLOR = [0, 0, 0]        # 背景
VISUALIZE_COLOR = None


def get_arg_parse():
    arg_parse = argparse.ArgumentParser(
        description="读取模型并预测")
    arg_parse.add_argument("--model_dir", default="./output/best_model", dest="model_dir",
                           metavar="", help="读取模型的目录，默认 './output/best_model'")
    arg_parse.add_argument("--predict_image", default="", dest="predict_image",
                           metavar="", help="预测的图像文件")
    arg_parse.add_argument("--predict_image_dir", default="", dest="predict_image_dir",
                           metavar="", help="预测的图像目录")
    arg_parse.add_argument("--weight", default=0.6, dest="weight",
                           metavar="", help="mask可视化结果与原图权重因子，weight表示原图的权重，默认 0.6")
    arg_parse.add_argument("--result_dir", default="./result", dest="result_dir",
                           metavar="", help="预测结果可视化的保存目录，默认 './result'")
    return arg_parse


def infer(args):
    # 读取模型
    model = pdx.load_model(args.model_dir)
    # 预测图像
    predict_image = args.predict_image
    # 预测图像目录
    predict_image_dir = args.predict_image_dir
    # 是预测图像目录
    is_predict_dir = False if predict_image_dir == "" else True
    # score阈值，将Box置信度低于该阈值的框过滤
    weight = args.weight
    # 预测结果可视化的保存目录
    result_dir = args.result_dir
    # 预测图像列表
    predict_image_list = []

    if not is_predict_dir:
        predict_image_list.append(predict_image)
    else:
        imgs = os.listdir(predict_image_dir)
        imgs.sort()
        for i in imgs:
            predict_image_list.append(os.path.join(predict_image_dir, i))

    for predict_image_idx in predict_image_list:
        # 预测结果
        result = model.predict(predict_image_idx)
        # 预测结果可视化
        pdx.seg.visualize(image=predict_image_idx, result=result, weight=weight,
                          save_dir=result_dir, color=VISUALIZE_COLOR)


def main():
    arg_parse = get_arg_parse()
    if (len(sys.argv) < 2):
        arg_parse.print_help()
    else:
        infer(arg_parse.parse_args())


if __name__ == "__main__":
    main()

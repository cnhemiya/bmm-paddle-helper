#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-23 21:29
文档说明: 目标检测预测
"""


import os
import sys
import cv2
import argparse
import paddlex as pdx


def get_arg_parse():
    arg_parse = argparse.ArgumentParser(
        description="读取模型并预测")
    arg_parse.add_argument("--model_dir", default="./output/best_model", dest="model_dir",
                           metavar="", help="读取模型的目录，默认 './output/best_model'")
    arg_parse.add_argument("--predict_image", default="", dest="predict_image",
                           metavar="", help="预测的图像文件")
    arg_parse.add_argument("--predict_image_dir", default="", dest="predict_image_dir",
                           metavar="", help="预测的图像目录，选择后 --result_list，--show_result 失效")
    arg_parse.add_argument("--threshold", default=0.5, dest="threshold",
                           metavar="", help="score阈值，将Box置信度低于该阈值的框过滤，默认 0.5")
    arg_parse.add_argument("--result_list", default="./result/result.txt", dest="result_list",
                           metavar="", help="预测的结果列表文件，默认 './result/result.txt'")
    arg_parse.add_argument("--result_dir", default="./result", dest="result_dir",
                           metavar="", help="预测结果可视化的保存目录，默认 './result'")
    arg_parse.add_argument("--show_result", action="store_true", dest="show_result",
                           help="显示预测结果的图像")
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
    threshold = args.threshold
    # 预测的结果列表文件
    result_list = args.result_list
    # 预测结果可视化的保存目录
    result_dir = args.result_dir

    predict_image_list = []

    if not is_predict_dir:
        predict_image_list.append(predict_image)
    else:
        imgs = os.listdir(predict_image_dir)
        imgs.sort()
        for i in imgs:
            predict_image_list.append(os.path.join(predict_image_dir, i))

    for predict_image_idx in predict_image_list:
        # 读取图像
        img = cv2.imread(predict_image_idx)
        # 预测结果
        result = model.predict(img)
        # 保留的结果
        keep_results = []
        # 面积
        areas = []
        # 写入文件的结果
        result_lines = []
        # 数量计数
        count = 0

        # 遍历结果，过滤
        for det in result:
            cname, bbox, score = det["category"], det["bbox"], det["score"]
            # 结果过滤
            if score >= threshold:
                count += 1
                keep_results.append(det)
                result_lines.append("{}\n".format(str(det)))
                # 面积：宽 * 高
                areas.append(bbox[2] * bbox[3])

        # 面积降序排列
        # areas = np.asarray(areas)
        # sorted_idxs = np.argsort(-areas).tolist()
        # keep_results = [keep_results[k]
        #                 for k in sorted_idxs] if len(keep_results) > 0 else []

        if not is_predict_dir:
            # 符合阈值 threshold 的结果数量
            total_str = "the total number is : {}".format(str(count))
            print(total_str)
            # 写入结果
            with open(result_list, "w") as f:
                f.writelines(result_lines)

        # 预测结果可视化
        pdx.det.visualize(
            predict_image_idx, result, threshold=threshold, save_dir=result_dir)

        if not is_predict_dir:
            # 显示预测结果的图像
            if args.show_result:
                image_path = os.path.join(result_dir, "visualize_{}".format(
                    os.path.basename(predict_image_idx)))
                img = cv2.imread(image_path)
                cv2.imshow("result", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def main():
    arg_parse = get_arg_parse()
    if (len(sys.argv) < 2):
        arg_parse.print_help()
    else:
        infer(arg_parse.parse_args())


if __name__ == "__main__":
    main()

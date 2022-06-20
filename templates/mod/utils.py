# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-08 21:52
文档说明: 杂项
"""


import os
import time
import random
import paddle
import paddle.nn.functional as F


def check_path(path: str, msg="路径错误"):
    """
    检查路径是否存在

    Args:
        path (str): 路径
        msg (str, optional): 异常消息, 默认 "路径错误"

    Raises:
        Exception: 路径错误, 异常
    """
    if not os.path.exists(path):
        raise Exception("{}: {}".format(msg, path))


def time_id(format="%Y-%m-%d_%H-%M-%S"):
    """
    根据时间生成的字符串 ID

    Args:
        format (str, optional): 格式化, 默认 "%Y-%m-%d_%H-%M-%S"

    Returns:
        str: 根据时间生成的字符串 ID
    """
    return time.strftime(format)


def predict_to_class(predict_result):
    """
    预测转分类标签

    Args:
        predict_result (tensor): tensor 数据

    Returns:
        int: 分类标签 id
    """
    result_list = F.softmax(predict_result[0]).tolist()
    result_idx = result_list.index(max(result_list))
    return result_idx


def str_to_list(str_arr: str, astype="int", split=" "):
    """
    数字字符串转列表

    Args:
        str_arr (str): 一组数字字符串，如："3 6 9"
        astype (str, optional): 转为类型, 默认 "int"
        split (str): 数据分割字符，默认 空格

    Raises:
        Exception: astype 错误

    Returns:
        _type_: astype 类型的列表
    """
    astype = astype.lower()

    if astype not in ["int", "float"]:
        raise Exception("astype 错误, 只能为: int, float")

    arr = str_arr.split(split)
    result = []
    for i in arr:
        if (astype == "int"):
            result.append(int(i))
        elif (astype == "float"):
            result.append(float(i))
    return result


def parse_dataset(dataset_path: str, dataset_list_path: str, inc_label: bool, shuffle: bool):
    """
    数据集解析

    Args:
        dataset_path (str): 数据集目录路径
        dataset_list_path (str): 数据集列表文件路径
        inc_label (bool): 包含标签
        shuffle (bool): 随机打乱数据

    Returns:
        image_paths: 图像路径集
        labels: 分类标签集
    """
    lines = []
    lines_n = []
    image_paths = []
    labels = []
    with open(dataset_list_path, "r") as f:
        lines_n = f.readlines()
    # 去掉行尾 \n
    for i in lines_n:
        i = i.rstrip("\n")
        lines.append(i)
    # 随机打乱数据
    if (shuffle):
        random.shuffle(lines)
    for i in lines:
        if inc_label:
            data = i.split(" ")
            if (len(data) < 2):
                raise Exception("数据集解析错误，数据少于 2")
            image_paths.append(os.path.join(dataset_path, data[0]))
            labels.append(int(data[1]))
        else:
            image_paths.append(os.path.join(dataset_path, i))
    return image_paths, labels


def image_to_tensor(image, img_c, img_h, img_w):
    """
    图像数据转 tensor

    Returns:
        tensor: 转换后的 tensor 数据
    """
    # 图像数据格式 CHW
    data = image.reshape([1, img_c, img_h, img_w]).astype("float32")
    return paddle.to_tensor(data)

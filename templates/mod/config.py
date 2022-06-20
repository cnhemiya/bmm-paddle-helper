# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-09 17:47
文档说明: 配置
"""


import os
import paddle
import paddle.vision.transforms as pptf
import mod.utils
import mod.report


# 分类文本, 按照标签排列
CLASS_TXT = ["石头", "剪子", "布"]

# 分类数量
NUM_CLASSES = 3

# 图像通道 彩色 3, 灰度 1
IMAGE_C = 3
# 图像高
IMAGE_H = 224
# 图像宽
IMAGE_W = 224

# 数据集路径
DATASET_PATH = "./dataset/"
# 训练数据
TRAIN_LIST_PATH = "train_list.txt"
# 评估数据
EVAL_LIST_PATH = "val_list.txt"
# 测试数据
TEST_LIST_PATH = "test_list.txt"
# 预测数据
INFER_LIST_PATH = "infer_list.txt"
# 标签数据
LABEL_LIST_PATH = "labels.txt"


# 模型参数保存的文件夹
SAVE_DIR_PATH = "./output/"
# 最佳参数保存的文件夹
SAVE_BAST_DIR = "best"
# 最佳参数保存的路径
SAVE_BEST_PATH = SAVE_DIR_PATH + SAVE_BAST_DIR + "/"
# 模型参数保存的前缀
SAVE_PREFIX = "model"

# 日志保存的路径
LOG_DIR = "./log"

# 报表文件名
REPORT_FILE = "report.json"

# PaddleX 报表文件名
REPORT_X_FILE = "report.json"

# 推理结果路径
INFER_PATH = "./result/"


def transform():
    """
    获取 transform 对数据进行转换

    Returns:
        Compose: 转换数据的操作组合
    """
    # Resize: 调整图像大小, Normalize: 图像归一化处理, Transpose: 转换图像 HWC 转为 CHW
    return pptf.Compose([pptf.Resize(size=[IMAGE_H, IMAGE_W]),
                         pptf.Normalize(mean=[127.5, 127.5, 127.5], std=[
                                        127.5, 127.5, 127.5], data_format='HWC'),
                         pptf.Transpose()])


def image_to_tensor(image):
    """
    图像数据转 tensor

    Returns:
        tensor: 转换后的 tensor 数据
    """
    # 图像数据格式 CHW
    data = image.reshape([1, IMAGE_C, IMAGE_H, IMAGE_W]).astype("float32")
    return paddle.to_tensor(data)


def train_dataset(transform: pptf.Compose):
    """
    获取训练数据集

    Args:
        transform (Compose): 转换数据的操作组合

    Returns:
        ImageClass: ImageClass 图像分类数据集解析
    """
    return mod.dataset.ImageClass(dataset_path=DATASET_PATH, images_labels_txt_path=TRAIN_LIST_PATH, transform=transform)


def test_dataset(transform):
    """
    获取测试数据集

    Args:
        transform (Compose): 转换数据的操作组合

    Returns:
        ImageClass: ImageClass 图像分类数据集解析
    """
    return mod.dataset.ImageClass(dataset_path=DATASET_PATH, images_labels_txt_path=TEST_LIST_PATH, transform=transform)


def get_save_dir(save_dir=SAVE_DIR_PATH, time_id=mod.utils.time_id()):
    """
    获取 模型输出文件夹

    Args:
        save_dir (str, optional): 输出文件夹, 默认 SAVE_DIR
        time_id (str, optional): 根据时间生成的字符串 ID

    Returns:
        str : 输出文件夹
    """
    return os.path.join(save_dir, time_id)


def get_log_dir(log_dir=LOG_DIR, time_id=mod.utils.time_id()):
    """
    获取 VisualDL 日志文件夹

    Args:
        log_dir (str, optional): 日志文件夹, 默认 LOG_DIR
        time_id (str, optional): 根据时间生成的字符串 ID

    Returns:
        str : VisualDL 日志文件夹
    """
    return os.path.join(log_dir, time_id)


def get_result_file(result_dir=INFER_PATH, time_id=mod.utils.time_id()):
    """
    获取预测结果文件

    Args:
        result_dir (str, optional): 预测结果文件夹, 默认 PREDICT_PATH
        time_id (str, optional): 根据时间生成的字符串 ID

    Returns:
        str : 预测结果文件
    """
    return os.path.join(result_dir, time_id + ".txt")


def save_result(data: list, result: list, result_file: str):
    """
    保存预测结果

    Args:
        data (list): 数据列表
        result (list): paddle.Model.predict 结果类表
        result_file (str): 预测结果保存的文件
    """
    lines = []
    for dat, res in zip(data, result):
        pre_idx = res.index(max(res))
        dat = dat[len(DATASET_PATH):]
        lines.append("{} {}\n".format(dat, pre_idx))
    with open(result_file, "w") as f:
        f.writelines(lines)


def save_model(model, save_dir=SAVE_DIR_PATH, time_id=mod.utils.time_id(), save_prefix=SAVE_PREFIX):
    """
    保存模型参数

    Args:
        model (paddle.Model): 网络模型
        save_dir (str, optional): 保存模型的文件夹, 默认 SAVE_DIR
        time_id (str): 根据时间生成的字符串 ID
        save_prefix (str, optional): 保存模型的前缀, 默认 SAVE_PREFIX

    Returns:
        save_path (str): 保存的路径
    """
    save_path = os.path.join(save_dir, time_id)
    print("保存模型参数。。。")
    model.save(os.path.join(save_path, save_prefix))
    print("模型参数保存完毕！")
    return save_path


def load_model(model, loda_dir="", save_prefix=SAVE_PREFIX, reset_optimizer=False):
    """
    读取模型参数

    Args:
        model (paddle.Model): 网络模型
        loda_dir (str, optional): 读取模型的文件夹, 默认 ""
        save_prefix (str, optional): 保存模型的前缀, 默认 SAVE_PREFIX
        reset_optimizer (bool, optional): 重置 optimizer 参数, 默认 False 不重置
    """
    load_path = os.path.join(SAVE_DIR_PATH, loda_dir)
    mod.utils.check_path(load_path)
    load_path = os.path.join(load_path, save_prefix)
    print("读取模型参数。。。")
    model.load(path=load_path, reset_optimizer=reset_optimizer)
    print("模型参数读取完毕！")


def print_num_classes():
    print("分类数量:  {},  分类文本数量:  {}".format(NUM_CLASSES, len(CLASS_TXT)))


def save_report(save_dir: str, id: str, args=None, eval_result=None):
    """
    保存结果报表

    Args:
        save_path (str): 保存的路径
        id (str): 报表 id
        args (_type_, optional): 命令行参数, 默认 None
        eval_result (list, optional): 评估结果, 默认 None

    Raises:
        Exception: eval_result 不能为 None
    """
    if eval_result == None:
        raise Exception("评估结果不能为 None")
    report = mod.report.Report()
    report.id = id
    report.loss = float(eval_result["loss"][0])
    report.acc = float(eval_result["acc"])
    report.epochs = args.epochs
    report.batch_size = args.batch_size
    report.learning_rate = float(args.learning_rate)
    report.save(os.path.join(save_dir, REPORT_FILE))


def save_report_x(save_dir: str, id: str, model: str, args=None):
    """
    保存 PaddleX 结果报表

    Args:
        save_path (str): 保存的路径
        id (str): 报表 id
        args (_type_, optional): 命令行参数, 默认 None
    """
    report = mod.report.ReportX()
    report.id = id
    report.model = model
    report.epochs = args.epochs
    report.batch_size = args.batch_size
    report.learning_rate = float(args.learning_rate)
    report.save(os.path.join(save_dir, REPORT_X_FILE))


def user_cude(cuda=True):
    """
    使用 cuda gpu 还是 cpu 运算

    Args:
        cuda (bool, optional): cuda, 默认 True
    """
    paddle.device.set_device(
        "gpu:0") if cuda else paddle.device.set_device("cpu")

#!/usr/bin/python3
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-06-20 14:36
文档说明: 苞米面 Paddle 项目模板生成器
"""


import os
import sys
import argparse


# 项目类型，可选列表
PROJECT_LIST = ["paddlex_cls", "paddlex_det", "paddlex_seg"]

# 复制文件的列表
COPY_LIST_DIR = "copy_list"
MOD_LIST = os.path.join(COPY_LIST_DIR, "mod_list.txt")
UTILS_LIST = os.path.join(COPY_LIST_DIR, "utils_list.txt")
PADDLEX_CLS_LIST = os.path.join(COPY_LIST_DIR, "paddlex_cls_list.txt")
PADDLEX_DET_LIST = os.path.join(COPY_LIST_DIR, "paddlex_det_list.txt")
PADDLEX_SEG_LIST = os.path.join(COPY_LIST_DIR, "paddlex_seg_list.txt")


def get_arg_parse():
    arg_parse = argparse.ArgumentParser(
        description="苞米面 Paddle 项目生成器")
    arg_parse.add_argument("--project", default="", dest="project",
                           metavar="", help="项目类型，可选：paddlex_cls, paddlex_det, paddlex_seg")
    arg_parse.add_argument("--to_dir", default="./run", dest="to_dir",
                           metavar="", help="生成的项目模板保存目录，默认 './run'")
    return arg_parse


def read_list(file_list: str):
    """
    读取文件列表

    Args:
        file_list (str): 文件列表文件路径

    Returns:
        list: 绝对路径的文件列表
    """
    this_path = os.path.dirname(os.path.realpath(__file__))
    this_path = os.path.realpath(os.path.join(this_path, ".."))
    with open(os.path.join(this_path, file_list), "r") as f:
        lines = f.readlines()
    line_list = []
    for line in lines:
        line = line.rstrip("\n")
        line = line.rstrip("\r")
        if line:
            line_list.append(os.path.join(this_path, line))
    return line_list


def copy_files(file_list: str, to_dir: str):
    """
    复制文件到指定目录

    Args:
        file_list (str): 文件列表文件路径
        to_dir (str): 保存目录
    """
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    lines = read_list(file_list)
    for line in lines:
        os.system("cp -r \"" + line + "\" \"" + to_dir + "\"")


def make(project: str, to_dir: str):
    """
    项目模板生成器

    Args:
        project (str): 项目类型
        to_dir (str): 生成的项目模板保存目录
    """
    project = project.lower()
    if project not in PROJECT_LIST:
        print("项目类型不存在，可选：")
        print(PROJECT_LIST)
        return
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    copy_files(MOD_LIST, os.path.join(to_dir, "mod"))
    copy_files(UTILS_LIST, to_dir)
    if project == "paddlex_cls":
        copy_files(PADDLEX_CLS_LIST, to_dir)
    elif project == "paddlex_det":
        copy_files(PADDLEX_DET_LIST, to_dir)
    elif project == "paddlex_seg":
        copy_files(PADDLEX_SEG_LIST, to_dir)


def main():
    arg_parse = get_arg_parse()
    if (len(sys.argv) < 2):
        arg_parse.print_help()
    else:
        args = arg_parse.parse_args()
        make(args.project, args.to_dir)


if __name__ == "__main__":
    main()

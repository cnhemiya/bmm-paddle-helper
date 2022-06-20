#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-06-05 21:28
文档说明: 图像分割训练
"""


import paddlex as pdx
from paddlex import transforms as T
import mod.utils
import mod.args
import mod.config as config
import mod.pdxconfig as pdxcfg


def train():
    # 解析命令行参数
    args = mod.args.TrainXSeg()
    # 检查文件或目录是否存在
    args.check()
    # 使用 cuda gpu 还是 cpu 运算
    config.user_cude(not args.cpu)

    # 定义训练和验证时的 transforms
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
    train_transforms = T.Compose([
        T.Resize(target_size=512),
        T.RandomHorizontalFlip(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transforms = T.Compose([
        T.Resize(target_size=512),
        T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 定义训练和验证所用的数据集
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/datasets.md
    train_dataset = pdx.datasets.SegDataset(
        data_dir=args.dataset,
        file_list=args.train_list,
        label_list=args.label_list,
        transforms=train_transforms,
        num_workers=args.num_workers,
        shuffle=True)

    eval_dataset = pdx.datasets.SegDataset(
        data_dir=args.dataset,
        file_list=args.eval_list,
        label_list=args.label_list,
        transforms=eval_transforms,
        num_workers=args.num_workers,
        shuffle=False)

    # 分类数量
    num_classes = len(train_dataset.labels)
    # 获取 PaddleX 模型
    model, model_name = pdxcfg.pdx_seg_model(model_name=args.model,
                                             num_classes=num_classes,
                                             backbone=args.backbone,
                                             hrnet_width=args.hrnet_width,
                                             use_mixed_loss=args.use_mixed_loss,
                                             align_corners=args.align_corners)

    # 优化器
    # https://gitee.com/paddlepaddle/PaddleX/blob/develop/paddlex/cv/models/segmenter.py#L189

    # 模型训练
    # API说明：https://gitee.com/paddlepaddle/PaddleX/blob/develop/docs/apis/models/semantic_segmentation.md
    # 参数调整：https://gitee.com/paddlepaddle/PaddleX/blob/develop/docs/parameters.md
    # 可使用 VisualDL 查看训练指标，参考：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/visualdl.md
    print("开始训练 。。。模型：{}".format(model_name))
    model.train(num_epochs=args.epochs,
                train_dataset=train_dataset,
                train_batch_size=args.batch_size,
                eval_dataset=eval_dataset,
                save_dir=args.save_dir,
                save_interval_epochs=args.save_interval_epochs,
                log_interval_steps=args.log_interval_steps,
                learning_rate=args.learning_rate,
                lr_decay_power=args.lr_decay_power,
                early_stop=args.early_stop,
                early_stop_patience=args.early_stop_patience,
                resume_checkpoint=args.resume_checkpoint,
                pretrain_weights=args.pretrain_weights,
                use_vdl=True)
    print("结束训练 。。。模型：{}".format(model_name))


def main():
    # 解析命令行参数
    args = mod.args.TrainXSeg()
    # PaddleX 模型名称
    if (args.model_list):
        pdxcfg.print_pdx_seg_model_name()
    else:
        # 训练
        train()


if __name__ == '__main__':
    main()

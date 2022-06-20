#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-09 19:16
文档说明: 图像分类训练
"""


import paddlex as pdx
from paddlex import transforms as T
import mod.utils
import mod.config as config
import mod.args
import mod.pdxconfig as pdxcfg


# 训练 transforms 图像大小
TRAIN_IMAGE_SIZE = 224

# 评估 transforms 图像大小
EVAL_IMAGE_SIZE = 256

# 测试 transforms 图像大小
TEST_IMAGE_SIZE = 224


def train():
    # 解析命令行参数
    args = mod.args.TrainX()
    # 检查文件或目录是否存在
    args.check()
    # 使用 cuda gpu 还是 cpu 运算
    config.user_cude(not args.cpu)

    # 定义训练和验证时的 transforms
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
    train_transforms = T.Compose([
        T.RandomCrop(crop_size=TRAIN_IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.Normalize()])

    eval_transforms = T.Compose([
        T.ResizeByShort(short_size=EVAL_IMAGE_SIZE),
        T.CenterCrop(crop_size=TRAIN_IMAGE_SIZE),
        T.Normalize()])

    # 定义训练和验证所用的数据集
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/datasets.md
    train_dataset = pdx.datasets.ImageNet(
        data_dir=args.dataset,
        file_list=args.train_list,
        label_list=args.label_list,
        transforms=train_transforms,
        num_workers=args.num_workers,
        shuffle=True)

    eval_dataset = pdx.datasets.ImageNet(
        data_dir=args.dataset,
        file_list=args.eval_list,
        label_list=args.label_list,
        transforms=eval_transforms,
        num_workers=args.num_workers,
        shuffle=False)

    # 分类数量
    num_classes = len(train_dataset.labels)
    # 获取 PaddleX 模型
    model, model_name = pdxcfg.pdx_cls_model(
        model_name=args.model, num_classes=num_classes)

    # 优化器
    # https://gitee.com/paddlepaddle/PaddleX/blob/develop/paddlex/cv/models/classifier.py#L147
    optimizer = None
    if args.opti_scheduler != "auto":
        optimizer = model.default_optimizer(parameters=model.net.parameters(),
                                            learning_rate=args.learning_rate,
                                            warmup_steps=args.warmup_steps,
                                            warmup_start_lr=args.warmup_start_lr,
                                            lr_decay_epochs=args.lr_decay_epochs,
                                            lr_decay_gamma=args.lr_decay_gamma,
                                            num_steps_each_epoch=len(
                                                train_dataset),
                                            reg_coeff=args.opti_reg_coeff,
                                            scheduler=args.opti_scheduler,
                                            num_epochs=args.epochs
                                            )

    # 模型训练
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/classification.md
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
                warmup_steps=args.warmup_steps,
                warmup_start_lr=args.warmup_start_lr,
                lr_decay_epochs=args.lr_decay_epochs,
                lr_decay_gamma=args.lr_decay_gamma,
                use_ema=args.use_ema,
                early_stop=args.early_stop,
                early_stop_patience=args.early_stop_patience,
                resume_checkpoint=args.resume_checkpoint,
                pretrain_weights=args.pretrain_weights,
                optimizer=optimizer,
                use_vdl=True)
    print("结束训练 。。。模型：{}".format(model_name))


def main():
    # 解析命令行参数
    args = mod.args.TrainX()
    # PaddleX 模型名称
    if (args.model_list):
        pdxcfg.print_pdx_cls_model_name()
    else:
        # 训练
        train()


if __name__ == '__main__':
    main()

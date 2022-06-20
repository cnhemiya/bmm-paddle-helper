#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-25 15:41
文档说明: 图像分类裁剪
"""


import paddlex as pdx
from paddlex import transforms as T
import mod.utils
import mod.args
import mod.config as config


def prune():
    # 解析命令行参数
    args = mod.args.PruneX()
    # 检查文件或目录是否存在
    args.check()
    # 使用 cuda gpu 还是 cpu 运算
    config.user_cude(not args.cpu)

    # 定义训练和验证时的 transforms
    # API说明：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
    train_transforms = T.Compose([
        T.RandomCrop(crop_size=224),
        T.RandomHorizontalFlip(),
        T.Normalize()])

    eval_transforms = T.Compose([
        T.ResizeByShort(short_size=256),
        T.CenterCrop(crop_size=224),
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

    # 加载模型
    print("读取模型 。。。读取路径：{}".format(args.model_dir))
    model = pdx.load_model(args.model_dir)

    # Step 1/3: 分析模型各层参数在不同的裁剪比例下的敏感度
    # 注意：目标检测模型的裁剪依赖PaddleSlim 2.1.0
    # 注意：如果之前运行过该步骤，第二次运行时会自动加载已有的 'save_dir'/model.sensi.data，不再进行敏感度分析
    # API说明：https://gitee.com/paddlepaddle/PaddleX/blob/develop/docs/apis/models/classification.md#analyze_sensitivity
    # 使用参考：https://gitee.com/paddlepaddle/PaddleX/tree/develop/tutorials/slim/prune/image_classification
    if not args.skip_analyze:
        print("敏感度分析 。。。保存路径：{}".format(args.save_dir))
        model.analyze_sensitivity(
            dataset=eval_dataset,
            batch_size=args.batch_size,
            save_dir=args.save_dir)

    # Step 2/3: 根据选择的FLOPs减小比例对模型进行裁剪
    # API说明：https://gitee.com/paddlepaddle/PaddleX/blob/develop/docs/apis/models/classification.md#prune
    # 使用参考：https://gitee.com/paddlepaddle/PaddleX/tree/develop/tutorials/slim/prune/image_classification
    print("对模型进行裁剪 。。。FLOPS：{}".format(args.pruned_flops))
    model.prune(pruned_flops=args.pruned_flops)

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
    # API说明：https://gitee.com/paddlepaddle/PaddleX/blob/develop/docs/apis/models/classification.md
    # 参数调整：https://gitee.com/paddlepaddle/PaddleX/blob/develop/docs/parameters.md
    # 可使用 VisualDL 查看训练指标，参考：https://gitee.com/PaddlePaddle/PaddleX/blob/develop/docs/visualdl.md
    print("开始训练 。。。保存路径：{}".format(args.save_dir))
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
                early_stop=args.early_stop,
                early_stop_patience=args.early_stop_patience,
                resume_checkpoint=args.resume_checkpoint,
                pretrain_weights=args.pretrain_weights,
                optimizer=optimizer,
                use_vdl=True)
    print("结束训练 。。。保存路径：{}".format(args.save_dir))


def main():
    # 裁剪
    prune()


if __name__ == '__main__':
    main()

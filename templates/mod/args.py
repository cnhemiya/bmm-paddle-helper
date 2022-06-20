# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-15 16:54
文档说明: 命令行参数解析
"""


import os
import argparse
import mod.utils
import mod.config as config


class Train():
    """
    返回训练命令行参数
    """

    def __init__(self, args=None) -> None:
        self.args = self.parse() if args == None else args
        self.cpu = args.cpu
        self.num_workers = args.num_workers
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.no_save = args.no_save
        self.load_dir = args.load_dir
        self.log = args.log
        self.summary = args.summary

    def parse(self):
        """
        返回命令行参数

        Returns:
            argparse: 命令行参数
        """
        arg_parse = argparse.ArgumentParser()
        arg_parse.add_argument("--cpu", action="store_true",
                               dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
        arg_parse.add_argument("--num_workers", type=int, default=2,
                               dest="num_workers", metavar="", help="线程数量，默认 2")
        arg_parse.add_argument("--learning_rate", type=float, default=0.001,
                               dest="learning_rate", metavar="", help="学习率，默认 0.001")
        arg_parse.add_argument("--epochs", type=int, default=2,
                               dest="epochs", metavar="", help="训练几轮，默认 2 轮")
        arg_parse.add_argument("--batch_size", type=int, default=2,
                               dest="batch_size", metavar="", help="一批次数量，默认 2")
        arg_parse.add_argument("--no_save", action="store_true",
                               dest="no_save", help="是否保存模型参数，默认保存, 选择后不保存模型参数")
        arg_parse.add_argument("--load_dir", dest="load_dir", default="",
                               metavar="", help="读取模型参数，读取 params 目录下的子文件夹, 默认不读取")
        arg_parse.add_argument("--log", action="store_true",
                               dest="log", help="是否输出 VisualDL 日志，默认不输出")
        arg_parse.add_argument("--summary", action="store_true",
                               dest="summary", help="输出网络模型信息，默认不输出，选择后只输出信息，不会开启训练")
        return arg_parse.parse_args()


class Test():
    """
    返回测试命令行参数
    """

    def __init__(self, args=None) -> None:
        self.args = self.parse() if args == None else args
        self.cpu = args.cpu
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.load_dir = args.load_dir

    def parse(self):
        """
        返回命令行参数

        Returns:
            argparse: 命令行参数
        """
        arg_parse = argparse.ArgumentParser()
        arg_parse.add_argument("--cpu", action="store_true",
                               dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
        arg_parse.add_argument("--num_workers", type=int, default=2,
                               dest="num_workers", metavar="", help="线程数量，默认 2")
        arg_parse.add_argument("--batch_size", type=int, default=2,
                               dest="batch_size", metavar="", help="一批次数量，默认 2")
        arg_parse.add_argument("--load_dir", dest="load_dir", default="best",
                               metavar="", help="读取模型参数，读取 params 目录下的子文件夹, 默认 best 目录")
        return arg_parse.parse_args()


class Predict():
    """
    返回预测命令行参数
    """

    def __init__(self, args=None) -> None:
        self.args = self.parse() if args == None else args
        self.cpu = args.cpu
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.load_dir = args.load_dir
        self.no_save = args.no_save

    def parse(self):
        """
        返回命令行参数

        Returns:
            argparse: 命令行参数
        """
        arg_parse = argparse.ArgumentParser()
        arg_parse.add_argument("--cpu", action="store_true",
                               dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
        arg_parse.add_argument("--num_workers", type=int, default=2,
                               dest="num_workers", metavar="", help="线程数量，默认 2")
        arg_parse.add_argument("--batch_size", type=int, default=1,
                               dest="batch_size", metavar="", help="一批次数量，默认 1")
        arg_parse.add_argument("--load_dir", dest="load_dir", default="best",
                               metavar="", help="读取模型参数，读取 params 目录下的子文件夹, 默认 best 目录")
        arg_parse.add_argument("--no_save", action="store_true",
                               dest="no_save", help="是否保存预测结果，默认保存, 选择后不保存预测结果")
        return arg_parse.parse_args()


class BaseArgsX():
    """
    返回 PaddleX 基本命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 train_list_path=config.TRAIN_LIST_PATH,
                 eval_list_path=config.EVAL_LIST_PATH,
                 label_list_path=config.LABEL_LIST_PATH,
                 save_dir_path=config.SAVE_DIR_PATH):
        self._dataset_path = dataset_path
        self._train_list_path = train_list_path
        self._eval_list_path = eval_list_path
        self._label_list_path = label_list_path
        self._save_dir_path = save_dir_path

        self._arg_parse = argparse.ArgumentParser()
        self._add_argument()
        self.args = self._arg_parse.parse_args() if args == None else args

        self.cpu = self.args.cpu
        self.num_workers = "auto" if self.args.num_workers == 0 else self.args.num_workers

        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.learning_rate
        self.early_stop = self.args.early_stop
        self.early_stop_patience = self.args.early_stop_patience
        self.save_interval_epochs = self.args.save_interval_epochs
        self.log_interval_steps = self.args.log_interval_steps
        self.resume_checkpoint = self.args.resume_checkpoint
        self.save_dir = save_dir_path if self.args.save_dir == "" else self.args.save_dir
        self.dataset = dataset_path if self.args.dataset == "" else self.args.dataset

        self.train_list = os.path.join(
            self.dataset, train_list_path) if self.args.train_list == "" else self.args.train_list
        self.eval_list = os.path.join(
            self.dataset, eval_list_path) if self.args.eval_list == "" else self.args.eval_list
        self.label_list = os.path.join(
            self.dataset, label_list_path) if self.args.label_list == "" else self.args.label_list

    def _add_argument(self):
        """
        添加命令行参数
        """
        self._arg_parse.add_argument("--cpu", action="store_true",
                                     dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
        self._arg_parse.add_argument("--num_workers", type=int, default=0,
                                     dest="num_workers", metavar="", help="线程数量，默认 auto，为CPU核数的一半")
        self._arg_parse.add_argument("--epochs", type=int, default=4,
                                     dest="epochs", metavar="", help="训练几轮，默认 4 轮")
        self._arg_parse.add_argument("--batch_size", type=int, default=16,
                                     dest="batch_size", metavar="", help="一批次数量，默认 16")
        self._arg_parse.add_argument("--learning_rate", type=float, default=0.025,
                                     dest="learning_rate", metavar="", help="学习率，默认 0.025")
        self._arg_parse.add_argument("--early_stop", action="store_true",
                                     dest="early_stop", help="是否使用提前终止训练策略。默认为 False")
        self._arg_parse.add_argument("--early_stop_patience", type=int, default=5,
                                     dest="early_stop_patience", metavar="", help="当使用提前终止训练策略时，如果验证集精度在" +
                                     "early_stop_patience 个 epoch 内连续下降或持平，则终止训练。默认为 5")
        self._arg_parse.add_argument("--save_interval_epochs", type=int, default=1,
                                     dest="save_interval_epochs", metavar="", help="模型保存间隔(单位: 迭代轮数)。默认为 1")
        self._arg_parse.add_argument("--log_interval_steps", type=int, default=10,
                                     dest="log_interval_steps", metavar="", help="训练日志输出间隔（单位：迭代次数）。默认为 10")
        self._arg_parse.add_argument("--resume_checkpoint", dest="resume_checkpoint", default="",
                                     metavar="", help="恢复训练时指定上次训练保存的模型路径, 默认不会恢复训练")
        self._arg_parse.add_argument("--save_dir", dest="save_dir", default="{}".format(self._save_dir_path),
                                     metavar="", help="模型保存路径。默认为 {}".format(self._save_dir_path))
        self._arg_parse.add_argument("--dataset", dest="dataset", default="",
                                     metavar="", help="数据集目录，默认 {}".format(self._dataset_path))
        self._arg_parse.add_argument("--train_list", dest="train_list", default="", metavar="",
                                     help="训练集列表，默认 '--dataset' 参数目录下的 {}".format(self._train_list_path))
        self._arg_parse.add_argument("--eval_list", dest="eval_list", default="", metavar="",
                                     help="评估集列表，默认 '--dataset' 参数目录下的 {}".format(self._eval_list_path))
        self._arg_parse.add_argument("--label_list", dest="label_list", default="", metavar="",
                                     help="分类标签列表，默认 '--dataset' 参数目录下的 {}".format(self._label_list_path))

    def check(self):
        mod.utils.check_path(self.dataset)
        mod.utils.check_path(self.train_list)
        mod.utils.check_path(self.eval_list)
        mod.utils.check_path(self.label_list)

        # 恢复训练时指定上次训练保存的模型路径
        self.resume_checkpoint = None
        # 恢复训练
        if (self.args.resume_checkpoint != ""):
            mod.utils.check_path(self.args.resume_checkpoint)
            self.resume_checkpoint = self.args.resume_checkpoint


class BaseTrainX(BaseArgsX):
    """
    返回 PaddleX 基本训练命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 train_list_path=config.TRAIN_LIST_PATH,
                 eval_list_path=config.EVAL_LIST_PATH,
                 label_list_path=config.LABEL_LIST_PATH,
                 save_dir_path=config.SAVE_DIR_PATH):
        super(BaseTrainX, self).__init__(args=args,
                                         dataset_path=dataset_path,
                                         train_list_path=train_list_path,
                                         eval_list_path=eval_list_path,
                                         label_list_path=label_list_path,
                                         save_dir_path=save_dir_path)
        self.warmup_steps = self.args.warmup_steps
        self.warmup_start_lr = self.args.warmup_start_lr
        self.lr_decay_epochs = mod.utils.str_to_list(self.args.lr_decay_epochs)
        self.lr_decay_gamma = self.args.lr_decay_gamma
        self.use_ema = self.args.use_ema

        self.opti_scheduler = self.args.opti_scheduler.lower()
        self.opti_reg_coeff = self.args.opti_reg_coeff
        if self.opti_scheduler == "cosine" and self.opti_reg_coeff == 0.0:
            self.opti_reg_coeff = 4e-05
        elif self.opti_scheduler == "piecewise" and self.opti_reg_coeff == 0.0:
            self.opti_reg_coeff = 1e-04

    def _add_argument(self):
        """
        添加命令行参数
        """
        super(BaseTrainX, self)._add_argument()
        self._arg_parse.add_argument("--warmup_steps", type=int, default=0,
                                     dest="warmup_steps", metavar="", help="默认优化器的 warmup 步数，学习率将在设定的步数内，" +
                                     "从 warmup_start_lr 线性增长至设定的 learning_rate，默认为 0")
        self._arg_parse.add_argument("--warmup_start_lr", type=float, default=0.0,
                                     dest="warmup_start_lr", metavar="", help="默认优化器的 warmup 起始学习率，默认为 0.0")
        self._arg_parse.add_argument("--lr_decay_epochs", dest="lr_decay_epochs", default="30 60 90",
                                     metavar="", help="默认优化器的学习率衰减轮数。默认为 30 60 90")
        self._arg_parse.add_argument("--lr_decay_gamma", type=float, default=0.1,
                                     dest="lr_decay_gamma", metavar="", help="默认优化器的学习率衰减率。默认为 0.1")
        self._arg_parse.add_argument("--use_ema", action="store_true",
                                     dest="use_ema", help="是否使用指数衰减计算参数的滑动平均值。默认为 False")
        self._arg_parse.add_argument("--opti_scheduler", dest="opti_scheduler", default="auto",
                                     metavar="", help="优化器的调度器，默认 auto，可选 auto，cosine，piecewise")
        self._arg_parse.add_argument("--opti_reg_coeff", type=float, default=0.0,
                                     dest="opti_reg_coeff", metavar="", help="优化器衰减系数，" +
                                     "如果 opti_scheduler 是 Cosine，默认是 4e-05，" +
                                     "如果 opti_scheduler 是 Piecewise，默认是 1e-04")

    def check(self):
        super(BaseTrainX, self).check()
        if self.opti_scheduler not in ["auto", "cosine", "piecewise"]:
            raise Exception("优化器的调度器只能是 auto，cosine，piecewise，错误信息：{}"
                            .format(self.opti_scheduler))


class TrainX(BaseTrainX):
    """
    返回 PaddleX 训练命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 train_list_path=config.TRAIN_LIST_PATH,
                 eval_list_path=config.EVAL_LIST_PATH,
                 label_list_path=config.LABEL_LIST_PATH,
                 save_dir_path=config.SAVE_DIR_PATH):
        super(TrainX, self).__init__(args=args,
                                     dataset_path=dataset_path,
                                     train_list_path=train_list_path,
                                     eval_list_path=eval_list_path,
                                     label_list_path=label_list_path,
                                     save_dir_path=save_dir_path)
        self.pretrain_weights = self.args.pretrain_weights
        self.model = self.args.model
        self.model_list = self.args.model_list

    def _add_argument(self):
        super(TrainX, self)._add_argument()
        self._arg_parse.add_argument("--pretrain_weights", dest="pretrain_weights", default="",
                                     metavar="", help="若指定为'.pdparams'文件时，从文件加载模型权重；" +
                                     "若为字符串’IMAGENET’，则自动下载在ImageNet图片数据上预训练的模型权重；" +
                                     "若为字符串’COCO’，则自动下载在COCO数据集上预训练的模型权重；" +
                                     "若为None，则不使用预训练模型。默认为'IMAGENET'")
        self._arg_parse.add_argument("--model", dest="model", default="",
                                     metavar="", help="PaddleX 模型名称")
        self._arg_parse.add_argument("--model_list", action="store_true", dest="model_list",
                                     help="输出 PaddleX 模型名称，默认不输出，选择后只输出信息，不会开启训练")

    def check(self):
        super(TrainX, self).check()
        # 模型权重
        self.pretrain_weights = "IMAGENET"
        # 加载模型权重
        if (self.args.pretrain_weights == ""):
            self.pretrain_weights = None
        elif self.args.pretrain_weights == "IMAGENET":
            self.pretrain_weights = "IMAGENET"
        elif self.args.pretrain_weights == "COCO":
            self.pretrain_weights = "COCO"
        else:
            mod.utils.check_path(self.args.pretrain_weights)
            self.pretrain_weights = self.args.pretrain_weights
        # 恢复训练
        if (self.args.resume_checkpoint != ""):
            self.pretrain_weights = None


class TrainXCls(TrainX):
    """
    返回 PaddleX 图像分类训练命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 train_list_path=config.TRAIN_LIST_PATH,
                 eval_list_path=config.EVAL_LIST_PATH,
                 label_list_path=config.LABEL_LIST_PATH,
                 save_dir_path=config.SAVE_DIR_PATH):
        super(TrainXCls, self).__init__(args=args,
                                        dataset_path=dataset_path,
                                        train_list_path=train_list_path,
                                        eval_list_path=eval_list_path,
                                        label_list_path=label_list_path,
                                        save_dir_path=save_dir_path)


class TrainXDet(TrainX):
    """
    返回 PaddleX 目标检测训练命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 train_list_path=config.TRAIN_LIST_PATH,
                 eval_list_path=config.EVAL_LIST_PATH,
                 label_list_path=config.LABEL_LIST_PATH,
                 save_dir_path=config.SAVE_DIR_PATH):
        super(TrainXDet, self).__init__(args=args,
                                        dataset_path=dataset_path,
                                        train_list_path=train_list_path,
                                        eval_list_path=eval_list_path,
                                        label_list_path=label_list_path,
                                        save_dir_path=save_dir_path)
        self.backbone = self.args.backbone

    def _add_argument(self):
        super(TrainXDet, self)._add_argument()
        self._arg_parse.add_argument("--backbone", dest="backbone", default="",
                                     metavar="", help="目标检测模型的 backbone 网络")


class PruneX(BaseTrainX):
    """
    返回 PaddleX 模型裁剪命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 train_list_path=config.TRAIN_LIST_PATH,
                 eval_list_path=config.EVAL_LIST_PATH,
                 label_list_path=config.LABEL_LIST_PATH,
                 save_dir_path=config.SAVE_DIR_PATH):
        super(PruneX, self).__init__(args=args,
                                     dataset_path=dataset_path,
                                     train_list_path=train_list_path,
                                     eval_list_path=eval_list_path,
                                     label_list_path=label_list_path,
                                     save_dir_path=save_dir_path)
        self.model_dir = self.args.model_dir
        self.skip_analyze = self.args.skip_analyze
        self.pruned_flops = self.args.pruned_flops

    def _add_argument(self):
        super(PruneX, self)._add_argument()
        self._arg_parse.add_argument("--model_dir", dest="model_dir", default="{}"
                                     .format(os.path.join(self._save_dir_path, "best_model")),
                                     metavar="", help="模型读取路径。默认为 {}"
                                     .format(os.path.join(self._save_dir_path, "best_model")))
        self._arg_parse.add_argument("--skip_analyze", action="store_true",
                                     dest="skip_analyze", help="是否跳过分析模型各层参数在不同的裁剪比例下的敏感度，默认不跳过")
        self._arg_parse.add_argument("--pruned_flops", type=float, default=0.2,
                                     dest="pruned_flops", metavar="", help="根据选择的 FLOPS 减小比例对模型进行裁剪。默认为 0.2")

    def check(self):
        super(PruneX, self).check()
        mod.utils.check_path(self.model_dir)


class QuantX(BaseTrainX):
    """
    返回 PaddleX 模型在线量化命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 train_list_path=config.TRAIN_LIST_PATH,
                 eval_list_path=config.EVAL_LIST_PATH,
                 label_list_path=config.LABEL_LIST_PATH,
                 save_dir_path=config.SAVE_DIR_PATH):
        super(QuantX, self).__init__(args=args,
                                     dataset_path=dataset_path,
                                     train_list_path=train_list_path,
                                     eval_list_path=eval_list_path,
                                     label_list_path=label_list_path,
                                     save_dir_path=save_dir_path)
        self.model_dir = self.args.model_dir

    def _add_argument(self):
        super(QuantX, self)._add_argument()
        self._arg_parse.add_argument("--model_dir", dest="model_dir", default="{}"
                                     .format(os.path.join(self._save_dir_path, "best_model")),
                                     metavar="", help="模型读取路径。默认为 {}"
                                     .format(os.path.join(self._save_dir_path, "best_model")))

    def check(self):
        super(QuantX, self).check()
        mod.utils.check_path(self.model_dir)


class BaseTrainXSeg(BaseArgsX):
    """
    返回 PaddleX 图像分割命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 train_list_path=config.TRAIN_LIST_PATH,
                 eval_list_path=config.EVAL_LIST_PATH,
                 label_list_path=config.LABEL_LIST_PATH,
                 save_dir_path=config.SAVE_DIR_PATH):
        super(BaseTrainXSeg, self).__init__(args=args,
                                            dataset_path=dataset_path,
                                            train_list_path=train_list_path,
                                            eval_list_path=eval_list_path,
                                            label_list_path=label_list_path,
                                            save_dir_path=save_dir_path)
        self.lr_decay_power = self.args.lr_decay_power

    def _add_argument(self):
        super(BaseTrainXSeg, self)._add_argument()
        self._arg_parse.add_argument("--lr_decay_power", type=float, default=0.9,
                                     dest="lr_decay_power", metavar="", help="默认优化器学习率衰减指数。默认为 0.9")


class TrainXSeg(BaseTrainXSeg):
    """
    返回 PaddleX 图像分割训练命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 train_list_path=config.TRAIN_LIST_PATH,
                 eval_list_path=config.EVAL_LIST_PATH,
                 label_list_path=config.LABEL_LIST_PATH,
                 save_dir_path=config.SAVE_DIR_PATH):
        super(TrainXSeg, self).__init__(args=args,
                                        dataset_path=dataset_path,
                                        train_list_path=train_list_path,
                                        eval_list_path=eval_list_path,
                                        label_list_path=label_list_path,
                                        save_dir_path=save_dir_path)
        self.use_mixed_loss = self.args.use_mixed_loss
        self.align_corners = self.args.align_corners
        self.backbone = self.args.backbone
        self.hrnet_width = self.args.hrnet_width
        self.pretrain_weights = self.args.pretrain_weights
        self.model = self.args.model
        self.model_list = self.args.model_list

    def _add_argument(self):
        super(TrainXSeg, self)._add_argument()
        self._arg_parse.add_argument("--use_mixed_loss", action="store_true", dest="use_mixed_loss",
                                     help="是否使用混合损失函数。如果为True，混合使用CrossEntropyLoss和LovaszSoftmaxLoss，" +
                                     "权重分别为0.8和0.2。如果为False，则仅使用CrossEntropyLoss。" +
                                     "也可以以列表的形式自定义混合损失函数，列表的每一个元素为(损失函数类型，权重)元组，" +
                                     "损失函数类型取值范围为['CrossEntropyLoss', 'DiceLoss', 'LovaszSoftmaxLoss']。默认为False。")
        self._arg_parse.add_argument("--align_corners", action="store_true", dest="align_corners",
                                     help="是网络中对特征图进行插值时是否将四个角落像素的中心对齐。" +
                                     "若特征图尺寸为偶数，建议设为True。若特征图尺寸为奇数，建议设为False。默认为False。")
        self._arg_parse.add_argument("--backbone", dest="backbone", default="ResNet50_vd",
                                     metavar="", help="图像分割模型 DeepLabV3P 的 backbone 网络，" +
                                     "取值范围为['ResNet50_vd', 'ResNet101_vd']，默认为'ResNet50_vd'。")
        self._arg_parse.add_argument("--hrnet_width", type=int, default=48, dest="hrnet_width",
                                     metavar="", help="图像分割模型 HRNet 的 width 网络，" +
                                     "高分辨率分支中特征层的通道数量。默认为48。可选择取值为[18, 48]。")
        self._arg_parse.add_argument("--pretrain_weights", dest="pretrain_weights", default="CITYSCAPES",
                                     metavar="", help="若指定为'.pdparams'文件时，则从文件加载模型权重；" +
                                     "若为字符串'CITYSCAPES'，则自动下载在CITYSCAPES图片数据上预训练的模型权重；" +
                                     "若为字符串'PascalVOC'，则自动下载在PascalVOC图片数据上预训练的模型权重；" +
                                     "若为字符串'IMAGENET'，则自动下载在ImageNet图片数据上预训练的模型权重；" +
                                     "若为None，则不使用预训练模型。默认为'CITYSCAPES'。")
        self._arg_parse.add_argument("--model", dest="model", default="",
                                     metavar="", help="PaddleX 模型名称")
        self._arg_parse.add_argument("--model_list", action="store_true", dest="model_list",
                                     help="输出 PaddleX 模型名称，默认不输出，选择后只输出信息，不会开启训练")

    def check(self):
        super(TrainXSeg, self).check()
        # 模型权重
        self.pretrain_weights = "CITYSCAPES"
        # 加载模型权重
        if (self.args.pretrain_weights == ""):
            self.pretrain_weights = None
        elif self.args.pretrain_weights == "CITYSCAPES":
            self.pretrain_weights = "CITYSCAPES"
        elif self.args.pretrain_weights == "PascalVOC":
            self.pretrain_weights = "PascalVOC"
        elif self.args.pretrain_weights == "IMAGENET":
            self.pretrain_weights = "IMAGENET"
        else:
            mod.utils.check_path(self.args.pretrain_weights)
            self.pretrain_weights = self.args.pretrain_weights
        # 恢复训练
        if (self.args.resume_checkpoint != ""):
            self.pretrain_weights = None


class PruneXSeg(BaseTrainXSeg):
    """
    返回 PaddleX 图像分割模型裁剪命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 train_list_path=config.TRAIN_LIST_PATH,
                 eval_list_path=config.EVAL_LIST_PATH,
                 label_list_path=config.LABEL_LIST_PATH,
                 save_dir_path=config.SAVE_DIR_PATH):
        super(PruneXSeg, self).__init__(args=args,
                                        dataset_path=dataset_path,
                                        train_list_path=train_list_path,
                                        eval_list_path=eval_list_path,
                                        label_list_path=label_list_path,
                                        save_dir_path=save_dir_path)
        self.model_dir = self.args.model_dir
        self.skip_analyze = self.args.skip_analyze
        self.pruned_flops = self.args.pruned_flops

    def _add_argument(self):
        super(PruneXSeg, self)._add_argument()
        self._arg_parse.add_argument("--model_dir", dest="model_dir", default="{}"
                                     .format(os.path.join(self._save_dir_path, "best_model")),
                                     metavar="", help="模型读取路径。默认为 {}"
                                     .format(os.path.join(self._save_dir_path, "best_model")))
        self._arg_parse.add_argument("--skip_analyze", action="store_true",
                                     dest="skip_analyze", help="是否跳过分析模型各层参数在不同的裁剪比例下的敏感度，默认不跳过")
        self._arg_parse.add_argument("--pruned_flops", type=float, default=0.2,
                                     dest="pruned_flops", metavar="", help="根据选择的 FLOPS 减小比例对模型进行裁剪。默认为 0.2")

    def check(self):
        super(PruneXSeg, self).check()
        mod.utils.check_path(self.model_dir)


class QuantXSeg(BaseTrainXSeg):
    """
    返回 PaddleX 图像分割模型在线量化命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 train_list_path=config.TRAIN_LIST_PATH,
                 eval_list_path=config.EVAL_LIST_PATH,
                 label_list_path=config.LABEL_LIST_PATH,
                 save_dir_path=config.SAVE_DIR_PATH):
        super(QuantXSeg, self).__init__(args=args,
                                        dataset_path=dataset_path,
                                        train_list_path=train_list_path,
                                        eval_list_path=eval_list_path,
                                        label_list_path=label_list_path,
                                        save_dir_path=save_dir_path)
        self.model_dir = self.args.model_dir

    def _add_argument(self):
        super(QuantXSeg, self)._add_argument()
        self._arg_parse.add_argument("--model_dir", dest="model_dir", default="{}"
                                     .format(os.path.join(self._save_dir_path, "best_model")),
                                     metavar="", help="模型读取路径。默认为 {}"
                                     .format(os.path.join(self._save_dir_path, "best_model")))

    def check(self):
        super(QuantXSeg, self).check()
        mod.utils.check_path(self.model_dir)


class TestX():
    """
    返回 PaddleX 测试命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 test_list_path=config.TEST_LIST_PATH):
        self.__dataset_path = dataset_path
        self.__test_list_path = test_list_path

        self._arg_parse = argparse.ArgumentParser()
        self.add_argument()
        self.args = self._arg_parse.parse_args() if args == None else args

        self.cpu = self.args.cpu
        self.epochs = self.args.epochs

        self.dataset = dataset_path if self.args.dataset == "" else self.args.dataset
        self.test_list = os.path.join(
            self.dataset, test_list_path) if self.args.test_list == "" else self.args.test_list
        self.model_dir = self.args.model_dir

    def add_argument(self):
        """
        添加命令行参数
        """
        self._arg_parse.add_argument("--cpu", action="store_true",
                                     dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
        self._arg_parse.add_argument("--epochs", type=int, default=4,
                                     dest="epochs", metavar="", help="测试几轮，默认 4 轮")
        self._arg_parse.add_argument("--dataset", dest="dataset", default="",
                                     metavar="", help="数据集目录，默认 {}".format(self.__dataset_path))
        self._arg_parse.add_argument("--test_list", dest="test_list", default="", metavar="",
                                     help="训练集列表，默认 '--dataset' 参数目录下的 {}".format(self.__test_list_path))
        self._arg_parse.add_argument("--model_dir", dest="model_dir", default="./output/best_model",
                                     metavar="", help="读取训练后的模型目录，默认 ./output/best_model")

    def check(self):
        mod.utils.check_path(self.dataset)
        mod.utils.check_path(self.test_list)
        mod.utils.check_path(self.model_dir)


class PredictX():
    """
    返回 PaddleX 预测命令行参数
    """

    def __init__(self, args=None, dataset_path=config.DATASET_PATH,
                 infer_list_path=config.INFER_LIST_PATH):
        self.__dataset_path = dataset_path
        self.__infer_list_path = infer_list_path

        self._arg_parse = argparse.ArgumentParser()
        self.add_argument()
        self.args = self._arg_parse.parse_args() if args == None else args

        self.cpu = self.args.cpu

        self.dataset = dataset_path if self.args.dataset == "" else self.args.dataset
        self.infer_list = os.path.join(
            self.dataset, infer_list_path) if self.args.infer_list == "" else self.args.infer_list
        self.model_dir = self.args.model_dir
        self.result_info = self.args.result_info
        self.result_path = self.args.result_path
        self.split = self.args.split

    def add_argument(self):
        """
        添加命令行参数
        """
        self._arg_parse.add_argument("--cpu", action="store_true",
                                     dest="cpu", help="是否使用 cpu 计算，默认使用 CUDA")
        self._arg_parse.add_argument("--dataset", dest="dataset", default="",
                                     metavar="", help="数据集目录，默认 {}".format(self.__dataset_path))
        self._arg_parse.add_argument("--infer_list", dest="infer_list", default="",
                                     metavar="", help="预测集列表，默认 '--dataset' 参数目录下的 {}".format(self.__infer_list_path))
        self._arg_parse.add_argument("--model_dir", dest="model_dir", default="./output/best_model",
                                     metavar="", help="读取训练后的模型目录，默认 ./output/best_model")
        self._arg_parse.add_argument("--result_info", action="store_true",
                                     dest="result_info", help="显示预测结果详细信息，默认 不显示")
        self._arg_parse.add_argument("--result_path", dest="result_path", default="./result/result.csv",
                                     metavar="", help="预测结果文件路径，默认 ./result/result.csv")
        self._arg_parse.add_argument("--split", dest="split", default=",",
                                     metavar="", help="数据分隔符，默认 ','")

    def check(self):
        mod.utils.check_path(self.dataset)
        mod.utils.check_path(self.infer_list)
        mod.utils.check_path(self.model_dir)

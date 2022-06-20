## paddlex_det 参数说明

## train.py

```bash
[06-20 16:57:17 MainThread @logger.py:242] Argv: run/train.py -h
[06-20 16:57:17 MainThread @utils.py:73] paddlepaddle version: 2.2.2.
usage: train.py [-h] [--cpu] [--num_workers] [--epochs] [--batch_size]
                [--learning_rate] [--early_stop] [--early_stop_patience]
                [--save_interval_epochs] [--log_interval_steps]
                [--resume_checkpoint] [--save_dir] [--dataset] [--train_list]
                [--eval_list] [--label_list] [--warmup_steps]
                [--warmup_start_lr] [--lr_decay_epochs] [--lr_decay_gamma]
                [--use_ema] [--opti_scheduler] [--opti_reg_coeff]
                [--pretrain_weights] [--model] [--model_list] [--backbone]

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 是否使用 cpu 计算，默认使用 CUDA
  --num_workers         线程数量，默认 auto，为CPU核数的一半
  --epochs              训练几轮，默认 4 轮
  --batch_size          一批次数量，默认 16
  --learning_rate       学习率，默认 0.025
  --early_stop          是否使用提前终止训练策略。默认为 False
  --early_stop_patience 
                        当使用提前终止训练策略时，如果验证集精度在early_stop_patience 个 epoch
                        内连续下降或持平，则终止训练。默认为 5
  --save_interval_epochs 
                        模型保存间隔(单位: 迭代轮数)。默认为 1
  --log_interval_steps 
                        训练日志输出间隔（单位：迭代次数）。默认为 10
  --resume_checkpoint   恢复训练时指定上次训练保存的模型路径, 默认不会恢复训练
  --save_dir            模型保存路径。默认为 ./output/
  --dataset             数据集目录，默认 ./dataset/
  --train_list          训练集列表，默认 '--dataset' 参数目录下的 train_list.txt
  --eval_list           评估集列表，默认 '--dataset' 参数目录下的 val_list.txt
  --label_list          分类标签列表，默认 '--dataset' 参数目录下的 labels.txt
  --warmup_steps        默认优化器的 warmup 步数，学习率将在设定的步数内，从 warmup_start_lr
                        线性增长至设定的 learning_rate，默认为 0
  --warmup_start_lr     默认优化器的 warmup 起始学习率，默认为 0.0
  --lr_decay_epochs     默认优化器的学习率衰减轮数。默认为 30 60 90
  --lr_decay_gamma      默认优化器的学习率衰减率。默认为 0.1
  --use_ema             是否使用指数衰减计算参数的滑动平均值。默认为 False
  --opti_scheduler      优化器的调度器，默认 auto，可选 auto，cosine，piecewise
  --opti_reg_coeff      优化器衰减系数，如果 opti_scheduler 是 Cosine，默认是 4e-05，如果
                        opti_scheduler 是 Piecewise，默认是 1e-04
  --pretrain_weights    若指定为'.pdparams'文件时，从文件加载模型权重；若为字符串’IMAGENET’，则自动下载在Ima
                        geNet图片数据上预训练的模型权重；若为字符串’COCO’，则自动下载在COCO数据集上预训练的模型权重；
                        若为None，则不使用预训练模型。默认为'IMAGENET'
  --model               PaddleX 模型名称
  --model_list          输出 PaddleX 模型名称，默认不输出，选择后只输出信息，不会开启训练
  --backbone            目标检测模型的 backbone 网络
```

## quant.py

```bash
[06-20 16:57:20 MainThread @logger.py:242] Argv: run/quant.py -h
[06-20 16:57:20 MainThread @utils.py:73] paddlepaddle version: 2.2.2.
usage: quant.py [-h] [--cpu] [--num_workers] [--epochs] [--batch_size]
                [--learning_rate] [--early_stop] [--early_stop_patience]
                [--save_interval_epochs] [--log_interval_steps]
                [--resume_checkpoint] [--save_dir] [--dataset] [--train_list]
                [--eval_list] [--label_list] [--warmup_steps]
                [--warmup_start_lr] [--lr_decay_epochs] [--lr_decay_gamma]
                [--use_ema] [--opti_scheduler] [--opti_reg_coeff]
                [--model_dir]

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 是否使用 cpu 计算，默认使用 CUDA
  --num_workers         线程数量，默认 auto，为CPU核数的一半
  --epochs              训练几轮，默认 4 轮
  --batch_size          一批次数量，默认 16
  --learning_rate       学习率，默认 0.025
  --early_stop          是否使用提前终止训练策略。默认为 False
  --early_stop_patience 
                        当使用提前终止训练策略时，如果验证集精度在early_stop_patience 个 epoch
                        内连续下降或持平，则终止训练。默认为 5
  --save_interval_epochs 
                        模型保存间隔(单位: 迭代轮数)。默认为 1
  --log_interval_steps 
                        训练日志输出间隔（单位：迭代次数）。默认为 10
  --resume_checkpoint   恢复训练时指定上次训练保存的模型路径, 默认不会恢复训练
  --save_dir            模型保存路径。默认为 ./output/
  --dataset             数据集目录，默认 ./dataset/
  --train_list          训练集列表，默认 '--dataset' 参数目录下的 train_list.txt
  --eval_list           评估集列表，默认 '--dataset' 参数目录下的 val_list.txt
  --label_list          分类标签列表，默认 '--dataset' 参数目录下的 labels.txt
  --warmup_steps        默认优化器的 warmup 步数，学习率将在设定的步数内，从 warmup_start_lr
                        线性增长至设定的 learning_rate，默认为 0
  --warmup_start_lr     默认优化器的 warmup 起始学习率，默认为 0.0
  --lr_decay_epochs     默认优化器的学习率衰减轮数。默认为 30 60 90
  --lr_decay_gamma      默认优化器的学习率衰减率。默认为 0.1
  --use_ema             是否使用指数衰减计算参数的滑动平均值。默认为 False
  --opti_scheduler      优化器的调度器，默认 auto，可选 auto，cosine，piecewise
  --opti_reg_coeff      优化器衰减系数，如果 opti_scheduler 是 Cosine，默认是 4e-05，如果
                        opti_scheduler 是 Piecewise，默认是 1e-04
  --model_dir           模型读取路径。默认为 ./output/best_model
```

## prune.py

```bash
[06-20 16:57:22 MainThread @logger.py:242] Argv: run/prune.py -h
[06-20 16:57:22 MainThread @utils.py:73] paddlepaddle version: 2.2.2.
usage: prune.py [-h] [--cpu] [--num_workers] [--epochs] [--batch_size]
                [--learning_rate] [--early_stop] [--early_stop_patience]
                [--save_interval_epochs] [--log_interval_steps]
                [--resume_checkpoint] [--save_dir] [--dataset] [--train_list]
                [--eval_list] [--label_list] [--warmup_steps]
                [--warmup_start_lr] [--lr_decay_epochs] [--lr_decay_gamma]
                [--use_ema] [--opti_scheduler] [--opti_reg_coeff]
                [--model_dir] [--skip_analyze] [--pruned_flops]

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 是否使用 cpu 计算，默认使用 CUDA
  --num_workers         线程数量，默认 auto，为CPU核数的一半
  --epochs              训练几轮，默认 4 轮
  --batch_size          一批次数量，默认 16
  --learning_rate       学习率，默认 0.025
  --early_stop          是否使用提前终止训练策略。默认为 False
  --early_stop_patience 
                        当使用提前终止训练策略时，如果验证集精度在early_stop_patience 个 epoch
                        内连续下降或持平，则终止训练。默认为 5
  --save_interval_epochs 
                        模型保存间隔(单位: 迭代轮数)。默认为 1
  --log_interval_steps 
                        训练日志输出间隔（单位：迭代次数）。默认为 10
  --resume_checkpoint   恢复训练时指定上次训练保存的模型路径, 默认不会恢复训练
  --save_dir            模型保存路径。默认为 ./output/
  --dataset             数据集目录，默认 ./dataset/
  --train_list          训练集列表，默认 '--dataset' 参数目录下的 train_list.txt
  --eval_list           评估集列表，默认 '--dataset' 参数目录下的 val_list.txt
  --label_list          分类标签列表，默认 '--dataset' 参数目录下的 labels.txt
  --warmup_steps        默认优化器的 warmup 步数，学习率将在设定的步数内，从 warmup_start_lr
                        线性增长至设定的 learning_rate，默认为 0
  --warmup_start_lr     默认优化器的 warmup 起始学习率，默认为 0.0
  --lr_decay_epochs     默认优化器的学习率衰减轮数。默认为 30 60 90
  --lr_decay_gamma      默认优化器的学习率衰减率。默认为 0.1
  --use_ema             是否使用指数衰减计算参数的滑动平均值。默认为 False
  --opti_scheduler      优化器的调度器，默认 auto，可选 auto，cosine，piecewise
  --opti_reg_coeff      优化器衰减系数，如果 opti_scheduler 是 Cosine，默认是 4e-05，如果
                        opti_scheduler 是 Piecewise，默认是 1e-04
  --model_dir           模型读取路径。默认为 ./output/best_model
  --skip_analyze        是否跳过分析模型各层参数在不同的裁剪比例下的敏感度，默认不跳过
  --pruned_flops        根据选择的 FLOPS 减小比例对模型进行裁剪。默认为 0.2
```

## infer.py

```bash
[06-20 16:57:25 MainThread @logger.py:242] Argv: run/infer.py -h
[06-20 16:57:25 MainThread @utils.py:73] paddlepaddle version: 2.2.2.
usage: infer.py [-h] [--model_dir] [--predict_image] [--predict_image_dir]
                [--threshold] [--result_list] [--result_dir] [--show_result]

读取模型并预测

optional arguments:
  -h, --help            show this help message and exit
  --model_dir           读取模型的目录，默认 './output/best_model'
  --predict_image       预测的图像文件
  --predict_image_dir   预测的图像目录，选择后 --result_list，--show_result 失效
  --threshold           score阈值，将Box置信度低于该阈值的框过滤，默认 0.5
  --result_list         预测的结果列表文件，默认 './result/result.txt'
  --result_dir          预测结果可视化的保存目录，默认 './result'
  --show_result         显示预测结果的图像
```

## 项目地址

- [gitee](https://gitee.com/cnhemiya/bmm-paddle-helper)
- [github](https://github.com/cnhemiya/bmm-paddle-helper)


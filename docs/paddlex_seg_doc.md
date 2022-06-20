## paddlex_seg 参数说明

## train.py

```bash
[06-20 16:59:51 MainThread @logger.py:242] Argv: run/train.py -h
[06-20 16:59:51 MainThread @utils.py:73] paddlepaddle version: 2.2.2.
usage: train.py [-h] [--cpu] [--num_workers] [--epochs] [--batch_size]
                [--learning_rate] [--early_stop] [--early_stop_patience]
                [--save_interval_epochs] [--log_interval_steps]
                [--resume_checkpoint] [--save_dir] [--dataset] [--train_list]
                [--eval_list] [--label_list] [--lr_decay_power]
                [--use_mixed_loss] [--align_corners] [--backbone]
                [--hrnet_width] [--pretrain_weights] [--model] [--model_list]

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
  --lr_decay_power      默认优化器学习率衰减指数。默认为 0.9
  --use_mixed_loss      是否使用混合损失函数。如果为True，混合使用CrossEntropyLoss和LovaszSoftmaxL
                        oss，权重分别为0.8和0.2。如果为False，则仅使用CrossEntropyLoss。也可以以列表的
                        形式自定义混合损失函数，列表的每一个元素为(损失函数类型，权重)元组，损失函数类型取值范围为['CrossE
                        ntropyLoss', 'DiceLoss',
                        'LovaszSoftmaxLoss']。默认为False。
  --align_corners       是网络中对特征图进行插值时是否将四个角落像素的中心对齐。若特征图尺寸为偶数，建议设为True。若特征图尺寸为
                        奇数，建议设为False。默认为False。
  --backbone            图像分割模型 DeepLabV3P 的 backbone 网络，取值范围为['ResNet50_vd',
                        'ResNet101_vd']，默认为'ResNet50_vd'。
  --hrnet_width         图像分割模型 HRNet 的 width
                        网络，高分辨率分支中特征层的通道数量。默认为48。可选择取值为[18, 48]。
  --pretrain_weights    若指定为'.pdparams'文件时，则从文件加载模型权重；若为字符串'CITYSCAPES'，则自动下载在
                        CITYSCAPES图片数据上预训练的模型权重；若为字符串'PascalVOC'，则自动下载在PascalV
                        OC图片数据上预训练的模型权重；若为字符串'IMAGENET'，则自动下载在ImageNet图片数据上预训练
                        的模型权重；若为None，则不使用预训练模型。默认为'CITYSCAPES'。
  --model               PaddleX 模型名称
  --model_list          输出 PaddleX 模型名称，默认不输出，选择后只输出信息，不会开启训练
```

## quant.py

```bash
[06-20 16:59:54 MainThread @logger.py:242] Argv: run/quant.py -h
[06-20 16:59:54 MainThread @utils.py:73] paddlepaddle version: 2.2.2.
usage: quant.py [-h] [--cpu] [--num_workers] [--epochs] [--batch_size]
                [--learning_rate] [--early_stop] [--early_stop_patience]
                [--save_interval_epochs] [--log_interval_steps]
                [--resume_checkpoint] [--save_dir] [--dataset] [--train_list]
                [--eval_list] [--label_list] [--lr_decay_power] [--model_dir]

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
  --lr_decay_power      默认优化器学习率衰减指数。默认为 0.9
  --model_dir           模型读取路径。默认为 ./output/best_model
```

## prune.py

```bash
[06-20 16:59:56 MainThread @logger.py:242] Argv: run/prune.py -h
[06-20 16:59:56 MainThread @utils.py:73] paddlepaddle version: 2.2.2.
usage: prune.py [-h] [--cpu] [--num_workers] [--epochs] [--batch_size]
                [--learning_rate] [--early_stop] [--early_stop_patience]
                [--save_interval_epochs] [--log_interval_steps]
                [--resume_checkpoint] [--save_dir] [--dataset] [--train_list]
                [--eval_list] [--label_list] [--lr_decay_power] [--model_dir]
                [--skip_analyze] [--pruned_flops]

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
  --lr_decay_power      默认优化器学习率衰减指数。默认为 0.9
  --model_dir           模型读取路径。默认为 ./output/best_model
  --skip_analyze        是否跳过分析模型各层参数在不同的裁剪比例下的敏感度，默认不跳过
  --pruned_flops        根据选择的 FLOPS 减小比例对模型进行裁剪。默认为 0.2
```

## infer.py

```bash
[06-20 16:59:59 MainThread @logger.py:242] Argv: run/infer.py -h
[06-20 16:59:59 MainThread @utils.py:73] paddlepaddle version: 2.2.2.
usage: infer.py [-h] [--model_dir] [--predict_image] [--predict_image_dir]
                [--weight] [--result_dir]

读取模型并预测

optional arguments:
  -h, --help            show this help message and exit
  --model_dir           读取模型的目录，默认 './output/best_model'
  --predict_image       预测的图像文件
  --predict_image_dir   预测的图像目录
  --weight              mask可视化结果与原图权重因子，weight表示原图的权重，默认 0.6
  --result_dir          预测结果可视化的保存目录，默认 './result'
```

## 项目地址

- [gitee](https://gitee.com/cnhemiya/bmm-paddle-helper)
- [github](https://github.com/cnhemiya/bmm-paddle-helper)


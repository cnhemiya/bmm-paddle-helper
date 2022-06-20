# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-05-15 20:35
文档说明: PaddleX 配置
"""


import paddlex


# PaddleX 图像分类模型名称
PDX_CLS_MODEL_NAME = ['PPLCNet', 'PPLCNet_ssld', 'ResNet18', 'ResNet18_vd', 'ResNet34',
                      'ResNet34_vd', 'ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld', 'ResNet101',
                      'ResNet101_vd', 'ResNet101_vd_ssld', 'ResNet152', 'ResNet152_vd', 'ResNet200_vd',
                      'DarkNet53', 'MobileNetV1', 'MobileNetV2', 'MobileNetV3_small', 'MobileNetV3_small_ssld',
                      'MobileNetV3_large', 'MobileNetV3_large_ssld', 'Xception41', 'Xception65', 'Xception71',
                      'ShuffleNetV2', 'ShuffleNetV2_swish', 'DenseNet121', 'DenseNet161', 'DenseNet169',
                      'DenseNet201', 'DenseNet264', 'HRNet_W18_C', 'HRNet_W30_C', 'HRNet_W32_C',
                      'HRNet_W40_C', 'HRNet_W44_C', 'HRNet_W48_C', 'HRNet_W64_C', 'AlexNet']

# PaddleX 目标检测模型名称
PDX_DET_MODEL_NAME = ['PPYOLOv2', 'PPYOLO', 'PPYOLOTiny',
                      'PicoDet', 'YOLOv3', 'FasterRCNN', 'MaskRCNN']

# PaddleX 目标检测模型 PPYOLOv2 backbone 网络
PDX_DET_PPYOLOV2_BACKBONE = ['ResNet50_vd_dcn', 'ResNet101_vd_dcn']

# PaddleX 目标检测模型 PPYOLO backbone 网络
PDX_DET_PPYOLO_BACKBONE = ['ResNet50_vd_dcn', 'ResNet18_vd',
                           'MobileNetV3_large', 'MobileNetV3_small']

# PaddleX 目标检测模型 PPYOLOTiny backbone 网络
PDX_DET_PPYOLOTINY_BACKBONE = ['MobileNetV3']

# PaddleX 目标检测模型 PicoDet backbone 网络
PDX_DET_PICODET_BACKBONE = ['ESNet_s', 'ESNet_m', 'ESNet_l', 'LCNet',
                            'MobileNetV3', 'ResNet18_vd']

# PaddleX 目标检测模型 YOLOv3 backbone 网络
PDX_DET_YOLOV3_BACKBONE = ['MobileNetV1', 'MobileNetV1_ssld', 'MobileNetV3',
                           'MobileNetV3_ssld', 'DarkNet53', 'ResNet50_vd_dcn', 'ResNet34']

# PaddleX 目标检测模型 FasterRCNN backbone 网络
PDX_DET_FASTERRCNN_BACKBONE = ['ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld',
                               'ResNet34', 'ResNet34_vd', 'ResNet101', 'ResNet101_vd', 'HRNet_W18']

# PaddleX 目标检测模型 实例分割 MaskRCNN backbone 网络
PDX_DET_MASKRCNN_BACKBONE = ['ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld',
                             'ResNet101', 'ResNet101_vd']

# PaddleX 图像分割模型名称
PDX_SEG_MODEL_NAME = ["DeepLabV3P", "BiSeNetV2", "UNet", "HRNet", "FastSCNN"]

# PaddleX 图像分割模型 DeepLabV3P backbone 网络
PDX_SEG_DEEPLABV3P_BACKBONE = ['ResNet50_vd', 'ResNet101_vd']


def pdx_cls_model(model_name: str, num_classes: int):
    """
    获取 PaddleX 分类图像模型

    Args:
        model_name (str):  PaddleX 图像分类模型名称
        num_classes (int): 分类数量

    Returns:
        model: 模型
        model_name: 模型名称
    """
    model = None
    if model_name not in PDX_CLS_MODEL_NAME:
        raise Exception("PaddleX 图像分类模型名称错误，错误信息：{}".format(model_name))

    if model_name == "PPLCNet":
        model = paddlex.cls.PPLCNet(num_classes=num_classes)
    elif model_name == "PPLCNet_ssld":
        model = paddlex.cls.PPLCNet_ssld(num_classes=num_classes)
    elif model_name == "ResNet18":
        model = paddlex.cls.ResNet18(num_classes=num_classes)
    elif model_name == "ResNet18_vd":
        model = paddlex.cls.ResNet18_vd(num_classes=num_classes)
    elif model_name == "ResNet34":
        model = paddlex.cls.ResNet34(num_classes=num_classes)
    elif model_name == "ResNet34_vd":
        model = paddlex.cls.ResNet34_vd(num_classes=num_classes)
    elif model_name == "ResNet50":
        model = paddlex.cls.ResNet50(num_classes=num_classes)
    elif model_name == "ResNet50_vd":
        model = paddlex.cls.ResNet50_vd(num_classes=num_classes)
    elif model_name == "ResNet50_vd_ssld":
        model = paddlex.cls.ResNet50_vd_ssld(num_classes=num_classes)
    elif model_name == "ResNet101":
        model = paddlex.cls.ResNet101(num_classes=num_classes)
    elif model_name == "ResNet101_vd":
        model = paddlex.cls.ResNet101_vd(num_classes=num_classes)
    elif model_name == "ResNet101_vd_ssld":
        model = paddlex.cls.ResNet101_vd_ssld(num_classes=num_classes)
    elif model_name == "ResNet152":
        model = paddlex.cls.ResNet152(num_classes=num_classes)
    elif model_name == "ResNet152_vd":
        model = paddlex.cls.ResNet152_vd(num_classes=num_classes)
    elif model_name == "ResNet200_vd":
        model = paddlex.cls.ResNet200_vd(num_classes=num_classes)
    elif model_name == "DarkNet53":
        model = paddlex.cls.DarkNet53(num_classes=num_classes)
    elif model_name == "MobileNetV1":
        model = paddlex.cls.MobileNetV1(num_classes=num_classes, scale=1.0)
    elif model_name == "MobileNetV2":
        model = paddlex.cls.MobileNetV2(num_classes=num_classes, scale=1.0)
    elif model_name == "MobileNetV3_small":
        model = paddlex.cls.MobileNetV3_small(
            num_classes=num_classes, scale=1.0)
    elif model_name == "MobileNetV3_small_ssld":
        model = paddlex.cls.MobileNetV3_small_ssld(
            num_classes=num_classes, scale=1.0)
    elif model_name == "MobileNetV3_large":
        model = paddlex.cls.MobileNetV3_large(
            num_classes=num_classes, scale=1.0)
    elif model_name == "MobileNetV3_large_ssld":
        model = paddlex.cls.MobileNetV3_large_ssld(num_classes=num_classes)
    elif model_name == "Xception41":
        model = paddlex.cls.Xception41(num_classes=num_classes)
    elif model_name == "Xception65":
        model = paddlex.cls.Xception65(num_classes=num_classes)
    elif model_name == "Xception71":
        model = paddlex.cls.Xception71(num_classes=num_classes)
    elif model_name == "ShuffleNetV2":
        model = paddlex.cls.ShuffleNetV2(num_classes=num_classes, scale=1.0)
    elif model_name == "ShuffleNetV2_swish":
        model = paddlex.cls.ShuffleNetV2_swish(
            num_classes=num_classes, scale=1.0)
    elif model_name == "DenseNet121":
        model = paddlex.cls.DenseNet121(num_classes=num_classes)
    elif model_name == "DenseNet161":
        model = paddlex.cls.DenseNet161(num_classes=num_classes)
    elif model_name == "DenseNet169":
        model = paddlex.cls.DenseNet169(num_classes=num_classes)
    elif model_name == "DenseNet201":
        model = paddlex.cls.DenseNet201(num_classes=num_classes)
    elif model_name == "DenseNet264":
        model = paddlex.cls.DenseNet264(num_classes=num_classes)
    elif model_name == "HRNet_W18_C":
        model = paddlex.cls.HRNet_W18_C(num_classes=num_classes)
    elif model_name == "HRNet_W30_C":
        model = paddlex.cls.HRNet_W30_C(num_classes=num_classes)
    elif model_name == "HRNet_W32_C":
        model = paddlex.cls.HRNet_W32_C(num_classes=num_classes)
    elif model_name == "HRNet_W40_C":
        model = paddlex.cls.HRNet_W40_C(num_classes=num_classes)
    elif model_name == "HRNet_W44_C":
        model = paddlex.cls.HRNet_W44_C(num_classes=num_classes)
    elif model_name == "HRNet_W48_C":
        model = paddlex.cls.HRNet_W48_C(num_classes=num_classes)
    elif model_name == "HRNet_W64_C":
        model = paddlex.cls.HRNet_W64_C(num_classes=num_classes)
    elif model_name == "AlexNet":
        model = paddlex.cls.AlexNet(num_classes=num_classes)

    return model, model_name


def print_pdx_cls_model_name():
    """
    打印 PaddleX 图像分类模型名称
    """
    print("\nPaddleX 图像分类模型")
    print(PDX_CLS_MODEL_NAME)


def pdx_det_model(model_name: str, num_classes: int, backbone: str):
    """
    获取 PaddleX 目标检测模型

    Args:
        model_name (str):  PaddleX 目标检测模型名称
        num_classes (int): 分类数量
        backbone (str): 目标检测模型 backbone 网络

    Raises:
        Exception: PaddleX 目标检测模型名称错误
        Exception: PPYOLOv2 backbone 网络错误
        Exception: PPYOLO backbone 网络错误
        Exception: PPYOLOTiny backbone 网络错误
        Exception: PicoDet backbone 网络错误
        Exception: YOLOv3 backbone 网络错误
        Exception: FasterRCNN backbone 网络错误
        Exception: MaskRCNN backbone 网络错误

    Returns:
        model: 模型
        model_name: 模型名称
    """
    if model_name not in PDX_DET_MODEL_NAME:
        raise Exception("PaddleX 目标检测模型名称错误，错误信息：{}".format(model_name))
    if (model_name == 'PPYOLOv2') and (backbone not in PDX_DET_PPYOLOV2_BACKBONE):
        raise Exception(
            "PaddleX 目标检测模型 PPYOLOv2 backbone 网络错误，错误信息：{}".format(backbone))
    if (model_name == 'PPYOLO') and (backbone not in PDX_DET_PPYOLO_BACKBONE):
        raise Exception(
            "PaddleX 目标检测模型 PPYOLO backbone 网络错误，错误信息：{}".format(backbone))
    if (model_name == 'PPYOLOTiny') and (backbone not in PDX_DET_PPYOLOTINY_BACKBONE):
        raise Exception(
            "PaddleX 目标检测模型 PPYOLOTiny backbone 网络错误，错误信息：{}".format(backbone))
    if (model_name == 'PicoDet') and (backbone not in PDX_DET_PICODET_BACKBONE):
        raise Exception(
            "PaddleX 目标检测模型 PicoDet backbone 网络错误，错误信息：{}".format(backbone))
    if (model_name == 'YOLOv3') and (backbone not in PDX_DET_YOLOV3_BACKBONE):
        raise Exception(
            "PaddleX 目标检测模型 YOLOv3 backbone 网络错误，错误信息：{}".format(backbone))
    if (model_name == 'FasterRCNN') and (backbone not in PDX_DET_FASTERRCNN_BACKBONE):
        raise Exception(
            "PaddleX 目标检测模型 FasterRCNN backbone 网络错误，错误信息：{}".format(backbone))
    if (model_name == 'MaskRCNN') and (backbone not in PDX_DET_MASKRCNN_BACKBONE):
        raise Exception(
            "PaddleX 目标检测模型 MaskRCNN backbone 网络错误，错误信息：{}".format(backbone))

    model = None
    if (model_name == 'PPYOLOv2'):
        model = paddlex.det.PPYOLOv2(
            num_classes=num_classes, backbone=backbone)
    elif (model_name == 'PPYOLO'):
        model = paddlex.det.PPYOLO(num_classes=num_classes, backbone=backbone)
    elif (model_name == 'PPYOLOTiny'):
        model = paddlex.det.PPYOLOTiny(
            num_classes=num_classes, backbone=backbone)
    elif (model_name == 'PicoDet'):
        model = paddlex.det.PicoDet(num_classes=num_classes, backbone=backbone)
    elif (model_name == 'YOLOv3'):
        model = paddlex.det.YOLOv3(num_classes=num_classes, backbone=backbone)
    elif (model_name == 'FasterRCNN'):
        model = paddlex.det.FasterRCNN(
            num_classes=num_classes, backbone=backbone)
    elif (model_name == 'MaskRCNN'):
        model = paddlex.det.MaskRCNN(
            num_classes=num_classes, backbone=backbone)

    return model, model_name


def print_pdx_det_model_name():
    """
    打印 PaddleX 目标检测模型名称
    """
    print("\nPaddleX 目标检测模型")
    print(PDX_DET_MODEL_NAME)
    print("PPYOLOv2 backbone 网络")
    print(PDX_DET_PPYOLOV2_BACKBONE)
    print("PPYOLO backbone 网络")
    print(PDX_DET_PPYOLO_BACKBONE)
    print("PPYOLOTiny backbone 网络")
    print(PDX_DET_PPYOLOTINY_BACKBONE)
    print("PicoDet backbone 网络")
    print(PDX_DET_PICODET_BACKBONE)
    print("YOLOv3 backbone 网络")
    print(PDX_DET_YOLOV3_BACKBONE)
    print("FasterRCNN backbone 网络")
    print(PDX_DET_FASTERRCNN_BACKBONE)
    print("MaskRCNN backbone 网络")
    print(PDX_DET_MASKRCNN_BACKBONE)


def pdx_seg_model(model_name: str, num_classes: int, backbone: str, hrnet_width: int,
                  use_mixed_loss: False, align_corners: False):
    """
    获取 PaddleX 图像分割模型

    Args:
        model_name (str):  PaddleX 图像分割模型名称
        num_classes (int): 分类数量
        backbone (str): 图像分割模型 DeepLabV3P backbone 网络
        hrnet_width (int): hrnet_width 高分辨率分支中特征层的通道数量，可选择取值为[18, 48]
        use_mixed_loss (bool): 是否使用混合损失，可选择取值为[True, False]
        align_corners (bool): 网络中对特征图进行插值时是否将四个角落像素的中心对齐，可选择取值为[True, False]

    Raises:
        Exception: PaddleX 图像分割模型名称错误
        Exception: DeepLabV3P backbone 网络错误

    Returns:
        model: 模型
        model_name: 模型名称
    """
    if model_name not in PDX_SEG_MODEL_NAME:
        raise Exception("PaddleX 图像分割模型名称错误，错误信息：{}".format(model_name))
    if (model_name == 'DeepLabV3P') and (backbone not in PDX_SEG_DEEPLABV3P_BACKBONE):
        raise Exception(
            "PaddleX 图像分割模型 DeepLabV3P backbone 网络错误，错误信息：{}".format(backbone))

    model = None
    if (model_name == 'DeepLabV3P'):
        model = paddlex.seg.DeepLabV3P(
            num_classes=num_classes, backbone=backbone, use_mixed_loss=use_mixed_loss, align_corners=align_corners)
    elif (model_name == 'BiSeNetV2'):
        model = paddlex.seg.BiSeNetV2(
            num_classes=num_classes, use_mixed_loss=use_mixed_loss, align_corners=align_corners)
    elif (model_name == 'UNet'):
        model = paddlex.seg.UNet(
            num_classes=num_classes, use_mixed_loss=use_mixed_loss, align_corners=align_corners)
    elif (model_name == 'HRNet'):
        model = paddlex.seg.HRNet(num_classes=num_classes, width=hrnet_width,
                                  use_mixed_loss=use_mixed_loss, align_corners=align_corners)
    elif (model_name == 'FastSCNN'):
        model = paddlex.seg.FastSCNN(
            num_classes=num_classes, use_mixed_loss=use_mixed_loss, align_corners=align_corners)

    return model, model_name


def print_pdx_seg_model_name():
    """
    打印 PaddleX 图像分割模型名称
    """
    print("\nPaddleX 图像分割模型")
    print(PDX_SEG_MODEL_NAME)
    print("DeepLabV3P backbone 网络")
    print(PDX_SEG_DEEPLABV3P_BACKBONE)

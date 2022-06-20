# -*- coding: utf-8 -*-
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-08 21:52
文档说明: 结果报表
"""


import json


class Report:
    """
    结果报表, 使用 json 文件格式
    """

    def __init__(self):
        """
        构造函数, 大写是报表的键名, 小写是值
        """
        self._ID_NAME = "id"
        self._LOSS_NAME = "loss"
        self._ACC_NAME = "acc"
        self._EPOCHS_NAME = "epochs"
        self._BATCH_SIZE_NAME = "batch_size"
        self._LEARNING_RATE_NAME = "learning_rate"
        self.id = ""
        self.loss = 0.0
        self.acc = 0.0
        self.epochs = 0
        self.batch_size = 0
        self.learning_rate = 0.0

    def save(self, file_name: str):
        """
        保存报表

        Args:
            file_name (str): 报表文件名
        """
        data = {self._ID_NAME: self.id, self._LOSS_NAME: self.loss, self._ACC_NAME: self.acc,
                self._EPOCHS_NAME: self.epochs, self._BATCH_SIZE_NAME: self.batch_size,
                self._LEARNING_RATE_NAME: self.learning_rate}
        with open(file_name, "w") as f:
            json.dump(data, f)

    def load(self, file_name: str):
        """
        读取报表

        Args:
            file_name (str): 报表文件名
        """
        with open(file_name, "r") as f:
            data = json.load(f)
            self.id = data[self._ID_NAME]
            self.loss = data[self._LOSS_NAME]
            self.acc = data[self._ACC_NAME]
            self.epochs = data[self._EPOCHS_NAME]
            self.batch_size = data[self._BATCH_SIZE_NAME]
            self.learning_rate = data[self._LEARNING_RATE_NAME]


class ReportX:
    """
    PaddleX 结果报表, 使用 json 文件格式
    """

    def __init__(self):
        """
        构造函数, 大写是报表的键名, 小写是值
        """
        self._ID_NAME = "id"
        self._MODEL_NAME = "model"
        self._EPOCHS_NAME = "epochs"
        self._BATCH_SIZE_NAME = "batch_size"
        self._LEARNING_RATE_NAME = "learning_rate"
        self.id = ""
        self.model = ""
        self.epochs = 0
        self.batch_size = 0
        self.learning_rate = 0.0

    def save(self, file_name: str):
        """
        保存报表

        Args:
            file_name (str): 报表文件名
        """
        data = {self._ID_NAME: self.id, self._MODEL_NAME: self.model,
                self._EPOCHS_NAME: self.epochs, self._BATCH_SIZE_NAME: self.batch_size,
                self._LEARNING_RATE_NAME: self.learning_rate}
        with open(file_name, "w") as f:
            json.dump(data, f)

    def load(self, file_name: str):
        """
        读取报表

        Args:
            file_name (str): 报表文件名
        """
        with open(file_name, "r") as f:
            data = json.load(f)
            self.id = data[self._ID_NAME]
            self.model = data[self._MODEL_NAME]
            self.epochs = data[self._EPOCHS_NAME]
            self.batch_size = data[self._BATCH_SIZE_NAME]
            self.learning_rate = data[self._LEARNING_RATE_NAME]

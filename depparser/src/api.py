# -*- coding: utf-8 -*-
"""
API
----
封装对外提供接口
"""
from depparser.src.model import get_model

__all__ = ["train", "parser"]


def train():
    """
    训练模型
    """
    model = get_model()
    model.train()


def parser(sentence):
    """
    依存句法分析
    """
    model = get_model()
    return model.predict(sentence)

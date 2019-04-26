# -*- coding: utf-8 -*-
"""
API
----
封装对外提供接口
"""
from ner.src.corpus import get_corpus
from ner.src.model import get_model

__all__ = ["pre_process", "train", "recognize"]


def pre_process():
    """
    抽取语料特征
    """
    corpus = get_corpus()
    corpus.pre_process()


def train():
    """
    训练模型
    """
    model = get_model()
    model.train()


def recognize(sentence):
    """
    命名实体识别
    """
    model = get_model()
    return model.predict(sentence)

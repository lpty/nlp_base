# -*- coding: utf-8 -*-
"""
API
----
封装对外提供接口
"""
from corpus import get_corpus
from model import get_model


__all__ = ["train", "cut", "test"]


def train():
    """
    语料处理及hmm模型概率计算
    """
    corpus = get_corpus()
    corpus.initialize()
    corpus.cal_state()


def test():
    """
    模型测试
    """
    model = get_model()
    return model.test()


def cut(sentence):
    """
    分词
    """
    model = get_model()
    return model.cut(sentence)

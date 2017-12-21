# -*- coding: utf-8 -*-
"""
API
-----
对外提供接口封装
"""
from sentiment.model import get_model
from sentiment.dictionary import get_dictionary
from sentiment.config import get_config

__all__ = ["process_corpus", "train_model", "test_model"]


def process_corpus(**kwargs):
    """
    语料处理
    """
    dictionary = get_dictionary()
    return dictionary.cut(**kwargs)


def train_model():
    """
    模型训练
    """
    config = get_config()
    dictionary = get_dictionary()
    input_file = dictionary.get_corpus_path()
    output = config.get('train', 'model_path')
    model = get_model()
    model.train(input_file, output)


def test_model():
    """
    模型测试
    """
    config = get_config()
    test_file_path = config.get('test', 'test_seg_corpus_path')
    model = get_model()
    result = model.test(test_file_path)
    print('precision:', result.precision)
    print('recall:', result.recall)
    print('examples:', result.nexamples)

# -*- coding: utf-8 -*-
"""
MODEL
------
情感模型
"""
import os
import fasttext as ft
from sentiment.config import get_config

__comment_sentiment = None


class CommentSentiment:

    __model = None

    @classmethod
    def load_model(cls):
        """
        模型加载
        """
        config = get_config()
        model_path = '{}.bin'.format(config.get('train', 'model_path'))
        if os.path.exists(model_path):
            cls.__model = ft.load_model(model_path)

    @classmethod
    def train(cls, input_file, output, **kwargs):
        """
        模型训练

        * input_file             training file path (required)
        * output                 output file path (required)
        * label_prefix           label prefix ['__label__']
        * lr                     learning rate [0.1]
        * lr_update_rate         change the rate of updates for the learning rate [100]
        * dim                    size of word vectors [100]
        * ws                     size of the context window [5]
        * epoch                  number of epochs [5]
        * min_count              minimal number of word occurences [1]
        * neg                    number of negatives sampled [5]
        * word_ngrams            max length of word ngram [1]
        * loss                   loss function {ns, hs, softmax} [softmax]
        * bucket                 number of buckets [0]
        * minn                   min length of char ngram [0]
        * maxn                   max length of char ngram [0]
        * thread                 number of threads [12]
        * t                      sampling threshold [0.0001]
        * silent                 disable the log output from the C++ extension [1]
        * encoding               specify input_file encoding [utf-8]
        * pretrained_vectors     pretrained word vectors (.vec file) for supervised learning []
        """
        config = get_config()
        kwargs.setdefault('lr', config.get('model', 'lr'))
        kwargs.setdefault('lr_update_rate', config.get('model', 'lr_update_rate'))
        kwargs.setdefault('dim', config.get('model', 'dim'))
        kwargs.setdefault('ws', config.get('model', 'ws'))
        kwargs.setdefault('epoch', config.get('model', 'epoch'))
        kwargs.setdefault('word_ngrams', config.get('model', 'word_ngrams'))
        kwargs.setdefault('loss', config.get('model', 'loss'))
        kwargs.setdefault('bucket', config.get('model', 'bucket'))
        kwargs.setdefault('thread', config.get('model', 'thread'))
        kwargs.setdefault('silent', config.get('model', 'silent'))
        cls.__model = ft.supervised(input_file, output, **kwargs)
        return cls.__model

    @classmethod
    def test(cls, test_file_path):
        """
        模型测试
        """
        return cls.__model.test(test_file_path)


def get_model():
    """
    单例模型获取
    """
    global __comment_sentiment
    if not __comment_sentiment:
        __comment_sentiment = CommentSentiment
        __comment_sentiment.load_model()
    return __comment_sentiment

# -*- coding: utf-8 -*-
"""
CORPUS
-------
对语料处理的封装
"""
from depparser.src.config import get_config

__corpus = None


class Corpus:

    _config = get_config()

    # @classmethod
    # def pre_process(cls):
    #     train_path = cls._config.get('depparser', 'train_path')
    #     train_lines = cls.read_corpus_from_file(train_path)
    #     train_sentences = [sentence for sentence in cls.process_sentence(train_lines)]
    #     connections, words, pos = [], [], []
    #     for sentence in train_sentences:
    #         for word in sentence:
    #             connections.append(word[-1])
    #             words.append(word[1])
    #             pos.append(word[3])

    @staticmethod
    def process_sentence(lines):
        """
        处理句子
        """
        sentence = []
        for line in lines:
            if not line.strip():
                yield sentence
                sentence = []
            else:
                sentence.append(line.decode('utf-8').strip().split(u'\t'))

    @classmethod
    def initialize(cls):
        """
        语料初始化
        """
        train_process_path = cls._config.get('depparser', 'train_process_path')
        test_process_path = cls._config.get('depparser', 'test_process_path')
        train_lines = cls.read_corpus_from_file(train_process_path)
        test_lines = cls.read_corpus_from_file(test_process_path)
        cls.train_sentences = [sentence for sentence in cls.process_sentence(train_lines)]
        cls.test_sentences = [sentence for sentence in cls.process_sentence(test_lines)]

    @classmethod
    def generator(cls, train=True):
        """
        特征生成器
        """
        if train: sentences = cls.train_sentences
        else: sentences = cls.test_sentences
        return cls.extract_feature(sentences)

    @classmethod
    def extract_feature(cls, sentences):
        """
        提取特征
        """
        features, tags = [], []
        for index in range(len(sentences)):
            feature_list, tag_list = [], []
            for i in range(len(sentences[index])):
                feature = {"w0": sentences[index][i][0],
                           "p0": sentences[index][i][1],
                           "w-1": sentences[index][i-1][0] if i != 0 else "BOS",
                           "w+1": sentences[index][i+1][0] if i != len(sentences[index])-1 else "EOS",
                           "p-1": sentences[index][i-1][1] if i != 0 else "un",
                           "p+1": sentences[index][i+1][1] if i != len(sentences[index])-1 else "un"}
                feature["w-1:w0"] = feature["w-1"]+feature["w0"]
                feature["w0:w+1"] = feature["w0"]+feature["w+1"]
                feature["p-1:p0"] = feature["p-1"]+feature["p0"]
                feature["p0:p+1"] = feature["p0"]+feature["p+1"]
                feature["p-1:w0"] = feature["p-1"]+feature["w0"]
                feature["w0:p+1"] = feature["w0"]+feature["p+1"]
                feature_list.append(feature)
                tag_list.append(sentences[index][i][-1])
            features.append(feature_list)
            tags.append(tag_list)
        return features, tags

    @classmethod
    def read_corpus_from_file(cls, file_path):
        """
        读取语料
        """
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        return lines

    @classmethod
    def write_corpus_to_file(cls, data, file_path):
        """
        写语料
        """
        f = open(file_path, 'w')
        f.write(data)
        f.close()

    def __init__(self):
        raise Exception("This class have not element method.")


def get_corpus():
    """
    单例语料获取
    """
    global __corpus
    if not __corpus:
        __corpus = Corpus
    return __corpus

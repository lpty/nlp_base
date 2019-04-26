# -*- coding: utf-8 -*-
"""
RADICAL
-------
偏旁部首转换
"""
import csv
from postagger.src.config import get_config

__radical = None


class Radical(object):

    def __init__(self):
        self.config = get_config()
        self.dictionary = {}
        self.read_dictionary()
        self.origin_len = len(self.dictionary)

    def read_dictionary(self):
        """
        读取本地字典
        """
        path = self.config.get('postagger', 'dict_path')
        f = open(path, 'rU')
        reader = csv.reader(f)
        for line in reader:
            self.dictionary[line[0].decode('utf-8')] = line[1].decode('utf-8')
        f.close()

    def get_radical(self, word):
        """
        获取偏旁 
        """
        if word in self.dictionary:
            return self.dictionary[word]
        else:
            return word


def get_radical():
    """
    单例获取
    """
    global __radical
    if not __radical:
        __radical = Radical()
    return __radical

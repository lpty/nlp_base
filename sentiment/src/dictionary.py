# -*- coding: utf-8 -*-
"""
DICT
-----
字典类封装
"""
import os
import gzip
import jieba
import numpy as np
from io import StringIO
from collections import Counter
from sentiment.src.config import get_config

__dictionary = None


class Dictionary:

    _jieba = None
    _vocabs = None

    @classmethod
    def __get_stop_words(cls):
        puncs = set(u"？?!！·【】、；，。、\s+\t+~@#$%^&*()_+{}|:\"<"
                      u"~@#￥%……&*（）——+{}|：“”‘’《》>`\-=\[\]\\\\;',\./■")
        other = set(['hellip', 'quot', 'ldquo', 'rdquo', 'mdash'])
        return puncs | other

    @classmethod
    def label_maps(cls, item):
        """
        标签转换
        """
        _maps = {'1': '__label__1',
                 '2': '__label__2',
                 '3': '__label__3',
                 '__label__1': 'negative',
                 '__label__2': 'neutral',
                 '__label__3': 'positive'}
        return _maps.get(item)

    @classmethod
    def __load_user_dict(cls):
        """
        加载用户词典
        """
        config = get_config()
        user_dict_path = config.get('train', 'user_dict_path')
        gr = gzip.open(user_dict_path)
        lines = gr.readlines()
        words = set([line.strip() for line in lines if line.strip()])
        user_dict = ['{} {} n'.format(word, len(word)*1000) for word in words]
        buff_file = StringIO('\n'.join(user_dict))
        jieba.load_userdict(buff_file)
        cls._jieba = jieba
        gr.close()

    @classmethod
    def __get_vocabs(cls, vocabs_path):
        """
        读取本地词典
        """
        if not cls._vocabs:
            fr = open(vocabs_path, 'r')
            line = fr.read()
            vocabs = line.split('\n')
            fr.close()
        else:
            vocabs = cls._vocabs
        return vocabs

    @classmethod
    def __cut_corpus(cls, corpus_path, seg_corpus_path, sample, sample_corpus_path, vocabs_path, **kwargs):
        """
        语料分词
        """
        stop = cls.__get_stop_words()
        fr = open(corpus_path, 'r')
        fw = open(seg_corpus_path, 'w')
        lines = fr.readlines()
        for line in lines:
            words = [w.strip() for w in jieba.cut(line[1:]) if w not in stop and w.strip()]
            fw.write('{} {}\n'.format(cls.label_maps(line[0]), ' '.join(words).encode('utf-8')))
        fw.close()
        fr.close()

        if sample:
            cls.sample(seg_corpus_path, sample_corpus_path, vocabs_path)

    @classmethod
    def __cut_sentence(cls, sample, sentence, vocabs_path, **kwargs):
        """
        句子分词
        """
        words = [w.strip() for w in cls._jieba.cut(sentence[1:]) if w.strip()]
        if sample and os.path.exists(vocabs_path):
            vocabs = cls.__get_vocabs(vocabs_path)
            words = [w for w in words if w in vocabs]
        else:
            stop = cls.__get_stop_words()
            words = [w for w in words if w not in stop]
        return '{} {}'.format(cls.label_maps(sentence[0]), ' '.join(words).encode('utf-8'))

    @classmethod
    def cut(cls, **kwargs):
        """
        语料分词
        """
        config = get_config()
        kwargs.setdefault('corpus_path', config.get('train', 'corpus_path'))
        kwargs.setdefault('seg_corpus_path', config.get('train', 'seg_corpus_path'))
        kwargs.setdefault('sample_corpus_path', config.get('train', 'sample_corpus_path'))
        kwargs.setdefault('vocabs_path', config.get('train', 'vocabs_path'))
        kwargs.setdefault('sample', config.get('train', 'sample'))
        kwargs.setdefault('sentence', '')
        if not cls._jieba:
            cls.__load_user_dict()
        if not kwargs.get('sentence'):
            cls.__cut_corpus(**kwargs)
        else:
            return cls.__cut_sentence(**kwargs)

    @classmethod
    def __min_freq_sample(cls, words, freq=5):
        """
        最小采样,去除词频过低的词
        """
        word_counts = Counter(words)
        # 剔除出现频率低的词, 减少噪音
        return [word for word in words if word_counts[word] > freq]

    @classmethod
    def __down_sample(cls, words, t=1e-5, threshold=0.75):
        """
        下采样,去除词频过高的词
        """
        # 统计单词出现频次
        word_counts = Counter(words)
        total_count = len(words)
        # 计算单词频率
        word_freqs = {w: c / float(total_count) for w, c in word_counts.items()}
        # 计算被删除的概率
        prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in word_counts}
        # 剔除出现频率太高的词
        train_words = [w for w in words if prob_drop[w] < threshold]
        return set(train_words)

    @classmethod
    def sample(cls, seg_corpus_path, sample_corpus_path, vocabs_path):
        """
        词采样
        """
        fr = open(seg_corpus_path, 'r')
        lines = fr.readlines()
        all_words = []
        for line in lines:
            all_words.extend(line.split())
        temp_words = cls.__min_freq_sample(all_words)
        vocabs = cls.__down_sample(temp_words)

        fw = open(vocabs_path, 'w')
        fw.write('\n'.join(vocabs))
        fw.close()

        fw = open(sample_corpus_path, 'w')
        for line in lines:
            seg_words = line.split()
            sample_words = [w for w in seg_words[1:] if w in vocabs]
            if not sample_words: continue
            fw.write('{} {}\n'.format(seg_words[0], ' '.join(sample_words)))
        fw.close()

    @classmethod
    def get_corpus_path(cls, sample=None):
        """
        获取语料路径
        """
        config = get_config()
        if sample:
            corpus_path = config.get('train', 'sample_corpus_path')
        else:
            corpus_path = config.get('train', 'seg_corpus_path')
        return corpus_path


def get_dictionary():
    """
    单例词典获取
    """
    global __dictionary
    if not __dictionary:
        __dictionary = Dictionary
    return __dictionary

# -*- coding: utf-8 -*-
"""
CORPUS
-------
对语料处理的封装
"""
import pickle
import numpy as np
from datetime import datetime
from collections import Counter
from config import get_config
from radical import get_radical

__corpus = None


class Feature:

    _config = get_config()

    @classmethod
    def save_feature_to_file(cls, content, path):
        """
        序列化特征
        """
        f = open(path, 'wb')
        pickle.dump(content, f)
        f.close()

    @classmethod
    def read_feature_from_file(cls, path):
        """
        反序列化特征
        """
        f = open(path)
        content = pickle.load(f)
        f.close()
        return content

    @classmethod
    def load_pre_word_to_pos_feature(cls):
        """
        加载前一个词与当前词词性关系的特征
        """
        feature_dict_path = cls._config.get('postagger', 'pre_word_to_pos_feature_dict')
        feature_path = cls._config.get('postagger', 'pre_word_to_pos_feature')
        feature_dict = cls.read_feature_from_file(feature_dict_path)
        feature = cls.read_feature_from_file(feature_path)
        return feature_dict, feature

    @classmethod
    def load_word_to_pos_feature(cls):
        """
        加载当前词与当前词性关系的特征
        """
        feature_dict_path = cls._config.get('postagger', 'word_to_pos_feature_dict')
        feature_path = cls._config.get('postagger', 'word_to_pos_feature')
        feature_dict = cls.read_feature_from_file(feature_dict_path)
        feature = cls.read_feature_from_file(feature_path)
        return feature_dict, feature

    @classmethod
    def load_las_word_to_pos_feature(cls):
        """
        加载后一个词与当前词词性关系的特征
        """
        feature_dict_path = cls._config.get('postagger', 'las_word_to_pos_feature_dict')
        feature_path = cls._config.get('postagger', 'las_word_to_pos_feature')
        feature_dict = cls.read_feature_from_file(feature_dict_path)
        feature = cls.read_feature_from_file(feature_path)
        return feature_dict, feature

    @classmethod
    def load_pre_pos_to_pos_feature(cls):
        """
        加载前一个词词性与当前词词性关系的特征
        """
        feature_dict_path = cls._config.get('postagger', 'pre_pos_to_pos_feature_dict')
        feature_path = cls._config.get('postagger', 'pre_pos_to_pos_feature')
        feature_dict = cls.read_feature_from_file(feature_dict_path)
        feature = cls.read_feature_from_file(feature_path)
        return feature_dict, feature

    @classmethod
    def load_radical_to_pos_feature(cls):
        """
        加载当前词部首当前词词性关系的特征
        """
        feature_dict_path = cls._config.get('postagger', 'radical_to_pos_feature_dict')
        feature_path = cls._config.get('postagger', 'radical_to_pos_feature')
        feature_dict = cls.read_feature_from_file(feature_dict_path)
        feature = cls.read_feature_from_file(feature_path)
        return feature_dict, feature

    def __init__(self):
        raise Exception("This class have not element method.")


class Corpus:

    _words = []
    _pos = []
    _vocabs = []
    _states = []
    _word_to_radical = {}
    _puns = set(u"？?!！·【】、；，。、\s+\t+~@#$%^&*()_+{}|:\"<"
                u"~@#￥%……&*（）——+{}|：“”‘’《》>`\-=\[\]\\\\;',\./■")
    _config = get_config()
    _feature = Feature
    _radical = get_radical()

    @classmethod
    def initialize(cls):
        """
        初始化
        """
        train_corpus_path = cls._config.get('postagger', 'train_corpus_path')
        lines = cls.read_corpus_from_file(train_corpus_path)
        cls.corpus_to_words_at_pos(lines)
        cls.gen_vocabs_at_state()

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
    def corpus_to_words_at_pos(cls, lines):
        """
        词对应词性
        """
        for line in lines:
            words = [u'<BOS>']+[word.split('/')[0] for word in line.decode('utf-8').strip().split(' ')[1:] if word]+[u'<EOS>']
            pos = [u'un']+[word.split('/')[1] for word in line.decode('utf-8').strip().split(' ')[1:] if word]+[u'un']
            if len(words) != len(pos):continue
            cls._words.extend(words)
            cls._pos.extend(pos)

    @classmethod
    def gen_vocabs_at_state(cls):
        """
        语料词典、词性词典
        """
        cls._vocabs = list(set(cls._words))+[u'UNK']
        cls._states = list(set(cls._pos))
        cls._word_to_radical = {vocab: cls._radical.get_radical(vocab) for vocab in set(''.join(cls._vocabs))}

    @classmethod
    def cal_feature(cls):
        """
        特征计算
        """
        cls.cal_pre_word_to_pos_feature()
        cls.cal_word_to_pos_feature()
        cls.cal_las_word_to_pos_feature()
        cls.cal_pre_pos_to_pos_feature()
        cls.cal_radical_to_pos_feature()

    @classmethod
    def cal_pre_word_to_pos_feature(cls):
        """
        计算前一个词与当前词词性关系的特征
        """
        pos_dict = {state: 0.0 for state in cls._states}
        counts = {vocab: dict(pos_dict) for vocab in cls._vocabs}
        counter = Counter(cls._words)
        counter[u'UNK'] = 1
        for index in range(1, len(cls._pos)):
            counts[cls._words[index-1]][cls._pos[index]] += 1.0
        feature_dict_path = cls._config.get('postagger', 'pre_word_to_pos_feature_dict')
        feature_path = cls._config.get('postagger', 'pre_word_to_pos_feature')
        cls.process_feature(counter, counts, feature_dict_path, feature_path)

    @classmethod
    def cal_word_to_pos_feature(cls):
        """
        计算当前词与当前词性关系的特征
        """
        pos_dict = {state: 0.0 for state in cls._states}
        counts = {vocab: dict(pos_dict) for vocab in cls._vocabs}
        counter = Counter(cls._words)
        counter[u'UNK'] = 1
        for index in range(len(cls._pos)):
            counts[cls._words[index]][cls._pos[index]] += 1.0
        feature_dict_path = cls._config.get('postagger', 'word_to_pos_feature_dict')
        feature_path = cls._config.get('postagger', 'word_to_pos_feature')
        cls.process_feature(counter, counts, feature_dict_path, feature_path)

    @classmethod
    def cal_las_word_to_pos_feature(cls):
        """
        计算后一个词与当前词词性关系的特征
        """
        pos_dict = {state: 0.0 for state in cls._states}
        counts = {vocab: dict(pos_dict) for vocab in cls._vocabs}
        counter = Counter(cls._words)
        counter[u'UNK'] = 1
        for index in range(len(cls._pos)-1):
            counts[cls._words[index+1]][cls._pos[index]] += 1.0
        feature_dict_path = cls._config.get('postagger', 'las_word_to_pos_feature_dict')
        feature_path = cls._config.get('postagger', 'las_word_to_pos_feature')
        cls.process_feature(counter, counts, feature_dict_path, feature_path)

    @classmethod
    def cal_pre_pos_to_pos_feature(cls):
        """
        计算前一个词词性与当前词词性关系的特征
        """
        pos_dict = {state: 0.0 for state in cls._states}
        counts = {state: dict(pos_dict) for state in cls._states}
        counter = Counter(cls._pos)
        for index in range(1, len(cls._pos)):
            counts[cls._pos[index-1]][cls._pos[index]] += 1.0
        feature_dict_path = cls._config.get('postagger', 'pre_pos_to_pos_feature_dict')
        feature_path = cls._config.get('postagger', 'pre_pos_to_pos_feature')
        cls.process_feature(counter, counts, feature_dict_path, feature_path)

    @classmethod
    def cal_radical_to_pos_feature(cls):
        """
        计算当前词部首与当前词词性关系的特征 
        """
        pos_dict = {state: 0.0 for state in cls._states}
        radicals = [[cls._word_to_radical[word] for word in words] for words in cls._words]
        counts = {v: dict(pos_dict) for _, v in cls._word_to_radical.items()}
        counter = Counter(u''.join([u''.join(radical) for radical in radicals]))
        for index in range(len(cls._pos)):
            for i in range(len(radicals[index])):
                counts[radicals[index][i]][cls._pos[index]] += 1.0
        feature_dict_path = cls._config.get('postagger', 'radical_to_pos_feature_dict')
        feature_path = cls._config.get('postagger', 'radical_to_pos_feature')
        cls.process_feature(counter, counts, feature_dict_path, feature_path)

    @classmethod
    def cal_feature_list(cls, feature_dict):
        """
        特征列表 
        """
        feature_list = []
        for k, v in feature_dict.items():
            for kk, vv in v.items():
                feature_list.append((k, kk))
        return feature_list

    @classmethod
    def process_feature(cls, counter, counts, feature_dict_path, feature_path):
        """
        特征筛选、特征值计算及保存 
        """
        limit = int(cls._config.get('postagger', 'feature_limit'))
        feature_dict = {k: {kk: vv/counter[k] for kk, vv in v.items() if vv > limit} for k, v in counts.items()}
        feature_dict = {k: v for k, v in feature_dict.items() if v}
        feature = cls.cal_feature_list(feature_dict)
        cls._feature.save_feature_to_file(feature_dict, feature_dict_path)
        cls._feature.save_feature_to_file(feature, feature_path)

    @classmethod
    def load_feature(cls):
        """
        加载特征
        """
        pwtp_dict, pwtp = cls._feature.load_pre_word_to_pos_feature()
        wtp_dict, wtp = cls._feature.load_word_to_pos_feature()
        lwtp_dict, lwtp = cls._feature.load_las_word_to_pos_feature()
        pptp_dict, pptp = cls._feature.load_pre_pos_to_pos_feature()
        rtp_dict, rtp = cls._feature.load_radical_to_pos_feature()
        return pwtp_dict, pwtp, wtp_dict, wtp, lwtp_dict, lwtp, pptp_dict, pptp, rtp_dict, rtp

    @classmethod
    def filter_feature(cls, feature, x, feature_dict, feature_list):
        if feature in feature_list:
            x[feature_list.index(feature)] = feature_dict[feature[0]][feature[1]]

    @classmethod
    def get_batch_x(cls, words, pos, index, features):
        """
        特征匹配
        """
        pwtp_dict, pwtp, wtp_dict, wtp, lwtp_dict, lwtp, pptp_dict, pptp, rtp_dict, rtp = features
        pwtp_x, wtp_x, lwtp_x, pptp_x, rtp_x = [0.0] * len(pwtp), [0.0] * len(wtp), [0.0] * len(lwtp), [0.0] * len(pptp), [0.0] * len(rtp)
        for state in cls._states:
            cls.filter_feature((words[index][0], state), pwtp_x, pwtp_dict, pwtp)
            cls.filter_feature((words[index][1], state), wtp_x, wtp_dict, wtp)
            cls.filter_feature((state, words[index][2]), lwtp_x, lwtp_dict, lwtp)
            cls.filter_feature((pos[index][0], state), pptp_x, pptp_dict, pptp)
            for word in words[index][1]:
                cls.filter_feature((cls._word_to_radical[word], state), rtp_x, rtp_dict, rtp)
        return pwtp_x + wtp_x + lwtp_x + pptp_x + rtp_x

    @classmethod
    def feature_generator(cls):
        """
        语料特征生成器 
        """
        words, pos = cls.segment_by_window()
        batch_size = int(cls._config.get('model', 'batch_size'))
        chunk_size = len(words)/batch_size
        generator_list = [(i*batch_size) - 1 for i in range(chunk_size) if i]
        features = cls.load_feature()
        batch_x, batch_y = [], []
        for index in range(len(words)):
            x = cls.get_batch_x(words, pos, index, features)
            batch_x.append(x)
            batch_y.append(cls._states.index(pos[index][1]))
            if index in generator_list:
                print('{} Feature_size{}, Batch_size:{}, Chunk_count:{}, Now_chunk:{}'.format(
                    datetime.now().strftime('%c'), len(x), batch_size, chunk_size, (index+1)/batch_size))
                yield np.array(batch_x), np.array(batch_y)
                batch_x, batch_y = [], []

    @classmethod
    def segment_by_window(cls, words_list=None, window=3):
        """
        窗口切分
        """
        words, pos = [], []
        pre_words = cls._words if not words_list else words_list
        begin, end = 0, window
        for _ in range(1, len(pre_words)):
            if end > len(pre_words): break
            words.append(pre_words[begin:end])
            if not words_list: pos.append(cls._pos[begin:end])
            begin = begin + 1 if words[-1] != u'<EOS>' else begin + window
            end = end + 1 if words[-1] != u'<EOS>' else end + window
        return words, pos

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

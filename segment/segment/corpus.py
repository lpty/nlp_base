# -*- coding: utf-8 -*-
"""
CORPUS
-------
对语料处理的封装
"""
import pickle
from collections import Counter
from config import get_config

__corpus = None


class Corpus:

    _words = []
    _states = []
    _vocab = set([])
    _puns = set(u"？?!！·【】、；，。、\s+\t+~@#$%^&*()_+{}|:\"<"
                u"~@#￥%……&*（）——+{}|：“”‘’《》>`\-=\[\]\\\\;',\./■")

    @classmethod
    def initialize(cls):
        """
        初始化
        """
        config = get_config()
        train_corpus_path = config.get('segment', 'train_corpus_path')
        cls.read_corpus_from_file(train_corpus_path)
        cls.gen_vocabs()

    @classmethod
    def is_puns(cls, c):
        """
        判断是否符号
        """
        return c in cls._puns

    @classmethod
    def gen_vocabs(cls):
        """
        生成词典
        """
        cls._vocab = list(set(cls._words))+[u'<UNK>']

    @classmethod
    def read_corpus_from_file(cls, file_path):
        """
        读取语料
        """
        f = open(file_path, 'r')
        lines = f.readlines()
        for line in lines:
            cls._words.extend([word for word in line.decode('gbk').strip().split(' ') if word and not cls.is_puns(word)])
        f.close()

    @classmethod
    def word_to_states(cls, word):
        """
        词对应状态转换 
        """
        word_len = len(word)
        if word_len == 1:
            cls._states.append('S')
        else:
            state = ['M'] * word_len
            state[0] = 'B'
            state[-1] = 'E'
            cls._states.append(''.join(state))

    @classmethod
    def cal_init_state(cls):
        """
        计算初始概率
        """
        init_counts = {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0}
        for state in cls._states:
            init_counts[state[0]] += 1.0
        words_count = len(cls._words)
        # init_state = {k: log((v+1)/words_count) for k, v in init_counts.items()}
        init_state = {k: (v+1)/words_count for k, v in init_counts.items()}
        return init_state

    @classmethod
    def cal_trans_state(cls):
        """
        计算状态转移概率 
        """
        trans_counts = {'S': {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0},
                        'B': {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0},
                        'M': {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0},
                        'E': {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0}}
        states = ''.join(cls._states)
        counter = Counter(states)
        for index in range(len(states)):
            if index+1 == len(states): continue
            trans_counts[states[index]][states[index+1]] += 1.0
        # trans_state = {k: {kk: log((vv+1)/counter[k]) for kk, vv in v.items()} for k, v in trans_counts.items()}
        trans_state = {k: {kk: (vv+1)/counter[k] for kk, vv in v.items()} for k, v in trans_counts.items()}
        return trans_state

    @classmethod
    def cal_emit_state(cls):
        """
        计算观测概率
        """
        word_dict = {word: 0.0 for word in ''.join(cls._vocab)}
        emit_counts = {'S': dict(word_dict), 'B': dict(word_dict), 'M': dict(word_dict), 'E': dict(word_dict)}
        states = ''.join(cls._states)
        counter = Counter(states)
        for index in range(len(cls._states)):
            for i in range(len(cls._states[index])):
                emit_counts[cls._states[index][i]][cls._words[index][i]] += 1
        # emit_state = {k: {kk: log((vv+1)/counter[k]) for kk, vv in v.items()} for k, v in emit_counts.items()}
        emit_state = {k: {kk: (vv+1)/counter[k] for kk, vv in v.items()} for k, v in emit_counts.items()}
        return emit_state

    @classmethod
    def cal_state(cls):
        """
        计算三类状态概率 
        """
        for word in cls._words:
            cls.word_to_states(word)
        init_state = cls.cal_init_state()
        trans_state = cls.cal_trans_state()
        emit_state = cls.cal_emit_state()
        cls.save_state(init_state, trans_state, emit_state)

    @classmethod
    def save_state_to_file(cls, content, path):
        """
        保存到本地文件
        """
        f = open(path, 'wb')
        pickle.dump(content, f)
        f.close()

    @classmethod
    def read_state_from_file(cls, state_path):
        """
        读取文件
        """
        f = open(state_path)
        content = pickle.load(f)
        f.close()
        return content

    @classmethod
    def save_state(cls, init_state, trans_state, emit_state):
        """
        保存状态概率 
        """
        config = get_config()
        init_state_path = config.get('segment', 'init_state_path')
        trans_state_path = config.get('segment', 'trans_state_path')
        emit_state_path = config.get('segment', 'emit_state_path')
        cls.save_state_to_file(init_state, init_state_path)
        cls.save_state_to_file(trans_state, trans_state_path)
        cls.save_state_to_file(emit_state, emit_state_path)

    @classmethod
    def get_state(cls, name):
        """
        获取状态概率
        """
        config = get_config()
        if name == 'init':
            state_path = config.get('segment', 'init_state_path')
        elif name == 'trans':
            state_path = config.get('segment', 'trans_state_path')
        elif name == 'emit':
            state_path = config.get('segment', 'emit_state_path')
        else:
            raise ValueError('state name must in ["init", "trans", "emit"].')
        state = cls.read_state_from_file(state_path)
        return state

    @classmethod
    def process_content(cls, lines):
        return [''.join([word for word in line.decode('gbk').strip() if not cls.is_puns(word)]) for line in lines]

    @classmethod
    def get_test_corpus(cls, name):
        """
        获取测试语料 
        """
        config = get_config()
        if name == 'test':
            path = config.get('segment', 'test_corpus_path')
        elif name == 'test_gold':
            path = config.get('segment', 'test_corpus_gold_path')
        else:
            raise ValueError('test or test_gold')
        f = open(path, 'r')
        lines = f.readlines()
        corpus = cls.process_content(lines)
        f.close()
        return corpus

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

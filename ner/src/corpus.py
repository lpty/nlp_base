# -*- coding: utf-8 -*-
"""
CORPUS
-------
对语料处理的封装
"""
import re
from ner.src.config import get_config
from ner.src.util import q_to_b

__corpus = None


class Corpus:

    _config = get_config()
    _maps = {u't': u'T',
             u'nr': u'PER',
             u'ns': u'ORG',
             u'nt': u'LOC'}

    @classmethod
    def pre_process(cls):
        """
        语料预处理 
        """
        train_corpus_path = cls._config.get('ner', 'train_corpus_path')
        lines = cls.read_corpus_from_file(train_corpus_path)
        new_lines = []
        for line in lines:
            words = q_to_b(line.decode('utf-8').strip()).split(u'  ')
            pro_words = cls.process_t(words)
            pro_words = cls.process_nr(pro_words)
            pro_words = cls.process_k(pro_words)
            new_lines.append('  '.join(pro_words[1:]))
        process_corpus_path = cls._config.get('ner', 'process_corpus_path')
        cls.write_corpus_to_file(data='\n'.join(new_lines).encode('utf-8'), file_path=process_corpus_path)

    @classmethod
    def process_k(cls, words):
        """
        处理大粒度分词 
        """
        pro_words = []
        index = 0
        temp = u''
        while True:
            word = words[index] if index < len(words) else u''
            if u'[' in word:
                temp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=word.replace(u'[', u''))
            elif u']' in word:
                w = word.split(u']')
                temp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=w[0])
                pro_words.append(temp+u'/'+w[1])
                temp = u''
            elif temp:
                temp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=word)
            elif word:
                pro_words.append(word)
            else:
                break
            index += 1
        return pro_words

    @classmethod
    def process_nr(cls, words):
        """
        处理姓名 
        """
        pro_words = []
        index = 0
        while True:
            word = words[index] if index < len(words) else u''
            if u'/nr' in word:
                next_index = index + 1
                if next_index < len(words) and u'/nr' in words[next_index]:
                    pro_words.append(word.replace(u'/nr', u'') + words[next_index])
                    index = next_index
                else:
                    pro_words.append(word)
            elif word:
                pro_words.append(word)
            else:
                break
            index += 1
        return pro_words

    @classmethod
    def process_t(cls, words):
        """
        处理时间
        """
        pro_words = []
        index = 0
        temp = u''
        while True:
            word = words[index] if index < len(words) else u''
            if u'/t' in word:
                temp = temp.replace(u'/t', u'') + word
            elif temp:
                pro_words.append(temp)
                pro_words.append(word)
                temp = u''
            elif word:
                pro_words.append(word)
            else:
                break
            index += 1
        return pro_words

    @classmethod
    def pos_to_tag(cls, p):
        """
        由词性提取标签
        """
        t = cls._maps.get(p, None)
        return t if t else u'O'

    @classmethod
    def tag_perform(cls, tag, index):
        """
        标签使用BIO模式
        """
        if index == 0 and tag != u'O':
            return u'B_{}'.format(tag)
        elif tag != u'O':
            return u'I_{}'.format(tag)
        else:
            return tag

    @classmethod
    def pos_perform(cls, pos):
        """
        去除词性携带的标签先验知识
        """
        if pos in cls._maps.keys() and pos != u't':
            return u'n'
        else:
            return pos

    @classmethod
    def initialize(cls):
        """
        初始化 
        """
        corpus_path = cls._config.get('ner', 'process_corpus_path')
        lines = cls.read_corpus_from_file(corpus_path)
        words_list = [line.decode('utf-8').strip().split('  ') for line in lines if line.strip()]
        del lines
        cls.init_sequence(words_list)

    @classmethod
    def init_sequence(cls, words_list):
        """
        初始化字序列、词性序列、标记序列 
        """
        words_seq = [[word.split(u'/')[0] for word in words] for words in words_list]
        pos_seq = [[word.split(u'/')[1] for word in words] for words in words_list]
        tag_seq = [[cls.pos_to_tag(p) for p in pos] for pos in pos_seq]
        cls.pos_seq = [[[pos_seq[index][i] for _ in range(len(words_seq[index][i]))]
                        for i in range(len(pos_seq[index]))] for index in range(len(pos_seq))]
        cls.tag_seq = [[[cls.tag_perform(tag_seq[index][i], w) for w in range(len(words_seq[index][i]))]
                        for i in range(len(tag_seq[index]))] for index in range(len(tag_seq))]
        cls.pos_seq = [[u'un']+[cls.pos_perform(p) for pos in pos_seq for p in pos]+[u'un'] for pos_seq in cls.pos_seq]
        cls.tag_seq = [[t for tag in tag_seq for t in tag] for tag_seq in cls.tag_seq]
        cls.word_seq = [[u'<BOS>']+[w for word in word_seq for w in word]+[u'<EOS>'] for word_seq in words_seq]

    @classmethod
    def segment_by_window(cls, words_list=None, window=3):
        """
        窗口切分
        """
        words = []
        begin, end = 0, window
        for _ in range(1, len(words_list)):
            if end > len(words_list): break
            words.append(words_list[begin:end])
            begin = begin + 1
            end = end + 1
        return words

    @classmethod
    def extract_feature(cls, word_grams):
        """
        特征选取
        """
        features, feature_list = [], []
        for index in range(len(word_grams)):
            for i in range(len(word_grams[index])):
                word_gram = word_grams[index][i]
                feature = {u'w-1': word_gram[0], u'w': word_gram[1], u'w+1': word_gram[2],
                           u'w-1:w': word_gram[0]+word_gram[1], u'w:w+1': word_gram[1]+word_gram[2],
                           # u'p-1': cls.pos_seq[index][i], u'p': cls.pos_seq[index][i+1],
                           # u'p+1': cls.pos_seq[index][i+2],
                           # u'p-1:p': cls.pos_seq[index][i]+cls.pos_seq[index][i+1],
                           # u'p:p+1': cls.pos_seq[index][i+1]+cls.pos_seq[index][i+2],
                           u'bias': 1.0}
                feature_list.append(feature)
            features.append(feature_list)
            feature_list = []
        return features

    @classmethod
    def generator(cls):
        """
        训练数据
        """
        word_grams = [cls.segment_by_window(word_list) for word_list in cls.word_seq]
        features = cls.extract_feature(word_grams)
        return features, cls.tag_seq

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

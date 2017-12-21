# -*- coding: utf-8 -*-
"""
SEGMENT
--------
封装hmm分词模型
"""
import numpy as np
from hmmlearn.hmm import MultinomialHMM
from corpus import get_corpus

__model = None


class Segment:

    def __init__(self):
        self.corpus = get_corpus()
        self.states, self.init_p = self.get_init_state()
        self.trans_p = self.get_trans_state()
        self.vocabs, self.emit_p = self.get_emit_state()
        self.model = self.get_model()

    def get_init_state(self):
        """
        获取初始概率，转为hmm模型接受数据形式
        """
        states = ['S', 'B', 'M', 'E']
        init_state = self.corpus.get_state('init')
        init_p = np.array([init_state[s] for s in states])
        return states, init_p

    def get_trans_state(self):
        """
        获取转移概率，转为hmm模型接受数据形式
        """
        trans_state = self.corpus.get_state('trans')
        trans_p = np.array([[trans_state[s][ss] for ss in self.states] for s in self.states])
        return trans_p

    def get_emit_state(self):
        """
        获取发射概率，转为hmm模型接受数据形式
        """
        emit_state = self.corpus.get_state('emit')
        vocabs = []
        for s in self.states:
            vocabs.extend([k for k, v in emit_state[s].items()])
        vocabs = list(set(vocabs))
        emit_p = np.array([[emit_state[s][w] for w in vocabs] for s in self.states])
        return vocabs, emit_p

    def get_model(self):
        """
        初始化hmm模型
        """
        model = MultinomialHMM(n_components=len(self.states))
        model.startprob_ = self.init_p
        model.transmat_ = self.trans_p
        model.emissionprob_ = self.emit_p
        return model

    def pre_process(self, word):
        """
        未知字处理 
        """
        if word in self.vocabs:
            return self.vocabs.index(word)
        else:
            return len(self.vocabs)-1

    def cut(self, sentence):
        """
        分词
        """
        seen_n = np.array([[self.pre_process(w) for w in sentence]]).T
        log_p, b = self.model.decode(seen_n, algorithm='viterbi')
        states = map(lambda x: self.states[x], b)
        cut_sentence = ''
        for index in range(len(states)):
            if states[index] in ('S', 'E'):
                cut_sentence += sentence[index]+' '
            else:
                cut_sentence += sentence[index]
        return cut_sentence

    @staticmethod
    def stats(cut_corpus, gold_corpus):
        """
        正确率、召回率、F1
        """
        success_count = 0
        cut_count = 0
        gold_count = 0
        for index in range(len(cut_corpus)):
            cut_sentence = cut_corpus[index].split(' ')
            gold_sentence = gold_corpus[index].split(' ')
            cut_count += len(cut_sentence)
            gold_count += len(gold_sentence)
            for word in cut_sentence:
                if word in gold_sentence:
                    success_count += 1
        recall = float(success_count)/float(gold_count)
        precision = float(success_count)/float(cut_count)
        f1 = (2*recall*precision)/(recall+precision)
        return [precision, recall, f1]

    def test(self):
        """
        分词测试
        """
        test_corpus = self.corpus.get_test_corpus('test')
        gold_corpus = [sentence.replace('  ', ' ').strip() for sentence in self.corpus.get_test_corpus('test_gold') if sentence]
        cut_corpus = [self.cut(sentence).strip() for sentence in test_corpus if sentence]
        result = self.stats(cut_corpus, gold_corpus)
        return result


def get_model():
    """
    单例模型获取
    """
    global __model
    if not __model:
        __model = Segment()
    return __model

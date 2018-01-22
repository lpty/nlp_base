# -*- coding: utf-8 -*-
"""
NER
----
封装条件随机场命名实体识别
"""
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.externals import joblib
from util import q_to_b
from corpus import get_corpus
from config import get_config

__model = None


class NER:

    def __init__(self):
        self.corpus = get_corpus()
        self.corpus.initialize()
        self.config = get_config()
        self.model = None

    def initialize_model(self):
        """
        初始化
        """
        algorithm = self.config.get('model', 'algorithm')
        c1 = float(self.config.get('model', 'c1'))
        c2 = float(self.config.get('model', 'c2'))
        max_iterations = int(self.config.get('model', 'max_iterations'))
        self.model = sklearn_crfsuite.CRF(algorithm=algorithm, c1=c1, c2=c2,
                                          max_iterations=max_iterations, all_possible_transitions=True)

    def train(self):
        """
        训练
        """
        self.initialize_model()
        x, y = self.corpus.generator()
        x_train, y_train = x[500:], y[500:]
        x_test, y_test = x[:500], y[:500]
        self.model.fit(x_train, y_train)
        labels = list(self.model.classes_)
        labels.remove('O')
        y_predict = self.model.predict(x_test)
        metrics.flat_f1_score(y_test, y_predict, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(metrics.flat_classification_report(y_test, y_predict, labels=sorted_labels, digits=3))
        self.save_model()

    def predict(self, sentence):
        """
        预测
        """
        self.load_model()
        u_sent = q_to_b(sentence)
        word_lists = [[u'<BOS>']+[c for c in u_sent]+[u'<EOS>']]
        word_grams = [self.corpus.segment_by_window(word_list) for word_list in word_lists]
        features = self.corpus.extract_feature(word_grams)
        y_predict = self.model.predict(features)
        entity = u''
        for index in range(len(y_predict[0])):
            if y_predict[0][index] != u'O':
                if index > 0 and y_predict[0][index][-1] != y_predict[0][index-1][-1]:
                    entity += u' '
                entity += u_sent[index]
            elif entity[-1] != u' ':
                entity += u' '
        return entity

    def load_model(self, name='model'):
        """
        加载模型 
        """
        model_path = self.config.get('model', 'model_path').format(name)
        self.model = joblib.load(model_path)

    def save_model(self, name='model'):
        """
        保存模型
        """
        model_path = self.config.get('model', 'model_path').format(name)
        joblib.dump(self.model, model_path)


def get_model():
    """
    单例模型获取
    """
    global __model
    if not __model:
        __model = NER()
    return __model

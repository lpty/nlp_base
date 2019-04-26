# -*- coding: utf-8 -*-
"""
DEPPARSER
----------
封装条件随机场依存句法分析
"""
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.externals import joblib
from depparser.src.corpus import get_corpus
from depparser.src.config import get_config

__model = None


class DepParser:

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
        x_train, y_train = self.corpus.generator()
        self.model.fit(x_train, y_train)
        labels = list(self.model.classes_)
        x_test, y_test = self.corpus.generator(train=False)
        y_predict = self.model.predict(x_test)
        metrics.flat_f1_score(y_test, y_predict, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(metrics.flat_classification_report(y_test, y_predict, labels=sorted_labels, digits=3))
        self.save_model()

    def predict(self, sentences):
        """
        预测
        """
        self.load_model()
        features, _ = self.corpus.extract_feature(sentences)
        return self.model.predict(features)

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
        __model = DepParser()
    return __model

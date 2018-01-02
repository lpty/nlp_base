# -*- coding: utf-8 -*-
"""
POSTAGGER
----------
封装最大熵词性标注
"""
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from corpus import get_corpus
from config import get_config

__model = None


class PosTagger:

    def __init__(self):
        self.corpus = get_corpus()
        self.corpus.initialize()
        self.classes = len(self.corpus._states)
        self.config = get_config()
        self.model = None

    def initialize_model(self):
        """
        初始化模型
        """
        loss = self.config.get('model', 'loss')
        alpha = float(self.config.get('model', 'alpha'))
        n_jobs = int(self.config.get('model', 'n_jobs'))
        self.model = SGDClassifier(loss=loss, alpha=alpha, n_jobs=n_jobs)

    def train(self, epoch=None, show=None):
        """
        模型训练
        """
        self.initialize_model()
        g = self.corpus.feature_generator()
        batch_first_x, batch_first_y = next(g)
        self.model.partial_fit(batch_first_x, batch_first_y, classes=range(self.classes))
        train_score, progressive_validation_score = [], []
        epoch_count = 0
        for batch_x, batch_y in g:
            score = self.model.score(batch_x, batch_y)
            progressive_validation_score.append(score)
            print('{} Progressive_validation_score:{}'.format(datetime.now().strftime('%c'), score))
            self.model.partial_fit(batch_x, batch_y, classes=range(self.classes))
            score = self.model.score(batch_x, batch_y)
            train_score.append(score)
            print('{} Train_score:{}'.format(datetime.now().strftime('%c'), score))
            if epoch and epoch_count == epoch:break
            else: epoch_count += 1
        self.save_model()
        if show: self.show(train_score, progressive_validation_score)

    @staticmethod
    def show(train_score, progressive_validation_score):
        """
        绘制训练曲线
        """
        plt.plot(train_score, label="train score")
        plt.plot(progressive_validation_score, label="progressive validation score")
        plt.xlabel("Mini-batch")
        plt.ylabel("Score")
        plt.legend(loc='best')
        plt.show()

    def predict(self, sentence):
        """
        词性预测
        """
        if not self.model:
            self.load_model()
        origin_words_list = [word for word in sentence.split(' ') if word]
        words_list = [u'<BOS>'] + origin_words_list + [u'<EOS>']
        words, _ = self.corpus.segment_by_window(words_list)
        pos = [[self.corpus._states.index(u'un')]]
        features = self.corpus.load_feature()
        for index in range(len(words)):
            x = self.corpus.get_batch_x(words, pos, index, features)
            pos.append(list(self.model.predict(x)))
        tagger_list = [self.corpus._states[p[0]] for p in pos[1:]]
        return u'  '.join([u'{}/{}'.format(origin_words_list[index], tagger_list[index]) for index in range(len(origin_words_list))])

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
        __model = PosTagger()
    return __model

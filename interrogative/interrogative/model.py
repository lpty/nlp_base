# -*- coding: utf-8 -*-
"""
INTERROGATIVE
--------------
for model parameters selecting and model training
"""
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from sklearn.externals import joblib
from corpus import get_corpus
from config import get_config
from util import to_json

__model = None


class Interrogative:
    """
    Perform estimator of interrogative sentence model
    """
    def __init__(self):
        self.corpus = get_corpus()
        self.config = get_config()
        self.model = None
        self.vectorizer = None

    def initialize_model(self):
        """
        corpus feature and model param initialize
        """
        train, label = self.corpus.generator()
        self.train_matrix = xgb.DMatrix(train, label=label)

        # read params from config
        self.max_depth = to_json(self.config.get('model', 'max_depth'))
        self.eta = to_json(self.config.get('model', 'eta'))
        self.subsample = to_json(self.config.get('model', 'subsample'))
        self.objective = to_json(self.config.get('model', 'objective'))
        self.silent = to_json(self.config.get('model', 'silent'))
        self.num_boost_round = int(self.config.get('model', 'num_boost_round'))
        self.nfold = int(self.config.get('model', 'nfold'))
        self.stratified = True if int(self.config.get('model', 'stratified')) else False
        self.metrics = self.config.get('model', 'metrics')
        self.early_stopping_rounds = int(self.config.get('model', 'early_stopping_rounds'))

    def model_param_select(self):
        """
        k-folds cross validation to select the best param 
        """
        params = {'max_depth': self.max_depth,
                  'eta': self.eta,
                  'subsample': self.subsample,
                  'objective': self.objective,
                  'silent': self.silent}
        best_auc, best_param, best_iter_round = 0, {}, 0
        param_grid = ParameterGrid(params)
        for i, param in enumerate(param_grid):
            cv_result = xgb.cv(param, self.train_matrix,
                               num_boost_round=self.num_boost_round,  # max iter round
                               nfold=self.nfold,
                               stratified=self.stratified,
                               metrics=self.metrics,  # metrics focus on
                               early_stopping_rounds=self.early_stopping_rounds)  # stop when metrics not get better
            cur_auc = cv_result.ix[len(cv_result)-1, 0]
            cur_iter_round = len(cv_result)
            if cur_auc > best_auc:
                best_auc, best_param, best_iter_round = cur_auc, param, cur_iter_round
            print('Param select {}, auc: {}, iter_round: {}, params: {}, now best auc: {}'
                  .format(i, cur_auc, cur_iter_round, param, best_auc))
        return best_auc, best_param, best_iter_round

    def train(self):
        """
        model training
        """
        self.initialize_model()
        _, best_param, best_iter_round = self.model_param_select()
        self.model = xgb.train(dtrain=self.train_matrix, params=best_param, num_boost_round=best_iter_round)
        self.save_model()

    def predict(self, sentence):
        """
        predict the prob of sentence if it is interrogative.
        """
        assert isinstance(sentence, unicode), 'Sentence must be unicode.'
        if not self.model:
            self.load_model()
        feature_matrix = self.transform(sentence)
        prob = self.model.predict(feature_matrix)
        return prob

    def transform(self, sentence):
        """
        transform sentence to tf-idf features
        """
        if not self.vectorizer:
            self.load_model('tfidf_vectorizer', 'vectorizer')
        feature_matrix = self.vectorizer.transform([sentence])
        return xgb.DMatrix(feature_matrix)

    def load_model(self, name='model', model='model'):
        """
        load model from local 
        """
        model_path = self.config.get('model', 'model_path').format(name)
        self.__dict__[model] = joblib.load(model_path)

    def save_model(self, name='model'):
        """
        save model to local
        """
        model_path = self.config.get('model', 'model_path').format(name)
        joblib.dump(self.model, model_path)


def get_model():
    """
    singleton object generator
    """
    global __model
    if not __model:
        __model = Interrogative()
    return __model

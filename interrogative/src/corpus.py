# -*- coding: utf-8 -*-
"""
CORPUS
-------
For corpus pre-process and features extraction
"""
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from interrogative.src.config import get_config
from interrogative.src.util import analyzer

__corpus = None


class Corpus:
    _config = get_config()

    @classmethod
    def read_corpus_from_file(cls, file_path):
        """
        read interrogative train data from local
        """
        return pd.read_csv(file_path)

    @classmethod
    def perform_word_segment(cls, corpus):
        """
        process word segmenting use jieba tokenizer
        """
        tokenizer = jieba.Tokenizer()
        corpus['tokens'] = corpus.content.apply(lambda x: list(tokenizer.cut(x)))
        return corpus

    @classmethod
    def feature_extract(cls, train, tfidf_save=True):
        """
        feature engineering, extract Tf-idf feature
        """
        vectorizer = TfidfVectorizer(smooth_idf=True,
                                     analyzer=analyzer,
                                     ngram_range=(1, 1),
                                     min_df=1, norm='l1')
        sparse_vector = vectorizer.fit_transform(train.tokens.apply(lambda x: ' '.join(x)).tolist())
        label = train.label.tolist()

        # tf-idf vectorizer save
        if tfidf_save:
            joblib.dump(vectorizer, cls._config.get('interrogative', 'tfidf_vectorizer_path'))

        return sparse_vector, label

    @classmethod
    def generator(cls):
        """
        pre-process corpus and extract features
        """
        corpus_path = cls._config.get('interrogative', 'corpus_path')
        corpus = cls.read_corpus_from_file(corpus_path)
        train = cls.perform_word_segment(corpus)
        return cls.feature_extract(train)

    def __init__(self):
        raise NotImplementedError()


def get_corpus():
    """
    singleton object generator
    """
    global __corpus
    if not __corpus:
        __corpus = Corpus
    return __corpus


if __name__ == '__main__':
    get_corpus().generator()

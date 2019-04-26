# -*- coding: utf-8 -*-
"""
CONFIG
------
对配置的封装
"""
config_instance = None


class Config:

    def __init__(self):
        self.config_dict = {
            'postagger': {
                'train_corpus_path': 'data/rmrb1998.txt',
                'dict_path': 'data/xinhua.csv',
                'pre_word_to_pos_feature_dict': 'data/pre_word_to_pos_feature_dict.pkl',
                'pre_word_to_pos_feature': 'data/pre_word_to_pos_feature.pkl',

                'word_to_pos_feature_dict': 'data/word_to_pos_feature_dict.pkl',
                'word_to_pos_feature': 'data/word_to_pos_feature.pkl',

                'pre_pos_to_pos_feature_dict': 'data/pre_pos_to_pos_feature_dict.pkl',
                'pre_pos_to_pos_feature': 'data/pre_pos_to_pos_feature.pkl',

                'radical_to_pos_feature_dict': 'data/radical_to_pos_feature_dict.pkl',
                'radical_to_pos_feature': 'data/radical_to_pos_feature.pkl',

                'feature_limit': 5,
            },
            'model': {
                'batch_size': 256,
                'loss': 'log',
                'alpha': 0.001,
                'n_jobs': -1,
                'model_path': 'data/{}.pkl'
            }
        }

    def get(self, section_name, arg_name):
        return self.config_dict[section_name][arg_name]


def get_config():
    global config_instance
    if not config_instance:
        config_instance = Config()
    return config_instance

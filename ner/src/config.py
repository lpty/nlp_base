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
            'ner': {
                'train_corpus_path': 'data/rmrb199801.txt',
                'process_corpus_path': 'data/rmrb.txt',
            },
            'model': {
                'algorithm': 'lbfgs',
                'c1': 0.1,
                'c2': 0.1,
                'max_iterations': 100,
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

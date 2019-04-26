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
            'segment': {
                'train_corpus_path': 'data/msr_training.txt',
                'test_corpus_path': 'data/msr_test.txt',
                'test_corpus_gold_path': 'data/msr_test_gold.txt',

                'init_state_path': 'data/init_state.pkl',
                'trans_state_path': 'data/trans_state.pkl',
                'emit_state_path': 'data/emit_state.pkl',
            },
        }

    def get(self, section_name, arg_name):
        return self.config_dict[section_name][arg_name]


def get_config():
    global config_instance
    if not config_instance:
        config_instance = Config()
    return config_instance

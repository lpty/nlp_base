# -*- coding: utf-8 -*-
"""
CONFIG
------
configuration
"""
from ConfigParser import ConfigParser

__config = None


def get_config(config_file_path='interrogative/conf/config.conf'):
    """
    singleton object generator
    """
    global __config
    if not __config:
        config = ConfigParser()
        config.read(config_file_path)
    else:
        config = __config
    return config

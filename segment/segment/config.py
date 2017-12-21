# -*- coding: utf-8 -*-
"""
CONFIG
------
对配置的封装
"""
from ConfigParser import ConfigParser

__config = None


def get_config(config_file_path='segment/conf/config.conf'):
    """
    单例配置获取
    """
    global __config
    if not __config:
        config = ConfigParser()
        config.read(config_file_path)
    else:
        config = __config
    return config

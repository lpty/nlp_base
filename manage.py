# -*- coding: utf-8 -*-
import sys
from sentiment.api import *


def manage():
    arg = sys.argv[1]
    if arg == 'cut':
        process_corpus()
    elif arg == 'train':
        train_model()
    elif arg == 'test':
        test_model()
    else:
        print('Args must in ["cut", "train", "test"].')
    sys.exit()

if __name__ == '__main__':
    manage()

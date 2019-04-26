import sys
from ner.src.api import *


def manage():
    arg = sys.argv[1]
    if arg == 'train':
        train()
    elif arg == 'process':
        pre_process()
    else:
        print('Args must in ["process", "train"].')
    sys.exit()


if __name__ == '__main__':
    manage()

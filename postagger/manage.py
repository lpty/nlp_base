import sys
from postagger.src.api import *


def manage():
    arg = sys.argv[1]
    if arg == 'train':
        train()
    elif arg == 'extract':
        extract_feature()
    else:
        print('Args must in ["extract", "train"].')
    sys.exit()


if __name__ == '__main__':
    manage()

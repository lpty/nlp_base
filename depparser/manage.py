import sys
from depparser.src.api import *


def manage():
    arg = sys.argv[1]
    if arg == 'train':
        train()
    elif arg == 'parser':
        parser(sys.argv[2])
    else:
        print('Args must in ["parser", "train"].')
    sys.exit()


if __name__ == '__main__':
    manage()

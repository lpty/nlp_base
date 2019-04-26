import sys
from segment.src.api import *


def manage():
    arg = sys.argv[1]
    if arg == 'train':
        train()
    elif arg == 'test':
        test()
    else:
        print('Args must in ["train", "test"].')
    sys.exit()


if __name__ == '__main__':
    manage()

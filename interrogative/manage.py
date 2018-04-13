# -*- coding: utf-8 -*-
from interrogative.api import *

train()
tag = recognize(u'今天 来 点 兔子 吗')
output = '是疑问句' if tag else '不是疑问句'
print(output)

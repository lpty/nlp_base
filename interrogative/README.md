# 中文问句判断模型
一个使用xgboost的中文问句判断模型

# 模型效果

    Param select 0, auc: 0.9793438, iter_round: 207, params: {'objective': u'binary:logistic', 'subsample': 0.5, 'eta': 0.1, 'max_depth': 4, 'silent': 1}, now best auc: 0.9793438
    Param select 1, auc: 0.9799142, iter_round: 350, params: {'objective': u'binary:logistic', 'subsample': 0.7, 'eta': 0.1, 'max_depth': 4, 'silent': 1}, now best auc: 0.9799142
    Param select 2, auc: 0.9802402, iter_round: 280, params: {'objective': u'binary:logistic', 'subsample': 1.0, 'eta': 0.1, 'max_depth': 4, 'silent': 1}, now best auc: 0.9802402
    Param select 3, auc: 0.9791936, iter_round: 197, params: {'objective': u'binary:logistic', 'subsample': 0.5, 'eta': 0.1, 'max_depth': 5, 'silent': 1}, now best auc: 0.9802402
    Param select 4, auc: 0.979731, iter_round: 223, params: {'objective': u'binary:logistic', 'subsample': 0.7, 'eta': 0.1, 'max_depth': 5, 'silent': 1}, now best auc: 0.9802402
    Param select 5, auc: 0.97988, iter_round: 199, params: {'objective': u'binary:logistic', 'subsample': 1.0, 'eta': 0.1, 'max_depth': 5, 'silent': 1}, now best auc: 0.9802402
    Param select 6, auc: 0.9792362, iter_round: 213, params: {'objective': u'binary:logistic', 'subsample': 0.5, 'eta': 0.1, 'max_depth': 6, 'silent': 1}, now best auc: 0.9802402
    Param select 7, auc: 0.9798212, iter_round: 256, params: {'objective': u'binary:logistic', 'subsample': 0.7, 'eta': 0.1, 'max_depth': 6, 'silent': 1}, now best auc: 0.9802402
    Param select 8, auc: 0.9800464, iter_round: 181, params: {'objective': u'binary:logistic', 'subsample': 1.0, 'eta': 0.1, 'max_depth': 6, 'silent': 1}, now best auc: 0.9802402
    Param select 9, auc: 0.9795114, iter_round: 463, params: {'objective': u'binary:logistic', 'subsample': 0.5, 'eta': 0.05, 'max_depth': 4, 'silent': 1}, now best auc: 0.9802402
    Param select 10, auc: 0.979777, iter_round: 443, params: {'objective': u'binary:logistic', 'subsample': 0.7, 'eta': 0.05, 'max_depth': 4, 'silent': 1}, now best auc: 0.9802402
    Param select 11, auc: 0.9802696, iter_round: 484, params: {'objective': u'binary:logistic', 'subsample': 1.0, 'eta': 0.05, 'max_depth': 4, 'silent': 1}, now best auc: 0.9802696
    Param select 12, auc: 0.9794792, iter_round: 440, params: {'objective': u'binary:logistic', 'subsample': 0.5, 'eta': 0.05, 'max_depth': 5, 'silent': 1}, now best auc: 0.9802696
    Param select 13, auc: 0.9800954, iter_round: 427, params: {'objective': u'binary:logistic', 'subsample': 0.7, 'eta': 0.05, 'max_depth': 5, 'silent': 1}, now best auc: 0.9802696
    Param select 14, auc: 0.980495, iter_round: 483, params: {'objective': u'binary:logistic', 'subsample': 1.0, 'eta': 0.05, 'max_depth': 5, 'silent': 1}, now best auc: 0.980495
    Param select 15, auc: 0.9795902, iter_round: 436, params: {'objective': u'binary:logistic', 'subsample': 0.5, 'eta': 0.05, 'max_depth': 6, 'silent': 1}, now best auc: 0.980495
    Param select 16, auc: 0.9804718, iter_round: 416, params: {'objective': u'binary:logistic', 'subsample': 0.7, 'eta': 0.05, 'max_depth': 6, 'silent': 1}, now best auc: 0.980495
    Param select 17, auc: 0.9804004, iter_round: 382, params: {'objective': u'binary:logistic', 'subsample': 1.0, 'eta': 0.05, 'max_depth': 6, 'silent': 1}, now best auc: 0.980495
    Param select 18, auc: 0.979225, iter_round: 1011, params: {'objective': u'binary:logistic', 'subsample': 0.5, 'eta': 0.02, 'max_depth': 4, 'silent': 1}, now best auc: 0.980495
    Param select 19, auc: 0.980099, iter_round: 1135, params: {'objective': u'binary:logistic', 'subsample': 0.7, 'eta': 0.02, 'max_depth': 4, 'silent': 1}, now best auc: 0.980495
    Param select 20, auc: 0.9802448, iter_round: 1207, params: {'objective': u'binary:logistic', 'subsample': 1.0, 'eta': 0.02, 'max_depth': 4, 'silent': 1}, now best auc: 0.980495
    Param select 21, auc: 0.9792566, iter_round: 762, params: {'objective': u'binary:logistic', 'subsample': 0.5, 'eta': 0.02, 'max_depth': 5, 'silent': 1}, now best auc: 0.980495
    Param select 22, auc: 0.9802228, iter_round: 852, params: {'objective': u'binary:logistic', 'subsample': 0.7, 'eta': 0.02, 'max_depth': 5, 'silent': 1}, now best auc: 0.980495
    Param select 23, auc: 0.9798456, iter_round: 831, params: {'objective': u'binary:logistic', 'subsample': 1.0, 'eta': 0.02, 'max_depth': 5, 'silent': 1}, now best auc: 0.980495
    Param select 24, auc: 0.97926, iter_round: 694, params: {'objective': u'binary:logistic', 'subsample': 0.5, 'eta': 0.02, 'max_depth': 6, 'silent': 1}, now best auc: 0.980495
    Param select 25, auc: 0.9803058, iter_round: 824, params: {'objective': u'binary:logistic', 'subsample': 0.7, 'eta': 0.02, 'max_depth': 6, 'silent': 1}, now best auc: 0.980495
    Param select 26, auc: 0.980129, iter_round: 880, params: {'objective': u'binary:logistic', 'subsample': 1.0, 'eta': 0.02, 'max_depth': 6, 'silent': 1}, now best auc: 0.980495

# 实例

    from interrogative.api import *

    train()
    tag = recognize(u'今天 来 点 兔子 吗')
    output = '是疑问句' if tag else '不是疑问句'
    print(output)

### 博客
博客地址:https://blog.csdn.net/sinat_33741547/article/details/79933376

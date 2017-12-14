# 评论情感极性判断模型
基于京东评论的情感极性判断模型，采用了fasttext进行分类。

## 模型效果
使用京东80w条评论数据训练，10ｗ条评论数据测试:
#### 模型参数

    lr = 0.01
    lr_update_rate = 100
    dim = 300
    ws = 5
    epoch = 10
    word_ngrams = 3
    loss = hs
    bucket = 2000000
    thread = 4

#### 效果

    ('precision:', 0.85055)
    ('recall:', 0.85055)
    ('examples:', 100000)

## 快速开始
#### 语料分词

    python manage.py cut

#### 模型训练

    python manage.py train

#### 模型测试

    python manage.py test

## 文档
#### 代码文档
基于sphnix生成,请确保已经安装

    cd doc
    make html

#### 博客
博客地址:http://blog.csdn.net/sinat_33741547/article/details/78803766

## 参考
#### Enriching Word Vectors with Subword Information

[1] P. Bojanowski\*, E. Grave\*, A. Joulin, T. Mikolov, [*Enriching Word Vectors with Subword Information*](https://arxiv.org/pdf/1607.04606v1.pdf)

```
@article{bojanowski2016enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.04606},
  year={2016}
}
```

#### Bag of Tricks for Efficient Text Classification

[2] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/pdf/1607.01759v2.pdf)

```
@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}
```

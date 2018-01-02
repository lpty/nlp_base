# 中文词性标注模型
一个使用最大熵的词性标注模型

# 模型效果

    01/02/18 04:25:28 Feature_size63033, Batch_size:256, Chunk_count:9121, Now_chunk:9120
    01/02/18 04:25:29 Progressive_validation_score:0.6015625
    01/02/18 04:25:31 Train_score:0.6015625

# 快速开始
## 特征提取

    python manage.py extract

## 模型训练

    python manage.py train

# 文档
### 代码文档
基于sphnix生成,请确保已经安装

    cd doc
    make html

### 博客
博客地址:http://blog.csdn.net/sinat_33741547/article/details/78949819

# 参考

	1、汉语词性标注的特征工程  于江德等
	2、一种基于改进的最大熵模型的汉语词性自动标注的新方法  赵伟等
	3、基于最大熵的哈萨克语词性标注模型  桑海岩等

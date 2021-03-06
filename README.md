# DeepLin
## 简介
* 一个高度封装的基于PyTorch的深度学习框架
* 添加大量功能


## 需求
* PyTorch
* tqdm
* sklearn
* numpy


## 特性
* 支持复现模式，帮助生成可重复的预测结果
* 无监督（标签）训练模式
* 伪样本训练模式 【实验室】
* 支持l1、l2范数
* 混合精度训练  【实验室】
* Early Stop
* 多线程模式
* 评估性能
* RAdam优化函数

## 快速上手
### 前期操作：获取数据、定义网络结构等
```Python
from DeepLin import DeepLing
import torch.nn as nn
import numpy as np


# 产生训练集、测试集
training_data = np.random.random((1000, 10))
training_label = np.random.randint(0, 2, (1000, 1))
test_data = np.random.random((500, 10))
test_label = np.random.randint(0, 2, (500, 1))


# 定义网络结构
class DemoNet(nn.Module):
    def __init__(self):
        super(DemoNet, self).__init__()
        self.modelName = "DemoNet"          # 是为了early stop保存模型参数用的，请务必起个名字
        self.fc = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        return self.fc(x)
```
### DeepLin框架步骤
> 实例化—打包—训练—打包—测试—评估
```Python

model = DeepLing(DemoNet, 'CEP', batch_size=10, optFunc='Adam', epoch=100, early_stop=10)
"""
实例化
    利用DemoNet网络进行训练与测试，
    损失函数为CrossEntropy，
    batch size设定为10，
    利用Adam进行优化，
    最大epoch为100次，
    当loss没有提升超过10次时训练提前终止。
"""

print("训练")
model.getTrainingDataLoader(training_data, training_label)
model.train()

print('测试')
model.getTestDataLoader(test_data, test_label)
test_pred = model.test()

# 评估
model.score(test_label, test_pred)
```

## 核心代码
由此可见利用DeepLin进行训练与测试非常的简洁与方便，如果需要进行个性化的定制可参考如下内容

### 实例化
```Python
model = DeepLing(modelClass, lossFunc, optFunc='RAdam',
                 epoch=10, early_stop=0, batch_size=128, lr=0.001,
                 l1=0, l2=0, weight=None,
                 p_Labels=0, a=0, T1=0, T2=0,
                 isCuda=True, isSelf=False, isMask=False, showLoss=True, opt_level=None)
```
#### 参数说明


| 形参名 | 作用 | 数据类型 | 默认值 |
| --- | --- | --- | --- |
| modelClass | 用户定义的PyTorch网络结构 | Class | 无 |
| lossFunc | 损失函数 | str或'torch.nn.modules.loss' | 无 |
| optFunc | 优化函数 | str或'torch.optim' | 'RAdam' |
| epoch | 训练次数 | int | 10 |
| early_stop | 提前终止的等待次数，当其为0时禁用提前终止 | int | 0 |
| batch_size | 一批的样本数 | int | 128 |
| lr | 学习率 | float | 0.001 |
| l1 | l1范数的权重，当其为0时不计算l1范数 | float | 0 |
| l2 | l2范数的权重，当其为0时不计算l2范数 | float | 0 |
| p_Labels | 是否使用伪标签（半监督算法）进行训练，当其为0时不使用 | 0或1 | 0 |
| a | 退火算法的默认权重 | float | 0 |
| T1 | 退火算法中第一阶段的时间 | int，且0<T1<T2 | 0 |
| T2 | 退火算法中第二阶段的时间 | int，且T1<T2<epoch | 0 |
| isCuda | 是否使用GPU训练（前提需要环境支持cuda，否则还是cpu） | boolean | False |
| isSelf | 是否是无标签学习（例如自动编码器） | boolean | False |
| isMask | 在训练过程中是否需要使用MASK屏蔽缺失数据 | boolean | False |
| showLoss | 训练过程中是否显示loss | boolean | False |

此外，DeepLin支持多种损失函数与优化函数，写法如下：
* lossFunc：损失函数
    * 'CEP'：CrossEntropyLoss
    * 'MSE'：MSELoss
    * 直接支持'torch.nn.modules.loss'下的损失函数，可直接传入
* optFunc：优化函数
    * 'RAdam': RAdam（默认）
    * 'SGD': SGD
    * 'Adam': Adam
    * 直接支持'torch.optim'下的优化函数，可直接传入


### 打包训练集
```Python
model.getTrainingDataLoader(x, y=None, isShuffle=True, preShuffle=False)
```
#### 参数说明

| 形参名 | 作用 | 数据类型 | 默认值 |
| --- | --- | --- | --- |
| x | 样本特征数据 | array | 无，必须提供 |
| y | 样本标签数据 | array | 无，可选 |
| isShuffle | 是否打乱数据（利用PyTorch的DataLoader**在训练过程中**实时打乱） | boolean | True |
| preShuffle | 是否预先打乱数据（数据**在打包过程中**提前打乱） | boolean | False |

> Tips: 当在getTrainingDataLoader中未提供标签数据y时，DeepLin会自动切换到无监督学习模式，同时损失函数会自动切换到MSELoss，暂时不支持更改

### 训练网络
```Python
model.train()
```
### 打包测试集
```Python
model.getTestDataLoader(x, y=None, isShuffle=False, preShuffle=False)
```
#### 参数说明

| 形参名 | 作用 | 数据类型 | 默认值 |
| --- | --- | --- | --- |
| x | 样本特征数据 | array | 无，必须提供 |
| y | 样本标签数据 | array | 无，可选 |
| isShuffle | 是否打乱数据（利用PyTorch的DataLoader**在训练过程中**实时打乱） | boolean | False |
| preShuffle | 是否预先打乱数据（数据**在打包过程中**提前打乱） | boolean | False |

> Tips: 当在getTestDataLoader中要求打乱测试数据时，DeepLin会自动取消打乱操作，并弹出warning，暂时不支持更改

### 训练网络
```Python
test_pred = model.test()
```

**仅供实验, 切勿实用**

For experiment only, **DO NOT APPLY IT INTO ANY REAL PROJECT**.

<p align="right">Coded By iRinYe.CN</p>
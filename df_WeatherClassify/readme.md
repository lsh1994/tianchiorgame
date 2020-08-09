# “观云识天”-机器图像算法赛道-天气识别

[博客说明](https://blog.csdn.net/nima1994/article/details/102702524)

## 实验概述

训练所用方法：  
- 数据增强(放缩、随机大小裁剪、翻转)，Auto_augment，随机擦除（RandomErasing）
- 学习率：RAdam，Lookahead
- 半精度训练（APEX）
- Random Image Cropping And Patching（RICAP）
- 模型：EfficientB3、Densenet121、xception等

测试：
- TTA

其他：
- 数据分析（EDA，不平衡过采样）
- 训练日志记录（tensorboardX）
- ReduceLROnPlateau
- 集成方案：按类别和按输出概率

## 结果

线上单模型F1 0.845-0.860左右，集成0.865左右。  
线下训练集：验证集=9：1，验证集0.85左右。


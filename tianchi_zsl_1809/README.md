# tianchi zsl 2018

2018之江杯全球人工智能大赛-零样本图像目标识别

## basemodel
[blog网址：天池零样本目标识别新手笔记](https://blog.csdn.net/nima1994/article/details/82420637)
线上精度：0.0716


## basemodel2
线上精度：0.0905
参考了论坛中的多分类联合训练的思路，并做了更改：

- 没有构造如 'non_1'的特征，直接使用如原来的全0特征表示，并对每类特征除和
- 使用的vgg16

![](https://img-blog.csdn.net/2018091310530896?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbWExOTk0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

[比赛地址](https://www.kaggle.com/c/plant-seedlings-classification)

基于Discussion的一种解决方案，使用Xception微调（搬运工）。

训练集样本：
![在这里插入图片描述](https://img-blog.csdn.net/20180927104035745?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbWExOTk0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

在线精度：0.96347（375/836）。(评价标准：f1-score)
![在这里插入图片描述](https://img-blog.csdn.net/20180927104405397?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbWExOTk0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

训练曲线（橙色为全连接层微调、绿色为全部问题）：
![在这里插入图片描述](https://img-blog.csdn.net/20180927105046597?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbWExOTk0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/2018092710513794?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbWExOTk0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


# Speech Recognition (6/7) - RNN-T Training (optional)

接下来讲解：如何将所有的alignment求和



看这个RNN-T的计算图

<img src="C:\Users\26421\AppData\Roaming\Typora\typora-user-images\image-20210905200916721.png" alt="image-20210905200916630" style="zoom:50%;" />

每一个格子(i,j)对应一个概率P(i,j)  ,而且P与路径无关

>  既然每个格子的概率与路径无关，那概率又到底受什么影响呢？  答案是： 前面的𝜙的数目  +   前面Token ；
>
> 由于不同路径仅仅代表𝜙的排列不一样，但是𝜙的数目是一样的。而且前面的Token顺序也是固定的，所以不同的排列方法其实对P是没有影响的。

然后就是一个条件概率的求解，有点像动态规划。 

<img src="https://gitee.com/pinboy/typora-image/raw/master/img/202109052020622.png" alt="image-20210905202058559" style="zoom:50%;" />

通过∑P_~θ~_ h|X)  的求和（h就是每个Y对应的alignment），就可以求出P~θ~ (Y|X)了

我们的目标函数，是下面这个式子，也就是选择参数θ，尽可能使得P（Y|X）小。用梯度下降法。
$$
\frac{\partial P(\hat{Y} \mid X)}{\partial \theta}
$$
求偏导的时候，由链式法则就可以一样地做反向传播。具体的算法见视频。https://www.youtube.com/watch?v=L519dCHUCog

<img src="https://gitee.com/pinboy/typora-image/raw/master/img/202109052035991.png" alt="image-20210905203514916" style="zoom:50%;" />

计算起来比较复杂，求和起来比较难，一般用最大值代替。(？？)



总结：

RNN-T 和LAS会参考上下文的输出。

CTC和RNN-T 都需要alignment，而且训练起来比较麻烦，但是能够做到on-line输出。

![image-20210905204748282](https://gitee.com/pinboy/typora-image/raw/master/img/202109052047369.png)
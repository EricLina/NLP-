#  Speech Recognition (4/7) - HMM (optional)

从HMM角度来看

用统计模型来求解，穷举所有可能的Y（Token Sequence），概率最大的Y就是预测结果。以上过程成为Decode（解码）

## HMM的基本原理简要介绍

HMM的state： 是一个比Phoneme更小的单位（把发音单位进一步切分）

(Phoneme>tri-phoneme>state)

每一个State 有两个属性几率

1. Transition Probability ： 状态跳转概率（比如状态A下，跳到状态B的概率）
2. Emission Probability ：状态产生Feature的几率（比如状态A下，能够发出音节"u"的概率，所有的音节概率分布用一个向量表示）

问题： state过多，可以不同state指向同一个modern来解决，这就是Subspace GMM



一段acoustic sequence可以由不同的state sequence产生，如图![image-20210905163134389](https://gitee.com/pinboy/typora-image/raw/master/img/202109051631452.png)

而且不同的state sequence对应的概率不同，所以光知道上面的Transition Probability，Emission Probability，但还是无法根据一段state sequence 算出 acoustic sequence概率。

我们通过穷举的方法，计算出所有的可能然后相加即可得到State sequence条件下产生acoustic sequence的概率![](https://gitee.com/pinboy/typora-image/raw/master/img/202109051637774.png)

> alignment： 使得state sequence --> acoustic sequence 的一个Function。 要求把所有的state，与输出的大小匹配，并按照顺序排列（可以重复出现，比如aabccd）。![](https://gitee.com/pinboy/typora-image/raw/master/img/202109051640241.png)

## 以HMM为基础的深度学习模型

1. tandem?

用一个DNN训练出state 的 分类器，帮助我们更好地获得的acoustic feature ,其他的不变。

2.  DNN-HMM hybrid?

用一个DNN直接训练x-->P(a|x) 的网络，用贝叶斯概率公式转化后可以直接求得P(x|a)。



## 效果爆炸的State classifier 的训练方法

没有之间的训练数据，你可以知道state-sequence , 但是你不能直接得出对应的 acoustic  feature . 

所以处理方法是用一个HMM-GMM的模型来做一个alignment，也就是得到一个state -> acoustic feature 的分配方法使得几率最大。有了这个alignment以后（state -> acoustic feature的数据），你就可以去训练Classifier了（DNN1）。

也许我们第一次的alignment 不是很准确，于是我们利用刚刚得到的DNN1，再做一次alignment，重新再训练出另一个Classifer（DNN2）。

反复做，知道满意为止。![myfirst3](https://gitee.com/pinboy/typora-image/raw/master/img/202109051856233.gif)

以上的处理方法效果很好！让微软的语言识别达到人类的准确率！


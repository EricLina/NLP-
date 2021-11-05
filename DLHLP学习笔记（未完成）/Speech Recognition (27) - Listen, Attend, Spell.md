# Speech Recognition (2/7) - Listen, Attend, Spell



## LAS

### 2.1 神经网络的输入输出抽象

将语音信号  经过网络 ，得到更抽象的输出表示

这里的网络，可以是RNN，CNN等

​	RNN  可以双向传播；

​	CNN 每一个Filter 都是考虑到了周围的信息，可以叠加多层的Filter，这样就能收集到全部的信息，

![](https://gitee.com/pinboy/typora-image/raw/master/img/202109041432998.png)

![](https://gitee.com/pinboy/typora-image/raw/master/img/202109041433094.png)

目前的使用一般是RNN+CNN



### 2.2 对输入对Down-Sampling

第一层输入100个向量，第二层输出50个向量； 这样就可以降低运算量；

具体的来说，比如CNN 中的Time-Delay CNN  ：只选取首尾向量

 和 Truncated Self-Attention ：每一个Attention 控制范围，缩小向量长度

![image-20210904144201487](https://gitee.com/pinboy/typora-image/raw/master/img/202109041442564.png)



### 2.3 Attention

符号说明：

h~i~表示 语音信息编码， Z 表示 RNN的隐藏状态

1. Dot Product Attention

   ![image-20210904152820771](https://gitee.com/pinboy/typora-image/raw/master/img/202109041528829.png)

2. Additive Attention

   ![image-20210904152913527](https://gitee.com/pinboy/typora-image/raw/master/img/202109041529590.png)



### 2.4 Spell

![image-20210904155726089](https://gitee.com/pinboy/typora-image/raw/master/img/202109041557159.png)

 	每个h~i~ 和Z~i~经过Attention后，会得到一个α~i~，  接下来通过Softmax得到解析式C(h~i~)				![  ](https://gitee.com/pinboy/typora-image/raw/master/img/202109041544078.png)

 C(h~i~) 就当做Decoder的输入。 Decoder的输出就是Token的几率向量， 选出几率最大的作为最终输出（这里输出 字符C）

![image-20210904155638685](https://gitee.com/pinboy/typora-image/raw/master/img/202109041556733.png)



### 现在我们总结一下

需要输入 ： 

-  Z^0^    +   h~i~  -->  α~i~  -->   C^0^ 
-  C^0^  + Z^0^  ----(RNN)--->  预测向量  + Z^1^

到了下一步 

- Z^1^    +   h~i~  -->  α~i~  -->   C^1^ 
-  C^1^  + Z^1^  + 上一步的预测向量----(RNN)--->  预测向量  + Z^2^

也就是说 上一步的Z~j~还会用来做下一步的Attention函数的输入，一起得到α~j+1~

![image-20210904160523666](https://gitee.com/pinboy/typora-image/raw/master/img/202109041605745.png)



![image-20210904161125323](https://gitee.com/pinboy/typora-image/raw/master/img/202109041611363.png)





### 2.5 beam search 

二分搜索，求几率最大的路径

### 2.6 训练

目标函数：输出的预测向量，与下一个文字的独热表示的Cross Entropy 最小。

但是，训练的时候其实已经“看到了全文” ： Teaching Forcing



### 2.7 location Aware attention 

每一次计算α，都会参考附近位置的α值



LAS 在大样本训练的效果很好

但是LAS 需要全部输入后才开始输出第一个Token，无法“边听边写”


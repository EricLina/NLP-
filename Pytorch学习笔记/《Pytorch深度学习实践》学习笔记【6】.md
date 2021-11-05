# 《PyTorch深度学习实践》学习笔记 【6】RNN—1

学习资源：
[《PyTorch深度学习实践》完结合集](https://www.bilibili.com/video/BV1Y7411d7Ys?p=2)

## 十二、RNN（循环神经网络） Basic

回顾

DNN（Deep Neural Networks）:本课程的开始讲的网络

![image-20210820150711445](https://gitee.com/pinboy/typora-image/raw/master/img/202108201507543.png)



举例子：

根据三天的温度，气压 来预测第四天是否会下雨

计算的主要部分在全连接层

​	卷积层计算次数= In_channels * Out_channels * KernelSize

​	全连接层计算次数= 

> https://zhuanlan.zhihu.com/p/77471991
>
> 1. 卷积层的运算量 conv flops = Channel_out * Channel_in * k * k    (其中k为卷积核的宽度和高度)
>
>    ​		与输入输出的通道数，卷积核的大小有关
>
> 
>
> - 全连接层 就相当于卷积核的尺寸就是输入矩阵的尺寸， 输出矩阵为1*1尺寸  （矩阵相乘）
>
> 2. 全连接层的计算量 fullyconnected flops = BatchSize * Cout * Cin
>
> ![image-20210820162830618](https://gitee.com/pinboy/typora-image/raw/master/img/202108201628691.png)
>
> ​	与变化后的数据大小有关

全连接层的计算量权重占比比较多







RNN专门用来处理 带有序列模式的的问题

将x1 x2 x3 视为序列， 不仅考虑连接关系，还要考虑序列关系	（今天的天气和昨天前天的天气有关）【文本，股市等】





![image-20210820163446243](https://gitee.com/pinboy/typora-image/raw/master/img/202108201634312.png)

for(x in x):

​	h=liner(x,h)





![image-20210820163915878](https://gitee.com/pinboy/typora-image/raw/master/img/202108201639949.png)

循环神经网络

​	RNN cell 中 融合了前层的信息

​	激活函数常用tanh





定义一个RNN cell：

​		只需要知道input_size hidden_size

```python
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
```

![image-20210820164226221](https://gitee.com/pinboy/typora-image/raw/master/img/202108201642301.png)

使用一个RNN cell：

```python
hidden = cell(input, hidden)
```

input和hidden的维度需要满足：



  

​	![image-20210820164456262](https://gitee.com/pinboy/typora-image/raw/master/img/202108201644318.png)







数据集的维度比以前多了一个 seqLen（序列的长度）







RNN的调用

```
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
```

RNN的输出结果

```
out,hidden = cell(inputs,hidden)
```

各个参数对应的部分

![image-20210820165349227](https://gitee.com/pinboy/typora-image/raw/master/img/202108201653282.png)

输入参数的维度要求

- 输入：

  input : (seqSize, batch ,input_size)

  hidden: (numLayers, batch , hidden_size)

- 输出

  out : (seqLen, batch, hidden_size)

  hidden : (numLayers ,batch ,hidden_size)

![image-20210820165605987](https://gitee.com/pinboy/typora-image/raw/master/img/202108201656051.png)

注意到，

input 和output只有最后一个参数不同

输入部分和输出部分的hidden 的尺寸是一模一样的



参数小结：

​	![image-20210820170013385](https://gitee.com/pinboy/typora-image/raw/master/img/202108201700457.png)



NumLayers的理解：

一个RNN里面有NumLayers层，每一层的cell的输出，还会送到下一RNN层（图中向上）作为输入，最终经过NumLayer层后才得到h_i 。    

h~i~ 与下一个输入 x~i+1~  融和输入下一层节点（图中向右）

![image-20210820170817074](https://gitee.com/pinboy/typora-image/raw/master/img/202108201708156.png)

 



代码理解：





独热表示：

一个矩阵，每一行只有一个1，1的列数代表索引的值

![image-20210821224513194](https://gitee.com/pinboy/typora-image/raw/master/img/202108212245271.png)

## Connectionist Temporal Classification (CTC)

能够on-line地输出（一边输入一边输出结果），但是没有考虑上下文，会导致输出的“结巴”

![image-20210904185634781](https://gitee.com/pinboy/typora-image/raw/master/img/202109041856848.png)



每一个Classifier  都只是输入一个 向量 ， 马上输出一个向量；所以每一个输出都是独立的，这样带来的问题就是输出的“结巴”。

>  这里介绍一下空白 𝜙 ，𝜙 就相当于每一个Token之间的分隔符号；
>
> 在模型的内部产生，具体来说，是在softmax函数的输出向量中。
>
> 在真实输出的时候，会将𝜙去掉并合并𝜙与𝜙之间重复字符。
>
> 但是有个问题： 
>
> 𝜙 的位置有很多种插入方式，造成识别难度增加。（解决的方式是在训练的时候直接穷举）
>
> ![image-20210904191725655](https://gitee.com/pinboy/typora-image/raw/master/img/202109041917724.png)![image-20210904193310937](https://gitee.com/pinboy/typora-image/raw/master/img/202109041933014.png)

所谓“结巴”就是因为 ，就算上一个token已经有𝜙输出了，按理来说下一个输出应该减少与前面重复的几率，但是由于Classifier判别的独立性，不会有这样的考虑，带来的后果就是输出的“结巴”



## RNN Transducer (RNN-T)

- CTC--> RNA 的改进

  ​		RNA 将前面的输出也考虑了进来（相当于改成了LSTM）

  ​		但是两者都是 “吃一个输入，只输出一个token”，如何解决输出多个Token呢？ 下面对RNA 进行改进

  ![myfirst2](https://gitee.com/pinboy/typora-image/raw/master/img/202109041937647.gif)

- RNA-->RNN-T

  吃一个输入输出一个token--> 吃一个输入输出多个Token

  实现方式： 将多个Token视为一个单位，以𝜙  为标识隔开。（所以有多少个Token，就有多少个𝜙 )

  ​				𝜙 ：“可以输入下一个 Acoustic Feature （输入向量） 

  和CTC一样，在空白地方，必须插入𝜙 

  

  ![image-20210904195247134](https://gitee.com/pinboy/typora-image/raw/master/img/202109041952216.png)


  用一个额外的RNN，并过滤掉𝜙 

  1. 这个RNN 只看Token作为输入，所以你可以用大量的文字，先训练这个RNN。
  2. RNN训练的时候，你要穷举所有的𝜙 组合，这就需要一个不把𝜙 当做输入的RNN。

## Neural Transducer

CTC，RNN都是一次读一个Acoustic Feature ，能不能一次只能读一个Acoustic Feature ，而是读多个呢？Neural Transducer可以！（其实就是把Window里面的信息量增加）

由于读入了很多Acoustic Feature ，所以需要在Acoustic Feature 中也使用一个attention。

![image-20210904195626248](https://gitee.com/pinboy/typora-image/raw/master/img/202109041956316.png)

- 为什么要用attention呢？

  根据一些paper的实验结果，使用attention可以降低错误率。

## Monotonic Chunkwise Attention (MoChA)

能不能自由地决定，要把Window移动多少呢？

用一个z来决定，一个Window的终止位置。

![myfirst3](https://gitee.com/pinboy/typora-image/raw/master/img/202109042002723.gif)





总结一下：

> LAS: 就是 seq2seq
>
> CTC: decoder 是 linear classifier 的 seq2seq 
>
> RNA: 輸入一個東西就要 輸出一個東西的 seq2seq
>
> RNN-T: 輸入一個東西可以 輸出多個東西的 seq2seq 
>
> Neural Transducer: 每次輸入 一個 window 的 RNN-T 
>
> MoCha: window 移動伸縮 自如的 Neural Transducer


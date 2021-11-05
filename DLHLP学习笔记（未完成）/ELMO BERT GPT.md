# ELMO BERT GPT

## 1. 回顾

### 1 of N encoding

- ​	独热编码

![image-20210923185612947](https://gitee.com/pinboy/typora-image/raw/master/img/202109231856104.png)

### Word class

- ​	考虑同义词

![image-20210923185544885](https://gitee.com/pinboy/typora-image/raw/master/img/202109231855132.png)



Word Embedding

- ​	每一个词汇都用一个向量表示它，语义相近的向量距离就比较接近

![image-20210923185605849](https://gitee.com/pinboy/typora-image/raw/master/img/202109231856947.png)

​			训练原理：根据上下文训练出来的

​			我们以后就会用word embedding作为输入表示一个词汇了。



## 2. 

### 一词多义

​	同一个单词，比如bank，不同上下文会有不同的意思。如果用传统embedding，则不会区分。过去处理方法是：查词典，有几个语义解释，一个bank就有几个embedding方法。但是这很死板，因为语言总是在发展的，不同字典也没有统一。

​	**contextualized word embedding**: 每一个单词的token都会给他一个单独的embedding方法。

### 	**ELMO**

![image-20210923190947835](https://gitee.com/pinboy/typora-image/raw/master/img/202109231909202.png)

RNN学习，用上下文训练。我们将输出的embedding（这里的embedding已经获取到了上文信息） 作为每一个单词的word embedding

##### 有人说无法获取到下文内容，肿么办呢

我们用RNN反向训练一遍，做同样处理，然后和正向RNN输出结果拼接到一起即可。



##### RNN每一层输入处理完后都会吐出一个embedding，所以如何利用这些信息呢？

​	ELMO的处理方法是用加权和的方法，把它们加起来

​	这里的weight α1 α2 都是作为学习内容自己学出来



### **Bert**

##### bert是什么？

给一个句子进去，每一个句子都吐一个embedding给你

内部架构和transformer的encoder一样（self-attention机制）

中文训练bert的时候，用字当成单位是更为恰当的

##### 如何训练bert呢?

1. 模仿完形填空，把某个词遮住，要bert给它填回去。

![image-20210923193603555](https://gitee.com/pinboy/typora-image/raw/master/img/202109231936838.png)

2. 给两个句子，让bert判断能不能接在一起。（给出下一句，让bert告诉你是不是讲得通）

![image-20210923193754750](https://gitee.com/pinboy/typora-image/raw/master/img/202109231937988.png)

bert内部的self-attention机制，让两个词在句子中的位置距离被淡化

- ​	通常方法1，方法2要同时使用效果才好









 



​	


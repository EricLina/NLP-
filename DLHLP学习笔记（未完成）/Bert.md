# Bert



阅读大量的信息，得到知识后，稍微训练后就可以得到一个需要的模型



## 1. 什么是预训练

### 1.1 背景

- 过去，用一个编码向量代表每一个Token，这样的问题是相同的Token会对应相同的编码，而没有考虑上下文语境。这样的技术有：Word2Vec,Glov

- 由于英文的token无法穷举，（因为英文单词可以不断地造新的出来），所以可以将英文单词的每一个字母读入后再输出一个编码向量。（比如FastTest）![image-20211006101123280](C:\Users\26421\AppData\Roaming\Typora\typora-user-images\image-20211006101123280.png)
- 中文的一种思路，可以将偏旁和部首作为输入进行处理



以上方法均没有考虑上下文，比如“他脱单了，我们都是狗”（单身狗的意思），“那有一只狗在叫”（就是狗），两种“狗”应该有不一样的embedding。由此出现了contextualized Word embedding，也就是看一整个句子再输出embedding的值。内部结构一般为LSTM，Self-attention，Tree-base Layer

- Tree-base model 适用于文法规则比较强的任务，比如理解数学表达式



现在的Model越来越大，比如Bert，GPT。也有一些效果比较好的小模型。

为了处理很长的篇幅的文字（段落，甚至文章），出现了Transformer-XL，reFormer，longerFomer等专用模型



### 1.2 怎么将预训练模型改造以适应特定任务呢？

首先来明确NLP中的输入和输出：

> 1. 输入：
>    1. 一个句子
>    2. 多个句子 
> 2. 输出：
>    1. 一个类
>    2. 每个Token对应一个类
>    3. 从输入中复制
>    4. 一般文本

对于输入，一个句子直接丢进去就可以了，多个句子只要在句子之间加上分隔符[SEP]，变成一个很长的带[SEP]的句子一块丢进去就可以了

1. 对于输出“one class”，处理方法是在输入加入一个[CLS],当模型看到一个[CLS]，就会对应输出一个Class。而class的计算方法是，将Model输出的多个embedding输入到另一个分类器中，可能就是一个简单的线性层进行分类，得到输出Class。![image-20211006103023340](https://gitee.com/pinboy/typora-image/raw/master/img/202110061049920.png)
2. 对于输出是“class for each token” 如图![image-20211006103057226](https://gitee.com/pinboy/typora-image/raw/master/img/202110061030274.png)
3. 对于“copy from input” 。
   1. 先明确什么是“copy from input”？
      例如在问答系统中，需要输出的结果是文档中的某几个单词，那么回答实质上就是这几个单词的位置（图中的77,79）![image-20211006104339399](https://gitee.com/pinboy/typora-image/raw/master/img/202110061049157.png)
   2. 拿上面的QA系统举例，输入的是Question和Document，那么得到输出的embedding后，根据Task Specific的方法，对每一个document对应的embedding进行检测（图中用了个dot-product+softmax），最终得到分类结果。![image-20211006104618312](https://gitee.com/pinboy/typora-image/raw/master/img/202110061046374.png)
4. 对于“general Sequence” ，直觉上的做法，是将Pretrained-Model输出的Embedding直接做个Attention然后丢到一个Seq2Seq中去进行Decoder。但是这样的坏处是，后面的Seq2Seq没有训练完全。也许这不是最好的设计。![image-20211006104141594](https://gitee.com/pinboy/typora-image/raw/master/img/202110061041653.png)
   改进的方式是：Model的每一个输出的embedding都丢到一个Seq2Seq中输出一个sequence W~i~，同时将W~i~ 作为Model第i+1步的输入，这样就考虑到了前文的信息。最终把W都拼接起来成为最后的Sequence。![image-20211006103439390](https://gitee.com/pinboy/typora-image/raw/master/img/202110061049783.png)





### 1.3 weight feature

- 以往都是将最后一层layer的输出作为最后的Embedding。其实还有一种做法，就是将中间层的输出也考虑进来，然后加权作为最后的embedding输出。这种方法就叫 Weight feature。

- 对于不同任务，不同层的layer的权重会有所变化，但是每一个layer的权重w~i~都可以被训练出来。

- 这种思想和ELMO很像

![image-20211006110445250](https://gitee.com/pinboy/typora-image/raw/master/img/202110061104310.png)



## 2. 怎么微调



### 2.1 两种微调方法：

1. 微调Task-specific和PretrainedModel （整个微调）

2. 只微调Taks-specific部分，PretrainedModel固定住

   ![image-20211006105112842](https://gitee.com/pinboy/typora-image/raw/master/img/202110061051913.png)

   ​	由于预训练模型很大，微调整个模型很容易造成过拟合，但是一般都是将预训练模型单独训练完，这样就不容易过拟合了。

   ### Adaptor

   ​		每一个Model经过微调后，参数都会改变。但是model的参数是很大的（上亿个），所以调参起来很困难，我们如何针对性地去调整部分参数呢？这就是Adaptor做的工作。

   ![](https://gitee.com/pinboy/typora-image/raw/master/img/202110061055621.png)

   ​	可以从图里面看出，过去没有adaptor，一个新的微调模型都要重新保存所有参数（上图）。而有了Adaptor之后，每保存一个新的微调模型，其实保存的都是一个原有的模型加上一个Adaptor（下图）。

   ![image-20211006105752091](https://gitee.com/pinboy/typora-image/raw/master/img/202110061057157.png)

   Adaptor该插在哪里，该修改哪些参数，其实还待研究。





### 2.2 为什么预训练？为什么要微调？

1. 在训练的时候，可以加快收敛速度，减少训练的时间。![image-20211006110859673](https://gitee.com/pinboy/typora-image/raw/master/img/202110061111214.png)

2. 模型的泛化能力更强![image-20211006111121148](https://gitee.com/pinboy/typora-image/raw/master/img/202110061111407.png)

## 3. 怎么预训练



### 3.1 监督学习 自监督学习

​	对于翻译任务而言，过去的训练方式，是利用大量的pair data（语言A--语言B的对应）来训练，这种pair data较难获取。于是我们想到能不能用无标注文字直接去训练呢？（自监督学习：用部分输入，去预测另一些输入）

- ​		监督学习&自监督学习：	

  ​				![image-20211006111804961](https://gitee.com/pinboy/typora-image/raw/master/img/202110061118000.png)



### 3.2 在自监督学习中，如何预测下一个Token呢？ 

 输入句子W~t~和embedding h~t~，经过layer+softmax后再与每一个Token做一个Crossentropy。Loss函数即为与真实文本下一个单词的差异。 取Loss最小的Token作为输出。

把上一个预测的输出Token W~t~作为下一次预测的输入。<img src="https://gitee.com/pinboy/typora-image/raw/master/img/202110061128500.png" alt="image-20211006112139189" style="zoom: 67%;" />

需要注意的是，在预测下一个Token的时候，一定不能让Model看到下一个Token是什么，否则Model就会直接使用下一个Token，这样训练就是无效的。![image-20211006112828288](https://gitee.com/pinboy/typora-image/raw/master/img/202110061128335.png)

### 3.3 在PretrainedModel内部常用的预测方法

​	LSTM，Self-attention等		![image-20211006113006997](https://gitee.com/pinboy/typora-image/raw/master/img/202110061130053.png)

注意使用Self-attention的时候（self-attention机制可以看到每一个词汇，从而忽略文本距离的影响），要注意attention的范围，不能读到下一个Token。

![image-20211006113311141](https://gitee.com/pinboy/typora-image/raw/master/img/202110061133198.png)



# Bert家族 ELMO等

## ELMO

### 是怎么获取语义的呢？

语言学上有研究表明，要理解上下文！也就是说，要知道与这个单词常常一起出现的其他词汇。

**那就分析一下ELMO，它是怎么获取上下文的。**

ELMO中有双向的LSTM，左到右LSTM获取上文信息，右到左LSTM获取下文信息，整合起来一起表示W~4~ 这个Token的语义。

![image-20211006113805141](https://gitee.com/pinboy/typora-image/raw/master/img/202110061138192.png)

**但是上面说过，在训练的时候，是无法看到下一个Token的，也就是说无法获取下文的信息**。所以ELMO其实上下文是单独考虑，然后强行整合的。**所以ELMO实际上无法同时地看到上下文信息*。Bert恰恰解决了这个问题！

## Bert

**Bert的处理方法是：对于特定Token，用MASK或者随机用其他Token来代替，将这个Token“盖住”。** ![image-20211006114726680](https://gitee.com/pinboy/typora-image/raw/master/img/202110061147745.png)

这样就解决了Model能“看到”下一个需要预测的Token的问题，而且可以直接使用全局Self-attention而不用加以限制。



### Bert与CBOW 的相同与不同

相同：都是通过上下文，来预测中间某个Token

不同：

- CBOW的上下文是固定长度的，而Bert使用的SelfAttention使得其能看到的上下文是全部的。
- Bert的模型更加复杂（12层），而CBOW就简单很多（2层）

![image-20211006114856751](https://gitee.com/pinboy/typora-image/raw/master/img/202110061148815.png)



### Mask Input

随机的Mask方法效果可能不太好。

下面有几种mark的方法：

- Whole Word Marking （WWM）
  单位是一个Word

![image-20211006120818877](https://gitee.com/pinboy/typora-image/raw/master/img/202110061208014.png)

- SpanBert

  1. Mask的单位是一个Span，span的长度可以变化，可以是多个Token。
     1. 提出了SBO的方法：根据Span前后Token的Embedding来预测Span中的信息。![image-20211006123139876](https://gitee.com/pinboy/typora-image/raw/master/img/202110061231934.png)
     2. 原理：每一个Span的信息其实都可以在前后的Embedding中获取。
     3. 在coreference中应用效果比较好 

  

## 	XLNet

​	内部用到的预测模型是Transformer-XL 	

​	为了解决“不要让model看到下一个需要预测Token”的问题，XLNet**将Token的顺序打乱**。

![image-20211006123851255](https://gitee.com/pinboy/typora-image/raw/master/img/202110061238312.png)

​	相比于Bert 在self-attention时能看所有的Token（包括Mask本身），XLNet在做self-attention的时候会只看部分的Token，而且不让model看到Mask本身。	

​      	<img src="https://gitee.com/pinboy/typora-image/raw/master/img/202110061241429.png" alt="image-20211006124113365" style="zoom:80%;" />     





## Mass 和Bart

### Bert的限制

一般LanguageModel-Style是自回归的（auto-regression），对应训练方法是，看前面的文本，来预测下一个Token。但是由于Bert在训练的时候，需要看到左右两边所有的信息，所以在自回归的生成任务上就有了限制。所以Bert其实是不适合来做Seq2Seq的任务的。

但是最近出现了non-auto-regression的方法，可能Bert-style会更加合适。



​	其实预训练模型就是一个典型的Seq2Seq模型，在auto-regression框架下，如果想要Bert完成类似Seq2Seq的任务，Bert如何改进呢？



![image-20211006151039798](https://gitee.com/pinboy/typora-image/raw/master/img/202110061510856.png)

为了让Decoder得到很好的训练，需要对输入做“某种程度的破坏”。MASS，BART中有不同的“破坏”方法。



## 	

​	Mass： 把一些部分随机Mask起来。（只需要模型预测mask部分就可以）

​	Bart：

1. 普通的Mask
2. delete（直接删掉，不说这里有mask）
3. Permutation  ：循环乱序
4. Rotation：完全乱序
5. Text Infilling ： 盖住一整块（块大小随机）

![image-20211006154330110](https://gitee.com/pinboy/typora-image/raw/master/img/202110061543167.png)

​			结果发现，乱序的效果不佳。最好的方法是text Infilling 



## UniLM

它兼具encoder ，decoder，Seq2Seq 功能，可以按照不同的方式训练，获得不同的功能。

![image-20211006154946005](https://gitee.com/pinboy/typora-image/raw/master/img/202110061549081.png)



## ELECTRA

“预测”一个东西，需要的训练量是很大的。能不能避免“预测” Token呢？  ELECTRA方法就可以避免“预测”或者说“生成”

具体的做法：将某一个Token 替换成其他的Token 	，每一个Token都对应一个Yes/No （是否被置换）。

预测一个Yes/No 比预测一个详细的Token简单地多。

![image-20211006155553584](https://gitee.com/pinboy/typora-image/raw/master/img/202110061555668.png)

如果置换成一个固定的Token，其实很容易就能被Model发现。

一个解决办法是：用一个small bert 去产生被Masked 的东西，然后用small bert 的预测（ate）丢给Model去判断是不是被替换的。这种方法有点像 GAN。

用Small Bert 是因为，如果Bert训练得很好，那么预测出来的值就会和原文相差不大，就没有“替换”的效果了。用Small bert可以产生一些错误的预测值，作为原文的Token 的替换。

![image-20211006155928445](https://gitee.com/pinboy/typora-image/raw/master/img/202110061559500.png)

这里的Small Bert和GAN中的Generator作用相似。但是也有所不同，Small Bert是单独训练的，而在GAN中的Generator是要整体一起训练的。(ELECTRA这样做的原因可能是Bert的参数本身比较大，而GAN的Generator本来就很难训练。)







# Sentence Level

 有时候我们需要用到句子级别的Embedding。所以embedding代表的不再是一个Token，而是一个Sentence。

一个词的意思，与其前后词有关。那么一个Sentence的语义，也许也和上下句有关。

embedding的距离代表语义相似度，这一点和token的embedding是一致的。



原始的Bert在句子级别的方法：

**NSP（Next Sequence Prediction）**： 

- 在不同的句子之间，用一个[SEP]分割开，然后让model去判断[SEP]前后的句子连在一起是否合理（yes/no)。
- 但是NSP效果不是很好。

**SOP（sentence order prediction）**：

- 正常语句连接，预测yes；句子倒序连接，预测no。

- 用于Albert

从语义的辨识角度来看，SOP的难度可能要更大，对模型的训练更有帮助。

​			structBert也用到了SOP的想法



- 预训练所需要的资源是海量的

- Bert的用处，不只是文字，还有更多的无标注数据基础的任务，比如Audio Bert


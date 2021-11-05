# Speech Recognition (1/7) - Overview

## 一、Overview

### 1.1 语音识别的输入和输出

输入：一段speech

输出： 一段text



#### 1.1.1 输出

具体来说，输出可以看为一段一段的Token 

> ##  Token
>
>  比如Phoneme，Grapheme ，word，Morpheme
>
> - Phoneme: a unit of sound    （发音的基本单位）
>
> ​					与声音有一一对应的关系，需要专门的标识。
>
> - Grapheme: smallest unit of a writing system   （一些字符，比如26个字母和标点，中文文字）
>
> - word： 对于英文，就是空格分隔的词；但是对于中文，不能根据空格来分词，比较困难。
>
> - Morpheme：词根（ 比word还小，比grapheme大）  
>
>   ​	 		比如英文的 _ er，Auto_, Re_
>
> 所有不同的Token组合，都可以用Byte序列来描述  ；比如UTF-8 就可以来表示一个个的字符，组合起来就是一个词，一个句子，一篇文章。
>
> - 语音辨识的输出类型饼状图
>
> ![image-20210904134439881](https://gitee.com/pinboy/typora-image/raw/master/img/202109041344009.png)



#### 1.1.2  输入

- ###### Window 和Frame 

一个最小的音频处理单位被称为一个Window（ 长度25ms），用一个向量来描述这个Window下的音频信息，这个向量就是Frame（Acoustic Features）

接下来移动这个Window，继续处理剩下的音频。

![image-20210904135950522](https://gitee.com/pinboy/typora-image/raw/master/img/202109041359585.png)









- MFCC

这是一种语音特征提取的传统方法，详细过程如图。  但是随着深度学习的出现，MFCC渐渐被取代。

![image-20210904140902632](https://gitee.com/pinboy/typora-image/raw/master/img/202109041409706.png)

- 训练需要的数据量 

![image-20210904141112915](https://gitee.com/pinboy/typora-image/raw/master/img/202109041411967.png)

- Seq2Seq  和HMM

本课程的两个切入点，

未来还会介绍下面五种模型

> • Listen, Attend, and Spell (LAS)   【就是我们用的Seq2Seq】
>
> • Connectionist Temporal Classification (CTC) 
>
> • RNN Transducer (RNN-T) 
>
> • Neural Transducer 
>
> • Monotonic Chunkwise Attention (MoChA)

![image-20210904141319095](https://gitee.com/pinboy/typora-image/raw/master/img/202109041413157.png)


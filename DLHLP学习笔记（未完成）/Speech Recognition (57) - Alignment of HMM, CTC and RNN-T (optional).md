# Speech Recognition (5/7) - Alignment of HMM, CTC and RNN-T (optional)

这一节从End2End 的角度出发



回顾一下，模型的大致的处理流程

原始输的最小单位化(state sequence)--->提取特征Acoustic feature--->输出结果(Token)



## 回顾LAS

回顾一下LAS， LAS的每一步预测结果都与上一步有关，所以很容易得到P("ab")=P("a"|"b")*P("b")![image-20210905191748066](https://gitee.com/pinboy/typora-image/raw/master/img/202109051917126.png)

总之，由于LAS会考虑上下文的关系，所以某个acoustic feature 的几率就可以直接用全概率公式求出来。

![image-20210905191646684](https://gitee.com/pinboy/typora-image/raw/master/img/202109051916723.png)



## CTC和RNN-T

这和上一节的HMM的alignment问题很像，即由于不同alignment之间是独立的，每一个acoustic  feature 可以由多种alignment产生，所以无法通过单个 alignment 直接求得产生某个acoustic feature 的几率。

这里的解决方法同样是：把一个acoustic feature对应的所有可能的alignment穷举并把概率相加。![image-20210905191633799](https://gitee.com/pinboy/typora-image/raw/master/img/202109051916856.png)





## 如何穷举所有的alignment？

HMM，采用重复char的方法； CTC采用重复char+插入𝜙 的方法；RNN-T 采用插入𝜙 的方法

![image-20210905192918318](https://gitee.com/pinboy/typora-image/raw/master/img/202109051929415.png)

我们现在要解决的穷举alignment问题，就是要求出插入𝜙或者说重复char ，有多少种排列方式，各个排列方式对应的概率又是多少。

### HMM

采用重复char的方法

> 顺序性：  
>
> 注意，在重复char的过程中，只能够按顺序重复。想一想也很简单。比如，cat是我最终要输出的。那么我怎么输入才可能被HMM识别出来呢？ 我可以是caaaat，那么HMM就会删去重复的a。但是如果我是caaaaatccct，那么HMM按照规则就会识别出catct，这样就不对了。
>
> 完整性：
>
> cat 对应的每个字母必须出现才行，总不能用一个caaaaa产生一个cat把（连t都没有）



所以如何穷举呢？我们通过一个表格，表格的列就是我们需要计算的Token(这里是"cat")，利用路径来代表alignment。

当然这个路径有要求，因为alignment要求要满足“顺序性”和“完整性”两个要求，所以这条路径要满足1. 要往下走 2. 要走到最下端。

![image-20210905194850054](https://gitee.com/pinboy/typora-image/raw/master/img/202109051948121.png)

接下来就是模拟并计算了。



### CTC

采用重复char+插入𝜙 的方法

有𝜙参与，我们可以这么想，𝜙可以插入任意两个字符之间，比如我在上面产生了cattt，其实可以这么看：_ c_a_t_t_t_，𝜙可以放在任何一个 _ 的位置上，而且可以放0个或者无数个。 

看下面表格，放0个𝜙，就是跳过"𝜙"行，直接到下一个char；放1个𝜙，就是往下走1步；放n个𝜙，就是往右走n步。

![image-20210905195445488](https://gitee.com/pinboy/typora-image/raw/master/img/202109051954547.png)

但是注意，因为我们会碰到一些重复字母的输出，比如teeth，这里两个e之间，是必须要有𝜙来隔开的，否则就会合并掉。所以在表格中，如果有连续的元素，我们不能直接跳过“𝜙”行。



### RNN-T

采用插入𝜙 的方法

由于RNN-T 一次可以输出多个Token，所以这个Token可以是单个的字母比如“a𝜙”，也可以多个字母比如“ca𝜙” 或者“cat𝜙”。当然也可以只有“𝜙”

cat  其实是 _ c_a_t_ ，注意无论Token怎变化，最后t后面一定是有一个𝜙的。

注意RNN-T不会重复char！往右走就是产生𝜙,往下走就是输出Token。

![image-20210905200601981](https://gitee.com/pinboy/typora-image/raw/master/img/202109052006043.png)



总结HMM 和CTC 和 RNN-T

![image-20210905200431841](https://gitee.com/pinboy/typora-image/raw/master/img/202109052004885.png)




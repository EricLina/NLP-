# Week2

## Logistic 回归

Logistic 回归是一个用于二分分类的算法。

Logistic 回归中使用的参数如下：

- 输入的特征向量：$x \in R^{n_x}$，其中 nxnx是特征数量；

- 用于训练的标签：$y \in 0,1$

- 权重：w∈Rnx  w∈Rnx

- 偏置： $ b \in R$

- 输出：$\hat{y} = \sigma(w^Tx+b)$

- Sigmoid 函数：

  $s = \sigma(w^Tx+b) = \sigma(z) = \frac{1}{1+e^{-z}}$

为将 $w^Tx+b$约束在 [0, 1] 间，引入 Sigmoid 函数。从下图可看出，Sigmoid 函数的值域为 [0, 1]。

![image-20210920145733770](https://gitee.com/pinboy/typora-image/raw/master/img/202109201457853.png)

​	例子：

![image-20210920145831785](https://gitee.com/pinboy/typora-image/raw/master/img/202109201458893.png)





### 损失函数：

https://zhuanlan.zhihu.com/p/58883095

用于衡量预测结果与真实值之间的误差。

- 最简单的损失函数定义方式为平方差损失
- 但 Logistic 回归中我们并不倾向于使用这样的损失函数，因为之后讨论的优化问题会变成非凸的，最后会得到很多个局部最优解，梯度下降法可能找不到全局最优值。一般使用
  - $L(\hat{y},y) = -(y\log\hat{y})-(1-y)\log(1-\hat{y})$
- 损失函数是在单个训练样本中定义的，它衡量了在**单个**训练样本上的表现。而**代价函数（cost function，或者称作成本函数）**衡量的是在**全体**训练样本上的表现，即衡量参数 w 和 b 的效果。
  -  $J(w,b) = \frac{1}{m}\sum_{i=1}^mL(\hat{y}^{(i)},y^{(i)})$

## 梯度下降法

函数的**梯度（gradient）**指出了函数的最陡增长方向。即是说，按梯度的方向走，函数增长得就越快。那么按梯度的负方向走，函数值自然就降低得最快了。

模型的训练目标即是寻找合适的 w 与 b 以最小化代价函数值。通过不断计算下面的更新式子，就能够找到最优的w，b

$w := w - \alpha\frac{dJ(w, b)}{dw}$

$b := b - \alpha\frac{dJ(w, b)}{db}$



## 计算图

神经网络中的计算即是由多个计算网络输出的前向传播与计算梯度的后向传播构成。所谓的**反向传播（Back Propagation）**即是当我们需要计算最终值相对于某个特征变量的导数时，我们需要利用计算图中上一步的结点定义。

## Logistic 回归中的梯度下降法

假设输入的特征向量维度为 2，即输入参数共有 x1, w1, x2, w2, b 这五个。可以推导出如下的计算图：

![image-20210920153840546](https://gitee.com/pinboy/typora-image/raw/master/img/202109201538612.png)

首先反向求出 L 对于 a 的导数：$da=\frac{dL(a,y)}{da}=−\frac{y}{a}+\frac{1−y}{1−a}$

然后继续反向求出 L 对于 z 的导数：$dz=\frac{dL}{dz}=\frac{dL(a,y)}{dz}=\frac{dL}{da}\frac{da}{dz}=a−y$



依此类推求出最终的损失函数相较于原始参数的导数之后，根据如下公式进行参数更新：

- $w _1:=w _1−\alpha dw _1$
- $b:=b−\alpha db$

接下来我们需要将对于单个用例的损失函数扩展到整个训练集的代价函数：

- $J(w,b)=\frac{1}{m}\sum^m_{i=1}L(a^{(i)},y^{(i)})$
- $a^{(i)}=\hat{y}^{(i)}=\sigma(z^{(i)})=\sigma(w^Tx^{(i)}+b)$

我们可以对于某个权重参数 w1，其导数计算为：$\frac{\partial J(w,b)}{\partial{w_1}}=\frac{1}{m}\sum^m_{i=1}\frac{\partial L(a^{(i)},y^{(i)})}{\partial{w_1}}$

![image-20210920154550694](https://gitee.com/pinboy/typora-image/raw/master/img/202109201545743.png)

## 





## 向量化

在 Logistic 回归中，需要计算 z*=*w^T^x*+*b*

如果是非向量化的循环方式操作，代码可能如下：

```py
z = 0;
for i in range(n_x):
    z += w[i] * x[i]
z += b
```

而如果是向量化的操作，代码则会简洁很多，并带来近百倍的性能提升（并行指令）：

```py
z = np.dot(w, x) + b
```



## 广播

主要解决维度不匹配问题，默认调整为向shape最长的向量维度看齐

转置对秩为 1 的数组无效。因此，应该避免使用秩为 1 的数组，用 n * 1 的矩阵代替。例如，用`np.random.randn(5,1)`代替`np.random.randn(5)`。

如果得到了一个秩为 1 的数组，可以使用`reshape`进行转换。
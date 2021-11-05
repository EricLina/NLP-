# Week4

## 4.1 前向传播

**输入**：$a^{[l−1]}$

**输出**：$a^{[l]}，cache(z^{[l]})$

**计算过程**：

- $Z^{[l]}=W^{[l]}\cdot a^{[l-1]}+b^{[l]}$

- $a^{[l]}=g^{[l]}(Z^{[l]})$

```python
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T, X) + b)                                    # compute activation
    cost = -1/m*(np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)))                                 # compute cost
    ### END CODE HERE ###
```



## 4.2 反向传播

**输入**：$da^{[l]}$

**输出**：$da^{[l-1]}，dW^{[l]}，db^{[l]}$

**计算过程**：

- $dZ^{[l]}=da^{[l]}*g^{[l]}{'}(Z^{[l]})$

- $dW^{[l]}=dZ^{[l]}\cdot a^{[l-1]}$

- $db^{[l]}=dZ^{[l]}$

- $da^{[l-1]}=W^{[l]T}\cdot dZ^{[l]}$

代码：

```python
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = 1/m*(np.dot(X, ((A-Y).T)))
    db = 1/m*(np.sum(A-Y))
    ### END CODE HERE ###
```

```python
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate*dw
        b = b - learning_rate*db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
```



## 矩阵维度

![image-20210920172109068](https://gitee.com/pinboy/typora-image/raw/master/img/202109201721118.png)



- $Z^{[l]}=W^{[l]}\cdot a^{[l-1]}+b^{[l]}$

- **求导后不会改变向量维度**

- $n^{[l]}$代表第$l$层的神经元个数

  | $W^{[l]}: (n^{[l]}, n^{[l-1]})$  | $b^{[l]}: (n^{[l]}, 1)$  |
  | -------------------------------- | ------------------------ |
  | $dW^{[l]}: (n^{[l]}, n^{[l-1]})$ | $db^{[l]}: (n^{[l]}, 1)$ |

- 对于 Z、a，

- 向量化：$Z^{[l]}=W^{[l]}\cdot a^{[l-1]}+b^{[l]}$  ------> $Z^{[l]}=W^{[l]}\cdot X^{[l-1]}+b^{[l]}$    （a看成是一个样本得到的结果，X可以看成是用多个样本a训练得到的结果）

  | 向量化之前有                     | 向量化之后                       |
  | -------------------------------- | -------------------------------- |
  | $Z^{[l]}, a^{[l]}: (n^{[l]}, 1)$ | $Z^{[l]}, A^{[l]}: (n^{[l]}, m)$ |

  - （m是训练集X的大小） 向量化之后，相当于把所有的训练样本叠在一块

  在计算反向传播时，dZ、dA 的维度和 Z、A 是一样的。（求导后不会改变向量维度）





## 搭建神经网络

正向传播，输入$a^{[0]}$计算出 $a^{[1]}$,并且在计算过程中缓存（cache）  $Z^{[l]},a^{[1]},W^{[l]},b^{[l]}$ 供反向传播使用。 不断向前计算



![image-20210920173108435](https://gitee.com/pinboy/typora-image/raw/master/img/202109201731506.png)

反向传播，计算$ d a^{[l]}$，开始实现反向传播，用**链式法则**得到所有的导数项，W 和 b 也会在每一层被更新。

![image-20210920173446180](https://gitee.com/pinboy/typora-image/raw/master/img/202109201734260.png)



## 看个作业上的例子

![image-20210920202908366](https://gitee.com/pinboy/typora-image/raw/master/img/202109202029509.png)

一般的搭建顺序：

1. 定义神经网络结构（输入单元个数，隐层个数）
2. 随机初始化参数
3. 循环
   1. 前向传播
   2. 计算损失
   3. 反向传播
   4. 更新参数





## 为什么使用深层网络

> 对假设这是人脸识别过程，根据直觉，我们认为第1层识别边缘，第2层利用上一层的边缘结合成各个部位，第三层利用上一层的部位结合成人脸，再比如音频识别也类似，从简单到复杂一步步识别。前面的层识别一些低层次的特征，到后面的层就能结合前面的特征去探测更加复杂的东西。从而需要神经网络需要很多层，即深层网络。这些灵感来源是人类大脑，大脑识别也是从简单开始，然后再结合到整体。
>
> 同样的，对于语音识别，第一层神经网络可以学习到语言发音的一些音调，后面更深层次的网络可以检测到基本的音素，再到单词信息，逐渐加深可以学到短语、句子。
>
> 通过例子可以看到，随着神经网络的深度加深，模型能学习到更加复杂的问题，功能也更加强大。
>
>   其实深度学习只不过就是多隐藏层神经网络学习

## 参数和超参数

参数是模型的一个可计算出来的变量。（例如W,b)

超参数是人为提前设计的，能够控制网络的输出和性能。

典型的超参数有：

- 学习速率：α
- 迭代次数：N
- 隐藏层的层数：L
- 每一层的神经元个数：$n[1]n[1]，n[2]n[2]，...$
- 激活函数 g(z) 的选择


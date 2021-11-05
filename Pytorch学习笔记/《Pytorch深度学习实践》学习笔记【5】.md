# 《PyTorch深度学习实践》学习笔记 【5】CNN_1



## 十、CNN 1

### 10.1 卷积神经网络的基本结构

**卷积神经网络=特征提取+分类**

![image-20210818122528925](https://gitee.com/pinboy/typora-image/raw/master/img/202108181225985.png)

#### 10.1.1 特征提取

特征提取器，通过卷积运算，找到某种特征。由卷积层convolution和下采样Subsampling 构成。

一个图像，丢到一个卷积层里面仍然是一个3维的张量；

下采样，通道数不变，但是图片的宽度和高度变小（减少维数数量，减少计算量）

#### 10.1.2 分类器

将特征向量化后，用全连接网络进行来分类。

> [全连接层](https://blog.csdn.net/GoodShot/article/details/79330545?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162944639516780262514780%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=162944639516780262514780&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-79330545.first_rank_v2_pc_rank_v29&utm_term=%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82%E7%9A%84%E8%BE%93%E5%85%A5%E5%92%8C%E8%BE%93%E5%87%BA&spm=1018.2226.3001.4187)
>
> ![image-20210820160243689](https://gitee.com/pinboy/typora-image/raw/master/img/202108201602757.png)



### 10.2 Convolution （卷积层）



#### 10.2.1 背景知识

> RGB：
>
> 栅格图：按照像素格保存图像，每个像素格里面保存度值，如果是彩色图，每个格子里就是RGB三个通道的灰度。
>
> 矢量图：根据描述来绘制

#### 10.2.2 卷积的运算过程

https://zhuanlan.zhihu.com/p/63974249

1. #### 单通道卷积：

   **卷积核**

![image-20210818124400495](https://gitee.com/pinboy/typora-image/raw/master/img/202108181244568.png)

​				卷积核能够输出一层

- 卷积核数量=最终输出的通道数



2. #### **三通道卷积**

   每个通道对应一个卷积核（所以卷积核也得有三个）

   做数乘，再求和。

![image-20210818124436445](https://gitee.com/pinboy/typora-image/raw/master/img/202108181244667.png)

​	约定的画法：

![image-20210818125902410](https://gitee.com/pinboy/typora-image/raw/master/img/202108181259496.png)

- 卷积核的通道数量 = 输入通道数		（三通道输入---卷积核也有三个通道）

- 卷积核的总数= 	输出通道的数量    （M通道输出--   卷积核(filter)有M个）

  卷积核的大小自己定

  卷积层对图像的大小没有要求，只对通道数有关。所以定义一个卷积层只需要定义：1. 输入通道，2. 输出通道数，3. 卷积核数；

```
conv_layer = torch.nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size)
```



3. ####  卷积的Padding

   若对于一个大小为N×N的原图，经过大小为M×M的卷积核卷积后，仍然想要得到一个大小为N×N的图像，则需要对原图进行Padding，即外围填充。

- Padding的圈数： M/2 圈  （比如M=3 ， padding 3/2 = 1 圈； M=5， padding 5/2 = 2 圈）

  例如，对于一个5×5的原图，若想使用一个3×3的卷积核进行卷积，并获得一个同样5×5的图像，则需要进行Padding，**通常外围填充0**



![image-20210819085422047](https://gitee.com/pinboy/typora-image/raw/master/img/202108190854162.png)



- #### Stride

  每次移动卷积核时的步长。 

  大的步长能够有效地减小维数。



#### 10.3 Maxpooling 池化层

对于一个M*M的图像而言，被分割成大小相同的块，每个块的值取块内最大值，称为MaxPooling； （用均值就是AveragePooling）

通过池化层可以有效减小宽度和高度

只是缩小图片的宽度和高度，不改变通道数

![image-20210819090810089](https://gitee.com/pinboy/typora-image/raw/master/img/202108190908335.png)





### 10.4 简例

![image-20210819093744721](https://gitee.com/pinboy/typora-image/raw/master/img/202108190937996.png)

- 卷积层Conv2d  Layer和  池化层Pooling Layer 对 输入图像的大小没有要求（因为做的是简单的运算都能处理）
- 全过程中图像大小影响最大的是最后传入分类器的维度（传入的是向量，长度与图像大小有关）

#### 10.4.1 模型：

![image-20210819110738890](https://gitee.com/pinboy/typora-image/raw/master/img/202108191107972.png)

#### 10.4.2 代码：

```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)
    def forward(self, x):
    # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x))) #卷积层conv--> 池化层Pooling -->激活函数 relu
        x = F.relu(self.pooling(self.conv2(x)))	#传到下一层
        x = x.view(batch_size, -1) # flatten
        x = self.fc(x)
        return x
model = Net()
```

### 10.5  Using GPU：

1. 模型迁移到GPU

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
```

2. 把Input和target迁移到GPU（同一块显卡）

```
inputs, target = inputs.to(device), target.to(device)
```



## 十一、 CNN 2

**本节介绍非串行的网络，GoogleNet**

### 11.1  Inception Module

为了减少代码的冗余，有两种重要的方法，以抽象出相似的地方

1. 面向过程的函数
2. 面向对象的类

我们把网络中的相似块都封装起来，减少代码重复，每个模块称为Inception。

![image-20210819153655489](https://gitee.com/pinboy/typora-image/raw/master/img/202108191536586.png)

超参数难以选择

解决： 尝试不同的超参数配置，不同配置赋予不同的权重，然后根据效果好坏调整权重。





最终不同路径得出的结果 width和Height必须一致

如何保持输出的宽度和高度不变：

- Conv

  用padding外围填充 

- Average Pooling ：

  设置stride=1；padding

### 11.2 **1×1** 的卷积

理解起来很简单，就是按位置加权求和的一个算子；

what  is it？

1*1卷积核的数目和输入的通道有关（这里3个）

![image-20210819154836010](https://gitee.com/pinboy/typora-image/raw/master/img/202108191548077.png)

why we use it？

减少运算量

![image-20210819155158539](https://gitee.com/pinboy/typora-image/raw/master/img/202108191551600.png)

运算量直接缩小到1/10 ， 节约时间，也被称为network in network



### 11.3 四条路径的模型与代码

![image-20210819192430937](https://gitee.com/pinboy/typora-image/raw/master/img/202108191924069.png)



Concatenate 拼接：![image-20210819192746195](https://gitee.com/pinboy/typora-image/raw/master/img/202108191927345.png)

说明dim =1

输入的张量是(bacth,Channel,W,H) ，第一个维度是Channel

我们沿着Channel将四个拼接

#### 代码1：

定义Inception

```python
class InceptionA(nn.Module):
    def __init__(self, in_channels):    # 通道数作为输入
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels,16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)        # 按照通道数合并，总共16+24+24+24=88个通道
```

#### 代码2：

定义网络

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)
    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
```



- 训练轮数不是越多越好，可以在实验过程中保留较优的参数值







由于层数太多，带来了梯度消失问题：https://blog.csdn.net/superCally/article/details/55671064

- 可以用逐层训练解决（但是有些网络层数过多）

- Residual Net ： （DRN  ：Deep Residual Net）
  - H(x) = F(x) + x   这种设计可以有效解决梯度消失
  - F(x)与 x 是同维度的才能相加，所以经过层后的输出，尺寸要和原来的相同。
  
  

### 11.4 残差网络（Residual Net）

两种不同神经网络结构

1. 一般的
2. 跳链接

![image-20210819233913206](https://gitee.com/pinboy/typora-image/raw/master/img/202108192339359.png)



例子：

- 在“ 卷积-> ReLU-> 池化” 的架构后面加一层ResidualNet（红色块标注）

- 经过Residual Block， 数据的尺寸不变

![image-20210819234449266](https://gitee.com/pinboy/typora-image/raw/master/img/202108192344459.png)

​	ResidualBlock的实现：

![myfirst2](https://gitee.com/pinboy/typora-image/raw/master/img/202108192351616.gif)


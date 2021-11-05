# 《PyTorch深度学习实践》学习笔记 【1】
学习资源：
[《PyTorch深度学习实践》完结合集](https://www.bilibili.com/video/BV1Y7411d7Ys?p=2)

## 一. 绪论

###### 1. 人工智能的发展历程

1.  Rule—Based： 基于规则（如经典的积分计算）![image-20210814104752566](https://img-blog.csdnimg.cn/img_convert/f05bd661f1b7710fd6b8ae5be2c017fb.png)

2. Classic machine learning : 手动提取特征，使用特征![image-20210814104808634](https://img-blog.csdnimg.cn/img_convert/99473ec124427af9faff68bca84ec345.png)

3. representation learning ： 自动提取特征![image-20210814104854200](https://img-blog.csdnimg.cn/img_convert/afbcb039faad4fbf71e98b86b503cf08.png)

   **维度的诅咒**： (参数空间搜索组合爆炸)

   **解决方法**：使用更少的更有代表性的参数。![image-20210814105116283](https://img-blog.csdnimg.cn/img_convert/e0b3ca7168c17208237ee3c04691727e.png)

4. DeepLearning：

   将特征提取和特征函数训练两个过程结合起来，也称之为**End2End**（端到端）

###### 2. 经典的机器学习方法（见图）![image-20210814105908393](https://img-blog.csdnimg.cn/img_convert/52fc70c483e6fd04219d1e8225b68edf.png)

###### 3. SVM的兴衰

> SVM的局限性：*手工提取特征的限制；面对大数据处理效果不佳；更多无特征的数据出现；*
>
> DeepLearning出现后，效果比SVM更好

###### 4. 神经网络的发展历史

> 人工神经网络（Artificial Neural Networks，简写为ANNs）也简称为神经网络（NNs）或称作连接模型（Connection Model），它是一种模仿动物神经网络行为特征，进行分布式并行信息处理的算法数学模型。这种网络依靠系统的复杂程度，通过调整内部大量节点之间相互连接的关系，从而达到处理信息的目的。
>
> 神经网络是通过对人脑的基本单元——神经元的建模和联接，探索模拟人脑神经系统功能的模型，并研制一种具有学习、联想、记忆和模式识别等智能信息处理功能的人工系统。神经网络的一个重要特性是它能够从环境中学习，并把学习的结果分布存储于网络的突触连接中。神经网络的学习是一个过程，在其所处环境的激励下，相继给网络输入一些样本模式，并按照一定的规则（学习算法）调整网络各层的权值矩阵，待网络各层权值都收敛到一定值，学习过程结束。然后我们就可以用生成的神经网络来对真实数据做分类。

		神经网络的发展，是自底向上的，模块化的，随着基本块能力的增强，能解决的问题也越来越多。
	
	     深度学习的发展动力：模块的搭建更加方便， 数据的获取更容易，算力一直在发展。

###### 6. 常用工具

![image-20210814111137216](https://img-blog.csdnimg.cn/img_convert/0be8e5be19b9f86aaade276c09c286da.png)

###### 7. Pytorch的安装

[Pytorch cpu版本安装教程（附加GPU版本安装教程）](https://blog.csdn.net/qq_41375318/article/details/102483339?ops_request_misc=&request_id=&biz_id=102&utm_term=pytorch%20%E6%8C%89%E7%85%A7&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-.nonecase&spm=1018.2226.3001.4187)

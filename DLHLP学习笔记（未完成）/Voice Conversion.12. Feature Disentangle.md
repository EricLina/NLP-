## VC是什么？

- 输入一段声音，输出另一段声音（同声传译）（转化语调）（修音）（去噪）
  - voice conversation ： 向量到向量![image-20210923111854131](https://gitee.com/pinboy/typora-image/raw/master/img/202109231119441.png)
  - 需要一个Vocoder 将向量转化成语音  ，常见的有Griffin-Lim algorithm，WaveNet
- 大致分为两类，
  - 成对数据![image-20210923112017296](https://gitee.com/pinboy/typora-image/raw/master/img/202109231120355.png)
  - 非成对数据![image-20210923112025587](https://gitee.com/pinboy/typora-image/raw/master/img/202109231120726.png)

## Feature Disentangle 

​	将一段语音 拆解成 内容(content)+说话者特征(speaker)

![image-20210923112524286](https://gitee.com/pinboy/typora-image/raw/master/img/202109231125535.png)

训练好模型后，同一段内容可以换成不同的speakerEncoder来输出不同结果

![image-20210923112545865](C:\Users\26421\AppData\Roaming\Typora\typora-user-images\image-20210923112545865.png)


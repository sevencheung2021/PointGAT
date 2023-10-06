<!--
 * @Author: Seven Rong Cheung
 * @Date: 2023-04-27 11:36:12
 * @LastEditors: Seven Rong Cheung
 * @LastEditTime: 2023-09-30 11:37:23
 * @FilePath: /undefined/Users/rongzhang/Downloads/github/PointGAT/README.md
 * @Description: 
 * 
 * Copyright (c) 2023 by {rongzhangthu@yeah.net}, All Rights Reserved. 
-->


### 环境

python 3.8.13

torch==1.7.1

numpy==1.22.1

pandas==1.4.2

torch-geometric==1.7.2

pyGPGO==0.5.1

matplotlib-inline==0.1.6

rdkit==2021.09.5


### 运行项目
src/main.py，根据报错安装其他所需的 python 包。



```
/Data/    存放C10 训练需要的数据集相关文件
/RawData    存放创建C10数据集的原始数据文件，C10 数据集的 mol文件，log 文件,xyz文件，DM21 能量文件
/weights    存放模型权重

/src/    存放代码文件
/src/pointgat.py    项目主代码文件，所有的数据集都已经准备好，python pointgat.py 即可运行项目

README.md    介绍项目概况
```


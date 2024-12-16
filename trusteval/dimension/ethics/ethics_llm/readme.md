## Ethics code and data

#### 目前pipeline.py可以一键运行完成所有的generation部分 生成出来的内容会在data文件夹中


### 1. 文件夹介绍

- metadata_curator（存储了metadata curator的代码和处理后的数据）
  - code文件夹存储了处理留个数据集的code
  - done data存储了使用dataset pool之后的数据，具体的6个数据集的介绍和URL存储在"Readme_TrustAGI_Ethics_dataset_pool.md"中；
  - ori_dataset存储了原始数据集（处理前的）(文件有点大先不放进来了)


- test_case_builder
  - code文件夹存储了处理metadata curator之后生成具体case的代码
  - data存储了test case builder生成的具体数据


- contextual_variator
  - code存储了处理case的contextual variator代码
  - data存储了case经过contextual variator之后的数据


### 2. 代码解释

- metadata_curator：主要是根据不同raw数据集的格式分别处理得到json的代码。（根据数据集结果不同而有一定区别）
- test_case_builder：主要是根据不同的数据集数据形式和询问方式设计不同的prompt，和获取不同的key对应的value来给LLM生成case
- contextual_variator：主要是根据不同数据集能做的处理进行不同的多样性变化。

### 3. 主要的代码
- 一般命名为1、2、3.1、3.2、3.3、3.4、3.5、5、6、7
  - 3有五个不同的subset所以需要分别处理
  - 4数据集未公开所以没有4的部分
# 关于qwen0.5B 多语言推理可视化的理解

## 前八层现象对比

1-4 layers bn language
![1-4ceng bn language](./bn_tsne_plots/tsne_layers_1_to_4.png)

5-8 layers bn language
![5-8ceng bn language](./bn_tsne_plots/tsne_layers_5_to_8.png)

1-4 layers en language
![1-4ceng en language](./en_tsne_plots/tsne_layers_1_to_4.png)

5-8 layers en language
![5-8ceng en language](./en_tsne_plots/tsne_layers_5_to_8.png)

### 分布现象

#### de ru es fr 这四种语言作为1%语料的印欧语系 zh ja 语言作为1%语料的汉藏语系 

1. 同语系的语言 西班牙语和英语 在前四层的分布形状相似程度 相较于不同语系语言 英语和中文 相似程度大（可能的原因：）
![]
3. 在0-4层，甚至是所有层中 像孟加拉语和斯瓦希里语（训练语料中占比很少的语言） 样本点相较于英语聚集程度很高（说明模型对这种小众语言的理解程度低，没有发现各个问题之间的差异）

2. 在4-8层 其他语言 中文 分布形状逐渐像向文（可能是由于英文的训练语料占比巨大）靠近（可能的原因：）



4. 在8-12层中 尤其是12层，1%语料的语言 无论是何种语系的 样本点分布重叠度较高 但是都和100%语料（英语）的样本点分布区分开来了

5. 在12层之后 对比1%语料的不同语系语言 es zh 分布形状相似 (语系之间的差异基本消失，可能是因为（表层理解已经结束）翻译的过程结束) 

6. 在最终层 所有语言都相似 此时可能和小模型模型以英语回答各种语言问题（除了日语问题有30%以中文回答之外）有关 最后全部和英语分布形状




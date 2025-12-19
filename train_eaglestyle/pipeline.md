# MTP（multi-token prediction）
## 组成
Base model：自回归大模型

MTP head：基于扩散原理的单层transformer layer

## 工作流
base model最后一层输出的隐藏状态，作为condition，拼接上block_length长得mask，输入MTP head, head一次预测多个token。

我的MTP头不需要保存kvcache.

在输入到mtp head的时候，要注意hiddenstates是一个tensor，而input_ids是一个有整数组成的数组，因此不能直接concat，需要先将input转换成embedding。我已实现。训练的时候要注意


# 训练思路
## 扩散头冻结
扩散头不在数据集上训练，而是拿大模型某个位置最后输出的隐藏层和他吐出的后续L个token(logits)进行训练.
### step1 Data collection
假设给大模型一个输入，我们让LLM完成正常推理，并且保存下输出中每个位置的模型最终输出隐藏状态（h）和映射得到的token（t）(logits分布）。

### step2 训练过程
从输出的第一个词开始，执行：假设当前token位置是l，将h_l当作扩散头的条件，加入L长的mask掩码，用离散扩散语言模型的方式训练。生成L长序列，计算损失并反向传播。之后再从l+1位置开始进行上述过程。

### step3 损失函数
生成L长的序列，这些生成token的标签就是原始大模型的t_l到t_（l+L）的token，计算损失。

设大语言模型（LLM）对给定提示生成的序列为：
$
\mathbf{t} = (t_1, t_2, \dots, t_T), \quad t_i \in \{1, 2, \dots, V\},
$
其中 $V$ 为词表大小。

对应地，从 LLM 最后一层提取的隐藏状态序列为：
$
\mathbf{H} = (\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_T), \quad \mathbf{h}_i \in \mathbb{R}^d,
$
其中 $d$ 为隐藏维度。

我们训练一个以扩散过程为基础的多步预测头（diffusion-based multi-prediction head），其目标是：  
对于每个位置 $l = 1, 2, \dots, T - L$，以 $\mathbf{h}_l$ 作为条件，预测后续 $L$ 个真实 token：
$
\mathbf{t}_{l+1:l+L} = (t_{l+1}, t_{l+2}, \dots, t_{l+L}).
$

该预测头建模条件分布：
$
p_\theta(\tilde{\mathbf{t}}_{l+1:l+L} \mid \mathbf{h}_l),
$
并旨在使生成序列 $\tilde{\mathbf{t}}_{l+1:l+L}$ 逼近真实目标 $\mathbf{t}_{l+1:l+L}$。

训练时采用的损失函数（例如离散扩散中的交叉熵或去噪目标）可形式化为：
$
\mathcal{L} = \sum_{l=1}^{T - L} \sum_{k=1}^{L} -\log p_\theta\big(t_{l+k} \mid \mathbf{h}_l, \mathbf{z}_{l+k}^{(t)}\big),
$
其中 $\mathbf{z}_{l+k}^{(t)}$ 表示扩散过程中在时间步 $t$ 的带噪 token 表示（具体形式取决于所选扩散框架）。

每个起始位置 $l$ 构成一个独立的训练样本，整体训练集由所有有效位置 $\{l\}$ 构成。

------
## 训练进度

1. 按照上述方法训练后，效果不好。预测出的token正确率很低
分析：大模型隐藏状态，在自己训练时目标就只有ntp，不包含未来信息。因此mtp头难以生成未来token

需要让大模型学会生成未来信息。
而我不想对于原模型微调。能否再加一个attention层，综合历史token，充当adapter，将原来隐藏状态调整为包含未来信息的
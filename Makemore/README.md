# makemore

makemore 接收一个文本文件作为输入，其中每一行被假定为一个训练样本，然后生成更多类似的样本。其底层是一个自回归的字符级语言模型，可以选择的模型范围很广，从二元组 (bigrams) 一直到 Transformer（与 GPT 中看到的一样）。例如，我们可以给它输入一个名字数据库，makemore 就能生成听起来像名字但又不存在的酷炫婴儿名字。或者，如果我们给它输入一个公司名称数据库，那么我们就可以为公司生成新的名称创意。或者我们也可以只给它输入有效的拼字游戏单词，然后生成类似英语的胡言乱语。

这并非一个拥有无数开关和旋钮的重量级库。它只是一个可供修改的文件，主要用于教育目的。[PyTorch](https://pytorch.org) 是唯一的要求。

当前的实现遵循了几篇关键论文：

- Bigram（一个字符通过计数查找表预测下一个字符）
- MLP，遵循 [Bengio 等人 2003 年的论文](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- CNN，遵循 [DeepMind WaveNet 2016 年的论文](https://arxiv.org/abs/1609.03499) （进行中...）
- RNN，遵循 [Mikolov 等人 2010 年的论文](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM，遵循 [Graves 等人 2014 年的论文](https://arxiv.org/abs/1308.0850)
- GRU，遵循 [Kyunghyun Cho 等人 2014 年的论文](https://arxiv.org/abs/1409.1259)
- Transformer，遵循 [Vaswani 等人 2017 年的论文](https://arxiv.org/abs/1706.03762)

### 用法

以附带的 `names.txt` 数据集为例，它包含了从 [ssa.gov](https://www.ssa.gov/oact/babynames/) 获取的 2018 年最常见的 32K 个名字。看起来像这样：

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

让我们将脚本指向它：

```bash
$ python makemore.py -i names.txt -o names
```

训练进度、日志和模型都将保存到工作目录 `names` 中。默认模型是一个仅有 200K 参数的超小型 transformer；还有更多可用的训练配置——请查看 argparse 参数并阅读代码。训练不需要任何特殊硬件，它可以在我的 Macbook Air 上运行，也可以在其他任何设备上运行，但如果你有 GPU，训练速度会更快。随着训练的进行，脚本会打印一些样本。但是，如果你想手动采样，可以使用 `--sample-only` 标志，例如，在另一个终端中执行：

```bash
$ python makemore.py -i names.txt -o names --sample-only
```

这将加载迄今为止最好的模型，并根据需要打印更多样本。以下是一些在当前默认设置下最终生成的独特婴儿名字（测试对数似然约为 1.92，尽管通过一些超参数调整可以实现更低的对数似然）：

```
dontell
khylum
camatena
aeriline
najlah
sherrith
ryel
irmi
taislee
mortaz
akarli
maxfelynn
biolett
zendy
laisa
halliliana
goralynn
brodynn
romima
chiyomin
loghlyn
melichae
mahmed
irot
helicha
besdy
ebokun
lucianno
```

玩得开心！

+---------------+      +----------------------+      +-----------------+
| [CharDataset] | ---->| [InfiniteDataLoader] | ---->| [Training Loop] |
+---------------+      +----------------------+      +--------+--------+
                                                               |
                                                               v
+-------------------------------------------------------------------------------------------------------------------------+
|                                                    Model Training Branches                                                |
+-------------------------------------------------------------------------------------------------------------------------+
      |                     |                      |                      |                      |
      v                     v                      v                      v                      v
+---------------+     +-----------+          +-----------+          +-----------+          +-----------+
| [Transformer] |     |   [BoW]   |          |   [RNN]   |          |   [MLP]   |          |  [Bigram] |
+---------------+     +-----------+          +-----------+          +-----------+          +-----------+
      |                     |                      |                      |                      |
      v                     v                      v                      v                      v
+---------------+     +------------+         +-----------+          +-------------+        +-------------+
|    [Block]    |     | [BoWBlock] |         | [RNNCell] |          | (MLP Int.)  |        | (Bigram Int)|
| (Contains...) |     +------------+         |  or       |          | (Emb,Concat,|        | (Emb Lookup)|
+---------------+           |                | [GRUCell] |          | Hid+Act...) |        +-------------+
      |                     v                +-----------+          +-------------+              |
      v               +-------------+              |                      |                      |
+-------------+       | [CausalBoW] |              |                      |                      |
| [CausalSelf |       +-------------+              |                      |                      |
|  Attention] |             |                      |                      |                      |
+-------------+             |                      |                      |                      |
      |                     |                      |                      |                      |
      | (Block Output)      | (Pooled Embed)       | (Hidden State)       | (Last Hidden)        | (Embedding)
      v                     v                      v                      v                      v
+-------------+     +-------------+        +-------------+        +-------------+        +-------------+
| [Linear     |     | [Linear     |        | [Linear     |        | [Linear     |        | [Linear     |
| (Output)]   |     | (Output)]   |        | (Output)]   |        | (Output)]   |        | (Output)]   |
+-------------+     +-------------+        +-------------+        +-------------+        +-------------+
      |                     |                      |                      |                      |
      v                     v                      v                      v                      v
+-------------+     +-------------+        +-------------+        +-------------+        +-------------+
|   Logits    |     |   Logits    |        |   Logits    |        |   Logits    |        |   Logits    |
+-------------+     +-------------+        +-------------+        +-------------+        +-------------+
      |                     |                      |                      |                      |
      +---------------------+----------------------+----------------------+----------------------+
                                             |
                                             v
                                     +-----------------+
                                     | Loss Calculation|
                                     | (in Training L.)|
                                     +-----------------+
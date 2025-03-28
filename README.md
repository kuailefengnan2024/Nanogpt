# nanoGPT
# nanoGPT（纳米GPT）

![nanoGPT](assets/nanogpt.jpg)

最简单、最快速的中型GPT训练/微调代码库。它是[minGPT](https://github.com/karpathy/minGPT)的重写版本，注重实用性而非教育性。目前仍在积极开发中，但当前`train.py`文件可以在OpenWebText上复现GPT-2(124M)，在单个8XA100 40GB节点上训练约4天。代码本身简洁易读：`train.py`是一个约300行的标准训练循环，`model.py`是一个约300行的GPT模型定义，可以选择性地从OpenAI加载GPT-2权重。就这么简单。

![repro124m](assets/gpt2_124M_loss.png)

由于代码非常简单，因此很容易根据您的需求进行修改，从头开始训练新模型，或者微调预训练的检查点（例如，目前可用作起点的最大模型是OpenAI的GPT-2 1.3B模型）。

## 安装

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

依赖项：

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` 用于huggingface transformers <3 (加载GPT-2检查点)
-  `datasets` 用于huggingface数据集 <3 (如果您想下载+预处理OpenWebText)
-  `tiktoken` 用于OpenAI的快速BPE编码 <3
-  `wandb` 用于可选的日志记录 <3
-  `tqdm` 用于进度条 <3

## 快速开始

如果你不是深度学习专业人士，只想感受一下魔力并入门，最快的开始方式是在莎士比亚作品上训练一个字符级GPT。首先，我们将其下载为单个(1MB)文件，并将其从原始文本转换为一个大型整数流：

```sh
python data/shakespeare_char/prepare.py
```

这将在数据目录中创建`train.bin`和`val.bin`。现在是时候训练你的GPT了。它的大小很大程度上取决于你系统的计算资源：

**我有一个GPU**。太好了，我们可以使用[config/train_shakespeare_char.py](config/train_shakespeare_char.py)配置文件中提供的设置快速训练一个小型GPT：

```sh
python train.py config/train_shakespeare_char.py
```

如果你查看其中的内容，你会看到我们正在训练一个GPT，其上下文大小最多为256个字符，384个特征通道，它是一个6层Transformer，每层有6个头。在一个A100 GPU上，这个训练运行大约需要3分钟，最佳验证损失为1.4697。根据配置，模型检查点被写入`--out_dir`目录`out-shakespeare-char`。所以一旦训练完成，我们可以通过指向这个目录来从最佳模型中采样：

```sh
python sample.py --out_dir=out-shakespeare-char
```

这会生成一些样本，例如：

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

哈哈 `¯\_(ツ)_/¯`。对于在GPU上训练3分钟的字符级模型来说还不错。通过在此数据集上微调预训练的GPT-2模型，很可能获得更好的结果（稍后参见微调部分）。

**我只有一台MacBook**（或其他便宜的电脑）。别担心，我们仍然可以训练GPT，但我们要把要求降低一点。我建议在安装时获取最新的PyTorch nightly版本（[在这里选择](https://pytorch.org/get-started/locally/)），因为它很可能使你的代码更高效。但即使没有它，一个简单的训练运行可能如下所示：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

在这里，由于我们在CPU而不是GPU上运行，我们必须设置`--device=cpu`并关闭PyTorch 2.0编译功能`--compile=False`。然后，我们评估时会得到更嘈杂但更快的估计（`--eval_iters=20`，从200减少），我们的上下文大小只有64个字符而不是256个，每次迭代的批量大小只有12个示例，而不是64个。我们还将使用更小的Transformer（4层，4个头，128嵌入大小），并将迭代次数减少到2000（相应地通常将学习率衰减到`--lr_decay_iters`附近）。由于我们的网络如此之小，我们也减轻了正则化（`--dropout=0.0`）。这仍然需要约3分钟，但损失只有1.88，因此样本也更差，但仍然很有趣：

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
生成的样本如下：

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

对于CPU上约3分钟的训练来说，能捕捉到一些角色的特征已经不错了。如果你愿意等待更长时间，可以随意调整超参数，增加网络大小、上下文长度（`--block_size`）、训练长度等。

最后，对于苹果Silicon MacBook和较新的PyTorch版本，确保添加`--device=mps`（Metal Performance Shaders的缩写）；PyTorch然后会使用片上GPU，可以*显著*加速训练（2-3倍）并允许你使用更大的网络。详见[Issue 28](https://github.com/karpathy/nanoGPT/issues/28)。

## 复现GPT-2

更专业的深度学习人士可能对复现GPT-2结果更感兴趣。那么开始吧——我们首先对数据集进行分词，在这种情况下是[OpenWebText](https://openwebtext2.readthedocs.io/en/latest/)，这是OpenAI的（私有）WebText的开放复制版：

```sh
python data/openwebtext/prepare.py
```

这将下载并分词[OpenWebText](https://huggingface.co/datasets/openwebtext)数据集。它会创建一个`train.bin`和`val.bin`，其中包含GPT2 BPE词元ID，存储为原始uint16字节。然后我们准备开始训练。要复现GPT-2(124M)，你至少需要一个8X A100 40GB节点并运行：

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

这将使用PyTorch分布式数据并行(DDP)运行约4天，损失降至约2.85。现在，仅在OWT上评估的GPT-2模型的验证损失约为3.11，但如果你对其进行微调，它将降至约2.85（由于明显的领域差距），使两个模型大致匹配。

如果你在集群环境中，并且有幸拥有多个GPU节点，你可以让GPU跨节点协同工作，例如跨2个节点：

```sh
# 在第一个（主）节点上运行，IP示例为123.456.123.456：
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# 在工作节点上运行：
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

最好对你的互连进行基准测试（例如iperf3）。特别是，如果你没有Infiniband，那么还要在上述启动命令前加上`NCCL_IB_DISABLE=1`。你的多节点训练将会工作，但很可能_非常缓慢_。默认情况下，检查点会定期写入`--out_dir`。我们可以通过简单地运行`python sample.py`从模型中进行采样。

最后，要在单个GPU上训练，只需运行`python train.py`脚本。看看它的所有参数，该脚本力求非常可读、可黑客化和透明。你很可能需要根据自己的需求调整其中的许多变量。

## 基准测试

OpenAI GPT-2检查点允许我们为openwebtext建立一些基准。我们可以通过以下方式获取数字：

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

并观察训练和验证上的以下损失：

| 模型 | 参数 | 训练损失 | 验证损失 |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

然而，我们必须注意GPT-2是在（封闭的，从未发布的）WebText上训练的，而OpenWebText只是对该数据集的尽力而为的开放复制。这意味着存在数据集领域差距。实际上，取GPT-2(124M)检查点并直接在OWT上微调一段时间，损失可降至约2.85。这然后成为关于复现的更合适的基准。

## 微调

微调与训练没有区别，我们只需确保从预训练模型初始化并以较小的学习率进行训练。有关如何在新文本上微调GPT的例子，请到`data/shakespeare`并运行`prepare.py`下载小型莎士比亚数据集，并使用OpenAI的GPT-2 BPE分词器将其呈现为`train.bin`和`val.bin`。与OpenWebText不同，这将在几秒钟内运行。微调可能需要很少的时间，例如在单个GPU上只需几分钟。运行一个微调示例：

```sh
python train.py config/finetune_shakespeare.py
```

这将加载`config/finetune_shakespeare.py`中的配置参数覆盖（虽然我没有太多调整它们）。基本上，我们通过`init_from`从GPT2检查点初始化，并像平常一样训练，只是时间更短，学习率更小。如果你内存不足，请尝试减小模型大小（分别是`{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`）或可能减小`block_size`（上下文长度）。最佳检查点（最低验证损失）将在`out_dir`目录中，根据配置文件，默认为`out-shakespeare`。然后你可以运行`sample.py --out_dir=out-shakespeare`中的代码：

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

哇，GPT，进入了一些黑暗的地方啊。我没有太多调整配置中的超参数，随时尝试吧！

## 采样/推理

使用脚本`sample.py`从OpenAI发布的预训练GPT-2模型或你自己训练的模型中采样。例如，这里是从最大的可用`gpt2-xl`模型中采样的方法：

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

如果你想从你训练的模型中采样，使用`--out_dir`适当地指向代码。你也可以用文件中的一些文本来提示模型，例如```python sample.py --start=FILE:prompt.txt```。

## 效率注意事项

对于简单的模型基准测试和分析，`bench.py`可能很有用。它与`train.py`的训练循环中发生的情况相同，但省略了许多其他复杂性。

请注意，代码默认使用[PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/)。在撰写本文时（2022年12月29日），这使得`torch.compile()`在夜间版本中可用。一行代码的改进非常明显，例如将迭代时间从每次迭代约250毫秒减少到135毫秒。干得好，PyTorch团队！

## 待办事项

- 研究并添加FSDP而不是DDP
- 在标准评估上评估零样本困惑度（例如LAMBADA？HELM？等）
- 调优微调脚本，我认为超参数不是很好
- 在训练期间安排线性批量大小增加
- 合并其他嵌入（旋转，alibi）
- 在检查点中将优化缓冲区与模型参数分开
- 关于网络健康的额外日志记录（例如梯度裁剪事件，幅度）
- 更多关于更好初始化等的研究

## 故障排除

请注意，默认情况下，这个仓库使用PyTorch 2.0（即`torch.compile`）。这是相当新的和实验性的，尚未在所有平台上可用（例如Windows）。如果你遇到相关错误消息，请尝试通过添加`--compile=False`标志来禁用它。这会减慢代码但至少它能运行。

关于这个仓库、GPT和语言建模的一些背景，观看我的[Zero To Hero系列](https://karpathy.ai/zero-to-hero.html)可能会有所帮助。特别是，如果你有一些先前的语言建模背景，[GPT视频](https://www.youtube.com/watch?v=kCc8FmEb1nY)很受欢迎。

有关更多问题/讨论，请随时访问Discord上的**#nanoGPT**：

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## 致谢

所有nanoGPT实验都由[Lambda labs](https://lambdalabs.com)的GPU提供支持，这是我最喜欢的云GPU提供商。感谢Lambda labs赞助nanoGPT！

"""
你给这个脚本一些单词（每行一个），它会生成更多类似的东西。
使用最先进的 Transformer AI 技术
这段代码旨在非常易于修改。根据你的需求进行调整。

与 minGPT 的区别：
- 我删除了从 GPT2 权重初始化的 from_pretrained 函数
- 我删除了 dropout 层，因为我们在这里训练的模型很小，
  在这个阶段和这个规模下没有必要理解它。
- 我删除了权重衰减以及围绕哪些参数进行权重衰减的所有复杂性。
  我相信这在我们这里的操作规模下不会产生巨大差异。


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

"""



import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    block_size: int = None # 输入整数序列的长度
    vocab_size: int = None # 输入整数的范围在 [0 .. vocab_size -1] 之间
    # 下面的参数以略微不同的方式控制每个模型的大小
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4

# -----------------------------------------------------------------------------
# Transformer 语言模型（与 GPT-2 中使用的完全相同）

class NewGELU(nn.Module):
    """
    当前 Google BERT 仓库中 GELU 激活函数的实现（与 OpenAI GPT 相同）。
    参考：高斯误差线性单元 (GELU) 论文：https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    一个普通的多头掩码自注意力层，末尾带有一个投影。
    可以在这里使用 torch.nn.MultiheadAttention，但我在这里包含了一个
    显式的实现，以表明这里没有什么太可怕的东西。
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 所有头的键、查询、值投影，但在一个批次中
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 因果掩码，确保注意力只应用于输入序列的左侧
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # 批大小、序列长度、嵌入维度 (n_embd)

        # 计算批次中所有头的查询、键、值，并将头移动到批次维度
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 因果自注意力；自注意力：(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 重新组合所有头的输出

        # 输出投影
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ 一个不起眼的 Transformer 块 """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP 前向传播

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """ Transformer 语言模型，与 GPT-2 中看到的一样 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 报告参数数量（注意我们不计算 lm_head 中的解码器参数）
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # 形状 (1, t)

        # 前向传播 GPT 模型本身
        tok_emb = self.transformer.wte(idx) # token 嵌入，形状 (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # 位置嵌入，形状 (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # 如果给定了目标，则计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# 词袋 (BoW) 语言模型

class CausalBoW(nn.Module):
    """
    因果词袋。对前面的元素进行平均，看起来很像
    你在 Transformer 中找到的 CausalAttention 模块，原因不明 ;)
    """
    def __init__(self, config):
        super().__init__()

        # 用于屏蔽向量并保留自回归属性
        self.block_size = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # 批大小、序列长度、n_embd

        # 对所有前面的 token 特征进行加权平均
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ x # (B, T, T) x (B, T, C) -> (B, T, C)

        return y

class BoWBlock(nn.Module):
    """ 收集 BoW 特征并添加一个 MLP """

    def __init__(self, config):
        super().__init__()

        # 因果 BoW 模块
        self.cbow = CausalBoW(config)
        # MLP 组装器
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, config.n_embd2),
            c_proj  = nn.Linear(config.n_embd2, config.n_embd),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x))) # MLP 前向传播

    def forward(self, x):
        x = x + self.cbow(x)
        x = x + self.mlpf(x)
        return x

class BoW(nn.Module):
    """
    获取之前的 block_size 个 token，使用查找表对其进行编码，
    也使用查找表对其位置进行编码，然后将所有这些嵌入平均起来，
    并使用它来预测下一个 token。
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # token 嵌入
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置嵌入
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # 上下文块
        self.context_block = BoWBlock(config)
        # 语言模型头解码器层
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):

        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # 形状 (1, t)

        # 前向传播 token 和位置嵌入层
        tok_emb = self.wte(idx) # token 嵌入，形状 (b, t, n_embd)
        pos_emb = self.wpe(pos) # 位置嵌入，形状 (1, t, n_embd)
        # 相加并通过解码器 MLP
        x = tok_emb + pos_emb
        # 运行词袋上下文模块
        x = self.context_block(x)
        # 解码为下一个 token 的概率
        logits = self.lm_head(x)

        # 如果给定了目标，则计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
"""
循环神经网络语言模型：普通 RNN 循环或 GRU。
没有实现 LSTM，因为它的 API 有点烦人，因为它既有隐藏状态又有单元状态，
但它与 GRU 非常相似，并且在实践中效果一样好。
"""

class RNNCell(nn.Module):
    """
    'Cell' 的工作是：
    获取当前时间步 x_{t} 的输入和上一个时间步 h_{t-1} 的隐藏状态，
    并返回当前时间步的隐藏状态 h_{t}。
    """
    def __init__(self, config):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        ht = F.tanh(self.xh_to_h(xh))
        return ht

class GRUCell(nn.Module):
    """
    与 RNN cell 的工作相同，但循环公式更复杂一些，
    这使得 GRU 更具表现力且更易于优化。
    """
    def __init__(self, config):
        super().__init__()
        # 输入门、遗忘门、输出门、候选门
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        # 首先使用重置门将隐藏状态的某些通道置零
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev
        # 计算候选的新隐藏状态 hbar
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr))
        # 计算更新门，确定每个通道是否应该更新
        z = F.sigmoid(self.xh_to_z(xh))
        # 混合先前的隐藏状态和新的候选隐藏状态
        ht = (1 - z) * hprev + z * hbar
        return ht

class RNN(nn.Module):

    def __init__(self, config, cell_type):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2)) # 起始隐藏状态
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # token 嵌入表
        if cell_type == 'rnn':
            self.cell = RNNCell(config)
        elif cell_type == 'gru':
            self.cell = GRUCell(config)
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        # 为了效率，一次性嵌入所有整数
        emb = self.wte(idx) # (b, t, n_embd)

        # 按顺序迭代输入并更新每个时间步的 RNN 状态
        hprev = self.start.expand((b, -1)) # 扩展批次维度
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :] # (b, n_embd)
            ht = self.cell(xt, hprev) # (b, n_embd2)
            hprev = ht
            hiddens.append(ht)

        # 解码输出
        hidden = torch.stack(hiddens, 1) # (b, t, n_embd2)
        logits = self.lm_head(hidden)

        # 如果给定了目标，则计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# MLP 语言模型

class MLP(nn.Module):
    """
    获取之前的 block_size 个 token，使用查找表对其进行编码，
    连接向量并通过 MLP 预测下一个 token。

    参考：
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd) # token 嵌入表
        # 上面一行中的 +1 用于特殊的 <BLANK> token，如果在输入序列开始之前编码 token，则插入该 token
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):

        # 收集前 3 个词的词嵌入
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx) # token 嵌入，形状 (b, t, n_embd)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size # 特殊的 <BLANK> token
            embs.append(tok_emb)

        # 将所有嵌入连接在一起并通过 MLP
        x = torch.cat(embs, -1) # (b, t, n_embd * block_size)
        logits = self.mlp(x)

        # 如果给定了目标，则计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# Bigram 语言模型

class Bigram(nn.Module):
    """
    Bigram 语言模型 '神经网络'，仅仅是一个查找表，
    根据前一个字符给出下一个字符的 logits。
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1 # 这个模型只需要一个前驱字符来预测下一个

    def forward(self, idx, targets=None):

         # '前向传播'，哈哈
        logits = self.logits[idx]

        # 如果给定了目标，则计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# 用于评估和从模型中采样的辅助函数

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    获取一个条件序列索引 idx (形状为 (b, t) 的 LongTensor)，并完成序列 max_new_tokens 次，
    每次将预测反馈给模型。
    最可能的情况是，你需要确保为此操作处于 model.eval() 模式。
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # 如果序列上下文太长，我们必须在 block_size 处截断它
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # 前向传播模型以获取序列中索引的 logits
        logits, _ = model(idx_cond)
        # 获取最后一步的 logits 并按期望的温度缩放
        logits = logits[:, -1, :] / temperature
        # 可选地将 logits 裁剪为仅前 k 个选项
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # 应用 softmax 将 logits 转换为（归一化的）概率
        probs = F.softmax(logits, dim=-1)
        # 要么从分布中采样，要么取最可能的元素
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # 将采样到的索引附加到运行序列并继续
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def print_samples(num=10):
    """ 从模型中采样并漂亮地打印解码后的样本 """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 是因为我们已经以 <START> token (索引 0) 开始了
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # 获取采样整数的第 i 行，作为 python 列表
        row = X_samp[i, 1:].tolist() # 注意：我们需要裁剪掉第一个 <START> token
        # token 0 是 <STOP> token，所以我们在此处裁剪输出序列
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        # 分别跟踪我们见过和没见过的样本
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print('-'*80)
    for lst, desc in [(train_samples, '在训练集中'), (test_samples, '在测试集中'), (new_samples, '新的')]:
        print(f"{len(lst)} 个样本 {desc}:")
        for word in lst:
            print(word)
    print('-'*80)

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # 将模型重置回训练模式
    return mean_loss

# -----------------------------------------------------------------------------
# 用于创建发出单词的训练和测试数据集的辅助函数

class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} # 反向映射

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1 # 所有可能的字符和特殊的 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token 后跟单词

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # 索引 -1 将在不活动位置屏蔽损失
        return x, y

def create_datasets(input_file):

    # 预处理输入文本文件
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words] # 去除任何前导或尾随空格
    words = [w for w in words if w] # 去除任何空字符串
    chars = sorted(list(set(''.join(words)))) # 所有可能的字符
    max_word_length = max(len(w) for w in words)
    print(f"数据集中的示例数量：{len(words)}")
    print(f"最大单词长度：{max_word_length}")
    print(f"词汇表中唯一字符的数量：{len(chars)}")
    print("词汇表：")
    print(''.join(chars))

    # 将输入数据划分为训练集和测试集
    test_set_size = min(1000, int(len(words) * 0.1)) # 训练集的 10%，或最多 1000 个示例
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(f"将数据集划分为 {len(train_words)} 个训练示例和 {len(test_words)} 个测试示例")

    # 包装在数据集对象中
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset

class InfiniteDataLoader:
    """
    这真的很 hacky，我对此并不感到自豪，但是在 PyTorch 中似乎没有
    更好的方法来创建一个无限的 dataloader？
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # 这在技术上只会在 1e10 个样本之后发生...（即基本上永远不会）
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Make More")
    # 系统/输入/输出
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help="输入文件，每行一个内容")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="输出工作目录")
    parser.add_argument('--resume', action='store_true', help="使用此标志时，我们将从工作目录中现有的模型恢复优化")
    parser.add_argument('--sample-only', action='store_true', help="仅从模型采样并退出，不进行训练")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="训练/测试的数据工作线程数")
    parser.add_argument('--max-steps', type=int, default=-1, help="要运行的最大优化步数，或 -1 表示无限。")
    parser.add_argument('--device', type=str, default='cpu', help="用于计算的设备，例如：cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, default=3407, help="随机种子")
    # 采样
    parser.add_argument('--top-k', type=int, default=-1, help="用于采样的 top-k，-1 表示无 top-k")
    # 模型
    parser.add_argument('--type', type=str, default='transformer', help="要使用的模型类类型，bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4, help="层数")
    parser.add_argument('--n-head', type=int, default=4, help="头数（在 transformer 中）")
    parser.add_argument('--n-embd', type=int, default=64, help="模型中的特征通道数")
    parser.add_argument('--n-embd2', type=int, default=64, help="模型中其他地方的特征通道数")
    # 优化
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="优化期间的批大小")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="学习率")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="权重衰减")
    args = parser.parse_args()
    print(vars(args))

    # 系统初始化
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # 初始化数据集
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"数据集确定：{vocab_size=}, {block_size=}")

    # 初始化模型
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)
    if args.type == 'transformer':
        model = Transformer(config)
    elif args.type == 'bigram':
        model = Bigram(config)
    elif args.type == 'mlp':
        model = MLP(config)
    elif args.type == 'rnn':
        model = RNN(config, cell_type='rnn')
    elif args.type == 'gru':
        model = RNN(config, cell_type='gru')
    elif args.type == 'bow':
        model = BoW(config)
    else:
        raise ValueError(f'model type {args.type} is not recognized')
    model.to(args.device)
    print(f"模型参数数量：{sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only: # 注意：如果我们只采样，那么我们也假设我们正在恢复
        print("从工作目录中的现有模型恢复")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_samples(num=50)
        sys.exit()

    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

    # 初始化 dataloader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    # 训练循环
    best_loss = None
    step = 0
    while True:

        t0 = time.time()

        # 获取下一个批次，发送到设备，并将其解包为输入和目标
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # 输入模型
        logits, loss = model(X, Y)

        # 计算梯度，更新权重
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # 等待 GPU 上的所有 CUDA 工作完成，然后计算迭代所用时间
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # 日志记录
        if step % 10 == 0:
            print(f"步骤 {step} | 损失 {loss.item():.4f} | 步骤时间 {(t1-t0)*1000:.2f}ms")

        # 评估模型
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"步骤 {step} 训练损失：{train_loss} 测试损失：{test_loss}")
            # 如果模型有所改进，则将其保存到磁盘
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"测试损失 {test_loss} 是目前为止最好的，将模型保存到 {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # 从模型中采样
        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1
        # 终止条件
        if args.max_steps >= 0 and step >= args.max_steps:
            break


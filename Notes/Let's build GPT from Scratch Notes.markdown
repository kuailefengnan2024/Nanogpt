# 📚 Let's build GPT: from scratch 笔记

## 🎯 目标

从零实现GPT（生成式预训练变换器），理解Transformer解码器架构，构建一个字符级语言模型。

- **背景** 🔍: 从makemore系列升级，WaveNet处理长上下文，但Transformer更适合大规模语言建模。
- **重点** 🌟: 实现GPT的核心组件（多头自注意力、FFN、层归一化），训练一个简单字符级模型。

---

## 🧠 GPT 代码结构与流程

### 1️⃣ 数据准备

- **任务**: 使用文本数据集（如Tiny Shakespeare），构建字符级输入-输出对。

- **代码**:
  - 读取文本，创建字符到索引的映射（词汇表，如65个字符）。
  - 输入：上下文序列（`block_size`字符），输出：下一字符。

- **形象化**:

  ```
  文本: "to be or not"
  输入: [t,o, ] → 输出: b
  ```

- **图标**: 📜 数据像“文本的字符流”。

### 2️⃣ Transformer架构

- **核心组件**:

  - **嵌入层**: 字符索引 → 嵌入向量（`n_embd`，如384维）。
  - **位置编码**: 为每个位置添加固定或可学习的向量。
  - **多头自注意力**: 捕捉字符间依赖，允许并行计算。
  - **前馈网络（FFN）**: 每位置的非线性变换。
  - **层归一化（LayerNorm）**: 稳定训练。
  - **残差连接**: 缓解梯度消失。

- **代码**:

  ```python
  class GPT(nn.Module):
      def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer):
          self.token_emb = nn.Embedding(vocab_size, n_embd)
          self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
          self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
          self.ln_f = nn.LayerNorm(n_embd)
          self.head = nn.Linear(n_embd, vocab_size)

      def forward(self, idx):
          tok_emb = self.token_emb(idx)  # (B, T, C)
          pos_emb = self.pos_emb[:, :idx.size(1), :]  # (1, T, C)
          x = tok_emb + pos_emb  # (B, T, C)
          for block in self.blocks: x = block(x)
          x = self.ln_f(x)
          logits = self.head(x)  # (B, T, vocab_size)
          return logits
  ```

- **形象化**:

  ```
  输入 → 嵌入+位置编码 → N个Transformer块 → LayerNorm → 线性层 → logits
  ```

- **图标**: 🏗️ Transformer像“模块化的语言处理工厂”。

### 3️⃣ 自注意力机制

- **作用**: 每个字符根据上下文动态计算注意力权重。

- **代码**:

  ```python
  class Head(nn.Module):
      def __init__(self, head_size):
          self.key = nn.Linear(n_embd, head_size, bias=False)
          self.query = nn.Linear(n_embd, head_size, bias=False)
          self.value = nn.Linear(n_embd, head_size, bias=False)
          self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

      def forward(self, x):
          k = self.key(x)    # (B, T, head_size)
          q = self.query(x)  # (B, T, head_size)
          wei = q @ k.transpose(-2, -1) * head_size**-0.5  # (B, T, T)
          wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # 因果掩码
          wei = F.softmax(wei, dim=-1)
          v = self.value(x)  # (B, T, head_size)
          out = wei @ v      # (B, T, head_size)
          return out
  ```

- **形象化**:

  ```
  查询(Q) × 键(K) → 注意力权重 → 掩码 → softmax → 加权值(V)
  ```

- **图标**: 🔎 自注意力像“上下文的动态聚焦”。

### 4️⃣ 训练与生成

- **训练**:

  - 损失：交叉熵（`F.cross_entropy`）。
  - 优化器：Adam，学习率（如3e-4）。
  - 数据：mini-batch处理。

- **生成**:

  - 从初始上下文开始，采样logits，生成新字符。

- **图标**: 🎨 生成像“模型的文本创作”。

---

## 🔑 核心概念

- **自注意力** 🔗: 动态建模字符间依赖，允许并行计算。
- **因果掩码** 🛑: 确保只关注历史字符，适合生成任务。
- **残差连接** ➕: 缓解深层网络的梯度消失。
- **层归一化** 🛡️: 标准化激活，稳定训练。
- **位置编码** 📍: 提供序列顺序信息。

---

## 🛠️ 实用技巧

- **超参数** ⚙️: 调整`n_embd`（384）、`n_head`（6）、`n_layer`（6）、`block_size`（128）。
- **初始化** 🛠️: 权重用小标准差，防止初始损失过高。
- **生成多样性** 🌈: 用`temperature`控制采样随机性。
- **调试** 🔍: 检查注意力权重，确保因果掩码正确。

---

## 📚 资源

- **视频**: Let's build GPT: from scratch, in code, spelled out
- **代码**: nn-zero-to-hero GitHub
- **论文**: Vaswani et al. (2017)《Attention is All You Need》
- **Colab**: 视频描述中的Jupyter笔记本

---

## 🌟 总结

从零实现GPT，核心是Transformer解码器（自注意力+FFN+残差+LayerNorm）。代码模块化，清晰展示从嵌入到生成的全流程，字符级建模为理解GPT奠定基础。
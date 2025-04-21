# 📚 Let's reproduce GPT-2 (124M) 笔记

## 🎯 目标

复现OpenAI的GPT-2（124M参数），从头训练一个子词级语言模型，理解大规模Transformer的实现与训练。

- **背景** 🔍: GPT从字符级升级到子词级，GPT-2是强大且开源的语言模型。
- **重点** 🌟: 实现GPT-2架构，加载预训练权重，微调并生成文本。

---

## 🧠 GPT-2 (124M) 代码结构与流程

### 1️⃣ 模型架构

- **参数**:

  - 词汇表：50,257（BPE）。
  - 嵌入维度：768。
  - 上下文长度：1,024。
  - 层数：12。
  - 注意力头数：12。
  - 参数量：约1.24亿。

- **代码**:

  ```python
  class GPT2(nn.Module):
      def __init__(self):
          super().__init__()
          self.transformer = nn.ModuleDict(dict(
              wte = nn.Embedding(vocab_size, n_embd),
              wpe = nn.Embedding(block_size, n_embd),
              h = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)]),
              ln_f = nn.LayerNorm(n_embd),
          ))
          self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
  
      def forward(self, idx):
          tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
          pos = torch.arange(0, idx.size(1), device=idx.device)
          pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
          x = tok_emb + pos_emb
          for block in self.transformer.h: x = block(x)
          x = self.transformer.ln_f(x)
          logits = self.lm_head(x)  # (B, T, vocab_size)
          return logits
  ```

- **形象化**:

  ```
  输入 → 词嵌入+位置嵌入 → 12个Transformer块 → LayerNorm → 线性头 → logits
  ```

- **图标**: 🏰 GPT-2像“大规模语言处理城堡”。

### 2️⃣ Tokenizer

- **实现**: 使用预训练BPE Tokenizer（50,257词汇）。

- **代码**:

  - 加载Hugging Face的`gpt2` Tokenizer或自定义BPE。
  - 编码文本为token序列，解码回文本。

- **图标**: 🔄 Tokenizer像“子词的编码器”。

### 3️⃣ 训练数据

- **数据集**: WebText或公开文本（如Wikipedia）。

- **处理**:

  - 编码为token序列，切分为`block_size`（1,024）。
  - 构建mini-batch（如batch_size=8）。

- **图标**: 📚 数据像“互联网的文本宝库”。

### 4️⃣ 训练流程

- **损失**: 交叉熵，预测下一token。

- **优化器**: AdamW，学习率（如6e-4），带权重衰减。

- **分布式训练**:

  - 数据并行（多GPU分担batch）。
  - 梯度累积模拟大batch。

- **代码**:

  ```python
  model = GPT2().to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
  for step in range(max_steps):
      xb, yb = get_batch('train')
      logits = model(xb)
      loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  ```

- **图标**: 🚀 训练像“模型的深度学习马拉松”。

### 5️⃣ 加载预训练权重

- **方法**: 从Hugging Face加载GPT-2 (124M)权重。

- **代码**:

  - 映射权重到自定义模型（`wte` → 词嵌入，`wpe` → 位置嵌入）。
  - 验证前向传播一致性。

- **图标**: 📦 预训练权重像“模型的知识库”。

### 6️⃣ 生成文本

- **方法**: 自回归采样，top-k或top-p（nucleus）采样。

- **代码**:

  ```python
  def generate(idx, max_new_tokens, temperature=1.0):
      for _ in range(max_new_tokens):
          logits = model(idx[:, -block_size:])
          logits = logits[:, -1, :] / temperature
          probs = F.softmax(logits, dim=-1)
          idx_next = torch.multinomial(probs, num_samples=1)
          idx = torch.cat((idx, idx_next), dim=1)
      return idx
  ```

- **图标**: 🎨 生成像“模型的语言创作”。

---

## 🔑 核心概念

- **子词建模** 🧬: BPE Tokenizer平衡序列长度与语义表达。
- **大规模Transformer** 🏗️: 深层网络（12层）捕捉复杂语言模式。
- **预训练权重** 📚: 复用OpenAI的权重，快速启动。
- **采样策略** 🌈: top-k/top-p控制生成多样性与连贯性。

---

## 🛠️ 实用技巧

- **超参数** ⚙️: 学习率6e-4，batch_size 8，上下文1,024。
- **内存优化** 🛠️: 梯度累积支持小GPU训练。
- **生成控制** 🎛️: 调整`temperature`（0.8-1.2）或top-p（0.9）。
- **验证** 🔍: 检查预训练权重加载后损失是否合理。

---

## 📚 资源

- **视频**: Let's reproduce GPT-2 (124M)
- **代码**: nn-zero-to-hero GitHub
- **论文**: Radford et al. (2019)《Language Models are Unsupervised Multitask Learners》
- **Hugging Face**: GPT-2模型与Tokenizer
- **Colab**: 视频描述中的Jupyter笔记本

---

## 🌟 总结

复现GPT-2 (124M)通过子词Tokenizer和12层Transformer实现强大语言建模。代码从数据到训练再到生成，清晰模块化，加载预训练权重加速开发，展现现代NLP的威力。
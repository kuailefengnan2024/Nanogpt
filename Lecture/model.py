"""
GPT语言模型的完整定义，所有内容都在这个单一文件中。
参考资料：
1) OpenAI发布的官方GPT-2 TensorFlow实现：
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers的PyTorch实现：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ 带有可选偏置的LayerNorm。PyTorch不直接支持简单的bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 所有注意力头的key、query、value投影，但是在一个批次中
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # Flash Attention使GPU性能大幅提升，但仅在PyTorch >= 2.0中支持
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("警告：使用慢速注意力机制。Flash Attention需要PyTorch >= 2.0")
            # 因果掩码确保注意力只应用于输入序列中的左侧
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # 批量大小，序列长度，嵌入维度(n_embd)

        # 为批次中的所有头计算query、key、values，并将头前移作为批次维度
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 因果自注意力; 自注意力计算: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # 使用Flash Attention CUDA内核的高效注意力
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # 手动实现注意力
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 重新组合所有头的输出并排

        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2词汇表大小为50257，为了效率填充到最接近的64的倍数
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True：在Linear和LayerNorm中使用偏置，像GPT-2一样。False：稍微更好且更快

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 使用权重绑定时，使用torch.compile()可能会生成一些警告：
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # 不太确定这是什么，到目前为止似乎无害。待研究
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # 初始化所有权重
        self.apply(self._init_weights)
        # 根据GPT-2论文，对残差投影应用特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # 报告参数数量
        print("参数数量: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        返回模型中的参数数量。
        对于非嵌入计数（默认），位置嵌入会被减去。
        由于参数共享，词嵌入实际上在最后一层中用作权重，所以包括它们。
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"无法处理长度为{t}的序列，块大小仅为{self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # 形状为(t)

        # 前向传播GPT模型本身
        tok_emb = self.transformer.wte(idx) # 令牌嵌入，形状为(b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # 位置嵌入，形状为(t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 如果给定了目标，也计算损失
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理时的小优化：仅在最后一个位置前向传播lm_head
            logits = self.lm_head(x[:, [-1], :]) # 注意：使用列表[-1]以保留时间维度
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # 模型手术，在必要时减小块大小
        # 例如，我们可能加载GPT2预训练模型检查点（块大小为1024）
        # 但想为一些更小、更简单的模型使用更小的块大小
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # 默认为空字典
        # 只有dropout可以被覆盖，详见下面的注释
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("从预训练的gpt加载权重: %s" % model_type)

        # n_layer、n_head和n_embd由model_type决定
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M参数
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M参数
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M参数
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M参数
        }[model_type]
        print("强制设置vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # GPT模型检查点始终为50257
        config_args['block_size'] = 1024 # GPT模型检查点始终为1024
        config_args['bias'] = True # GPT模型检查点始终为True
        # 如果需要，我们可以覆盖dropout率
        if 'dropout' in override_args:
            print(f"覆盖dropout率为{override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # 创建一个从头初始化的minGPT模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # 丢弃这个掩码/缓冲区，不是参数

        # 初始化huggingface/transformers模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 复制时确保所有参数对齐，并在名称和形状上匹配
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # 忽略这些，只是缓冲区
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # 同上，只是掩码（缓冲区）
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 基本上，openai检查点使用"Conv1D"模块，但我们只想使用普通的Linear
        # 这意味着在导入时我们必须转置这些权重
        assert len(sd_keys_hf) == len(sd_keys), f"键不匹配: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 处理转置情况
                # 在openai的实现中，将(768, 768*3)看作(768, 768, 3)
                # 因此加载权重时需要转置
                assert k in sd
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].transpose(0, 1))
            else:
                # 复制
                assert k in sd
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 首先选择所有候选参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 创建需要进行权重衰减和不需要的白名单
        # 任何偏置项或Layer Norm/嵌入层的权重不应该进行权重衰减
        # Reference: https://github.com/pytorch/pytorch/pull/12402
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # 验证是否考虑了所有参数
        param_dict_sans_embed = {n: p for n, p in param_dict.items() 
                                if not n.endswith('transformer.wpe.weight')}
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_embed_params = self.transformer.wpe.weight.numel()
        assert num_decay_params + num_nodecay_params + num_embed_params == self.get_num_params(False)
        # 创建AdamW优化器，添加权重衰减项
        # Reference: https://github.com/openai/following-paper-rl/blob/main/bypass_batchnorm/adam.py#L18
        # AdamW优化器会使GPU内存使用更多（几百MB），但训练没有问题
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': self.transformer.wpe.weight, 'weight_decay': 0.0},
        ], lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ 估计MFU - 模型浮点计算利用率，或者说我们使用了理论浮点性能的多少百分比 """
        # 参考: https://github.com/openai/following-paper-rl/blob/main/scripts/mfu.py
        # 计算浮点运算次数
        N = self.get_num_params()
        L, H, Q, T = self.config.n_layer, self.config.n_head, self.config.n_embd//self.config.n_head, self.config.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # 除以时间，得到每秒浮点运算次数
        flops_achieved = flops_per_iter * (1.0/dt) # 每秒浮点运算次数
        # 除以GPU的理论峰值
        flops_promised = 312e12 # A100的理论峰值是312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        接收一个条件序列idx (LongTensor [b,t])，并生成max_new_tokens令牌。
        应用了简单的batched beam search-like技术来控制采样温度。
        """
        for _ in range(max_new_tokens):
            # 如果序列太长，截断
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 前向
            logits, _ = self(idx_cond)
            # 获取序列的最后一个时间步的预测
            logits = logits[:, -1, :] / temperature
            # 可选：裁剪probabilities变为仅top_k选项
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 附加到序列
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

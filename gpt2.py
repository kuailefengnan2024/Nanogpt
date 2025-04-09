# coding=utf-8
# 版权声明，表明代码的作者和使用许可 (Apache License 2.0)
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 你可以在以下地址获取许可副本：
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“现状”分发，不提供任何形式的担保。
# 请参阅许可的具体语言，了解权限和限制。

"""PyTorch OpenAI GPT-2 model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_gpt2 import GPT2Config

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openai-community/gpt2"
_CONFIG_FOR_DOC = "GPT2Config"

# 从 TensorFlow 加载预训练权重到 PyTorch 模型的函数
def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """加载 TensorFlow 检查点到 PyTorch 模型中"""
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error(
            "加载 TensorFlow 模型到 PyTorch 需要安装 TensorFlow。请查看 "
            "https://www.tensorflow.org/install/ 获取安装说明。"
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info(f"从 {tf_path} 转换 TensorFlow 检查点")
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"加载 TF 权重 {name}，形状为 {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # 去掉 "model/" 前缀
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"指针形状 {pointer.shape} 与数组形状 {array.shape} 不匹配")
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"初始化 PyTorch 权重 {name}")
        pointer.data = torch.from_numpy(array)
    return model

# 注意力机制的 eager 实现（非优化版本）
def eager_attention_forward(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    """实现标准的注意力计算"""
    attn_weights = torch.matmul(query, key.transpose(-1, -2))  # QK^T 计算注意力分数

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )  # 根据头维度缩放注意力分数 (1 / sqrt(d_k))

    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)  # 按层索引逆向缩放

    if not module.is_cross_attention:
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]  # 下三角掩码
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)  # 应用因果掩码

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask  # 应用额外的注意力掩码

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # 归一化注意力权重
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)  # 应用 dropout

    if head_mask is not None:
        attn_weights = attn_weights * head_mask  # 掩盖特定的注意力头

    attn_output = torch.matmul(attn_weights, value)  # 计算输出：注意力权重与 value 的加权和
    attn_output = attn_output.transpose(1, 2)  # 调整维度顺序

    return attn_output, attn_weights

# GPT-2 注意力模块
class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        # 创建下三角掩码，用于因果注意力（只关注之前的 token）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)  # 掩码值，负无穷近似

        self.embed_dim = config.hidden_size  # 嵌入维度
        self.num_heads = config.num_attention_heads  # 注意力头数
        self.head_dim = self.embed_dim // self.num_heads  # 每个头的维度
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` 必须能被 num_heads 整除 (得到 `embed_dim`: {self.embed_dim} 和 `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        # 初始化线性层，使用 Conv1D 实现（实际上是线性变换）
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)  # 交叉注意力需要单独的 key 和 value
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)  # 查询向量
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)  # 自注意力需要 query, key, value
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)  # 输出投影层

        self.attn_dropout = nn.Dropout(config.attn_pdrop)  # 注意力 dropout
        self.resid_dropout = nn.Dropout(config.resid_pdrop)  # 残差 dropout
        self.is_causal = True
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """修剪指定的注意力头"""
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        """提升精度并重新排序的注意力实现"""
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5
        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)
        with torch.amp.autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)
        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """注意力模块的前向传播"""
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError("交叉注意力需要定义 `q_attn` 权重")
            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)

        if use_cache is True:
            present = (key_states, value_states)
        else:
            present = None

        is_cross_attention = encoder_hidden_states is not None
        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
                using_eager = True
                logger.warning_once(
                    "`رtorch.nn.functional.scaled_dot_product_attention` 不支持 `output_attentions=True`。回退到 "
                    "eager 注意力实现。"
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

# GPT-2 MLP（多层感知器）模块
class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)  # 第一个线性层
        self.c_proj = Conv1D(embed_dim, intermediate_size)  # 第二个线性层
        self.act = ACT2FN[config.activation_function]  # 激活函数（如 GELU）
        self.dropout = nn.Dropout(config.resid_pdrop)  # 残差 dropout

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        """MLP 前向传播"""
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# GPT-2 Transformer 块
class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size  # 默认 4 倍隐藏维度

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # 第一层归一化
        self.attn = GPT2Attention(config=config, layer_idx=layer_idx)  # 注意力层
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # 第二层归一化

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)  # MLP 层

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        """Transformer 块的前向传播"""
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError("需要配置 `add_cross_attention=True` 来使用交叉注意力")
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs

# GPT-2 预训练模型基类
class GPT2PreTrainedModel(PreTrainedModel):
    """
    抽象类，用于处理权重初始化和预训练模型的加载。
    """
    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))

@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    """
    双头模型的输出类，用于语言建模和多选分类任务。
    """
    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    mc_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

GPT2_START_DOCSTRING = r"""
    该模型继承自 `PreTrainedModel`。请查看超类文档以了解通用方法（如下载、保存等）。
    该模型是一个 PyTorch `torch.nn.Module` 子类，可作为常规 PyTorch 模块使用。

    参数:
        config (`GPT2Config`): 模型配置类，包含所有参数。
            用配置文件初始化不会加载权重，仅加载配置。使用 `from_pretrained` 方法加载模型权重。
"""

GPT2_INPUTS_DOCSTRING = r"""
    参数:
        input_ids (`torch.LongTensor`): 输入序列的 token ID。
        past_key_values (`Tuple[Tuple[torch.Tensor]]`): 预计算的键值对，用于加速解码。
        attention_mask (`torch.FloatTensor`): 注意力掩码，0 表示掩盖，1 表示不掩盖。
        token_type_ids (`torch.LongTensor`): 分段 token ID。
        position_ids (`torch.LongTensor`): 位置 ID。
        head_mask (`torch.FloatTensor`): 注意力头掩码。
        inputs_embeds (`torch.FloatTensor`): 嵌入表示（替代 input_ids）。
        use_cache (`bool`): 是否使用缓存。
        output_attentions (`bool`): 是否输出注意力权重。
        output_hidden_states (`bool`): 是否输出隐藏状态。
        return_dict (`bool`): 是否返回字典格式。
"""

# GPT-2 基础模型
@add_start_docstrings(
    "不带特定头部的基础 GPT-2 模型，输出原始隐藏状态。",
    GPT2_START_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
    _supports_param_buffer_assignment = False

    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)  # 词嵌入
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)  # 位置嵌入
        self.drop = nn.Dropout(config.embd_pdrop)  # 嵌入 dropout
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])  # Transformer 层
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)  # 最终归一化

        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """修剪注意力头"""
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """基础模型的前向传播"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("不能同时指定 input_ids 和 inputs_embeds")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("必须指定 input_ids 或 inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)  # 将输入 token ID 转换为嵌入
        position_embeds = self.wpe(position_ids)  # 获取位置嵌入
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)  # 词嵌入与位置嵌入相加

        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.to(dtype=self.dtype)
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)  # 嵌入 dropout

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` 与梯度检查点不兼容，设置为 False")
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(len(self.h)):
            block, layer_past = self.h[i], past_key_values[i]
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)  # 最终归一化
        hidden_states = hidden_states.view(output_shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

# GPT-2 语言模型（带 LM 头）
@add_start_docstrings(
    "带语言建模头的 GPT-2 模型（权重与输入嵌入绑定）。",
    GPT2_START_DOCSTRING,
)
class GPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)  # 基础 GPT-2 模型
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 语言建模头

        self.model_parallel = False
        self.device_map = None

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """语言模型的前向传播"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(  # 调用基础模型
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]  # 获取隐藏状态

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)  # 通过语言建模头生成 logits

        loss = None
        if labels is not None:
            # 计算语言建模损失
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """重新排序缓存以匹配束搜索的索引"""
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
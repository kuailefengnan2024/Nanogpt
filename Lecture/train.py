"""
此训练脚本可以在单个GPU的调试模式下运行，
也可以在使用分布式数据并行(ddp)的更大训练运行中运行。

在单个GPU上运行的示例：
$ python train.py --batch_size=32 --compile=False

在1个节点上使用DDP运行4个gpu的示例：
$ torchrun --standalone --nproc_per_node=4 train.py

在2个节点上使用DDP运行4个gpu的示例：
- 在第一个（主）节点上运行，IP地址示例为123.456.123.456：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- 在工作节点上运行：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
（如果您的集群没有Infiniband互连，请在命令前加上NCCL_IB_DISABLE=1）
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 默认配置值，设计用于在OpenWebText上训练gpt2 (124M)
# 输入/输出
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # 如果为True，脚本在第一次评估后立即退出
always_save_checkpoint = True # 如果为True，每次评估后都保存检查点
init_from = 'scratch' # 'scratch'或'resume'或'gpt2*'
# wandb日志记录
wandb_log = False # 默认禁用
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# 数据
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # 用于模拟更大的批量大小
batch_size = 12 # 如果gradient_accumulation_steps > 1，这是微批量大小
block_size = 1024
# 模型
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # 对于预训练，0是好的，对于微调，尝试0.1+
bias = False # 我们是否在LayerNorm和Linear层中使用偏置？
# adamw优化器
learning_rate = 6e-4 # 最大学习率
max_iters = 600000 # 训练迭代的总数
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # 在此值处裁剪梯度，如果==0.0则禁用
# 学习率衰减设置
decay_lr = True # 是否衰减学习率
warmup_iters = 2000 # 预热的步数
lr_decay_iters = 600000 # 应该~=根据Chinchilla论文的max_iters
min_lr = 6e-5 # 最小学习率，应该~=根据Chinchilla论文的learning_rate/10
# DDP设置
backend = 'nccl' # 'nccl'，'gloo'等
# 系统
device = 'cuda' # 例如：'cpu'，'cuda'，'cuda:0'，'cuda:1'等，或在MacBook上尝试'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32'、'bfloat16'或'float16'，后者将自动实现GradScaler
compile = True # 使用PyTorch 2.0编译模型以提高速度
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # 从命令行或配置文件中覆盖
config = {k: globals()[k] for k in config_keys} # 对日志记录有用
# -----------------------------------------------------------------------------

# 各种初始化、派生属性、I/O设置
ddp = int(os.environ.get('RANK', -1)) != -1 # 这是ddp运行吗？
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # 此进程将进行日志记录、检查点等
    seed_offset = ddp_rank # 每个进程获得不同的种子
    # world_size数量的进程将同时训练，因此我们可以按比例
    # 减少每个进程所需的梯度累积迭代
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # 如果不是ddp，我们在单个gpu上运行，只有一个进程
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"每次迭代的令牌数将是: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # 允许在matmul上使用tf32
torch.backends.cudnn.allow_tf32 = True # 允许在cudnn上使用tf32
device_type = 'cuda' if 'cuda' in device else 'cpu' # 后续在torch.autocast中使用
# 注意：float16数据类型将自动使用GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 简易数据加载器
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # 我们每个批次重新创建np.memmap以避免内存泄漏，根据
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # 固定数组x,y，这允许我们异步移动它们到GPU（non_blocking=True）
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# 在这里初始化这些变量，如果init_from='resume'（即从检查点），可以覆盖
iter_num = 0
best_val_loss = 1e9

# 尝试从数据集推导vocab_size
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"在{meta_path}中找到vocab_size = {meta_vocab_size}")

# 模型初始化
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # 从命令行开始使用model_args
if init_from == 'scratch':
    # 从头初始化一个新模型
    print("从头初始化一个新模型")
    # 确定我们将用于从头训练的词汇表大小
    if meta_vocab_size is None:
        print("默认使用GPT-2的vocab_size为50304（50257向上舍入为效率）")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"从{out_dir}恢复训练")
    # 从检查点恢复训练。
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # 强制这些配置属性相等，否则我们甚至无法恢复训练
    # 其余属性（例如dropout）可以按照命令行中的需要保持不变
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # 创建模型
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 修复状态字典的键 :(
    # 老实说，不知道检查点有时如何获得这个前缀，需要更多调试
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"从OpenAI GPT-2权重初始化: {init_from}")
    # 从OpenAI GPT-2权重初始化
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # 读取创建的配置参数，以便我们可以正确地将它们存储到检查点中
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# 如果需要，使用模型手术减小模型块大小
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # 这样检查点将有正确的值
model.to(device)

# 初始化GradScaler。如果enabled=False，则scaler是一个空操作
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # 释放内存

# 编译模型
if compile:
    print("正在编译模型...（需要约一分钟）")
    unoptimized_model = model
    model = torch.compile(model) # 需要PyTorch 2.0

# 将模型包装到DDP容器中
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 帮助使用多个批次估计任一分割上的任意精确损失
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 学习率衰减调度器（带预热的余弦）
def get_lr(it):
    # 1) warmup_iters步的线性预热
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) 如果it > lr_decay_iters，返回最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 3) 在之间，使用余弦衰减到最小学习率
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff范围0..1
    return min_lr + coeff * (learning_rate - min_lr)

# 日志记录
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# 训练循环
X, Y = get_batch('train') # 获取第一批次
t0 = time.time()
local_iter_num = 0 # 在一个进程上运行的迭代次数
raw_model = model.module if ddp else model # 提取出原始模型，无论是否为DDP封装
running_mfu = -1.0
while True:

    # 确定和设置学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 每eval_interval评估损失一次
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"步骤 {iter_num}: 训练损失 {losses['train']:.4f}, 验证损失 {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # 转换为百分比
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"保存检查点到{out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # 前向传播
    t1 = time.time()
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps # 梯度累积，平均损失
    # 立即进行反向传播
    scaler.scale(loss).backward()
    
    # 到达梯度累积的整数倍时更新参数
    if (local_iter_num + 1) % gradient_accumulation_steps == 0:
        # 裁剪梯度
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # 参数更新
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # 计时和报告进度
    t2 = time.time()
    dt = t2 - t1
    if iter_num % log_interval == 0 and master_process:
        # 获取损失为固定格式的字符串，例如'0.635'
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # 让模型预热一下
            mfu = raw_model.estimate_mfu(gradient_accumulation_steps * batch_size, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: 损失 {lossf:.4f}, 时间 {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # 获取下一批次
    X, Y = get_batch('train')

    # 如果达到迭代次数上限，则退出
    if iter_num >= max_iters:
        break

if ddp:
    destroy_process_group()

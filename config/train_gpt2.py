# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
# 训练 GPT-2 (124M) 的配置，在 1 个节点的 8X A100 40GB 上可以达到非常好的约 2.85 的损失
# 按照以下方式启动（例如在 screen 会话中）并等待约 5 天：
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
# 这些设置使总批量大小约为 0.5M
# 12 批量大小 * 1024 块大小 * 5 梯度累积 * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
# 这使得 token 总数为 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
# 评估相关设置
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
# 权重衰减
weight_decay = 1e-1

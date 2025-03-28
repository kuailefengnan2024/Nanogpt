import time

out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on  # 随意开启
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2-xl' # this is the largest GPT-2 model  # 这是最大的 GPT-2 模型

# only save checkpoints if the validation loss improves
# 仅在验证损失改善时保存检查点
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
# 每次迭代的示例数：
# 1 批量大小 * 32 梯度累积 * 1024 tokens = 32,768 tokens/iter
# shakespeare 有 301,966 个 tokens，所以 1 个 epoch ~= 9.2 次迭代
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
# 以恒定学习率微调
learning_rate = 3e-5
decay_lr = False

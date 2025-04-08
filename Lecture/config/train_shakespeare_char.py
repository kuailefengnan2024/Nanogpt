# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
# 训练一个微型字符级莎士比亚模型
# 适合在 MacBook 等设备上调试和运行

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit  # 频繁评估因为模型会过拟合
eval_iters = 200
log_interval = 10 # don't print too too often  # 不要过于频繁地打印

# we expect to overfit on this small dataset, so only save when val improves
# 我们预计会在这个小数据集上过拟合，所以只在验证集性能提升时保存
always_save_checkpoint = False

wandb_log = False # override via command line if you like  # 如果需要可以通过命令行覆盖
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters  # 上下文最多包含256个前面的字符

# baby GPT model :)  # 小型GPT模型 :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher  # 小型网络可以使用稍高的学习率
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually  # 通常设置为等于max_iters
min_lr = 1e-4 # learning_rate / 10 usually  # 通常为learning_rate的十分之一
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small  # 设置大一点因为每次迭代的token数量较小

warmup_iters = 100 # not super necessary potentially  # 可能不是特别必要

# on macbook also add  # 在MacBook上还可以添加
# device = 'cpu'  # run on cpu only  # 仅在CPU上运行
# compile = False # do not torch compile the model  # 不要使用torch编译模型

# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
# 评估基础 gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M 参数
batch_size = 8
eval_iters = 500 # use more iterations to get good estimate  # 使用更多迭代以获得良好的估计
eval_only = True
wandb_log = False
init_from = 'gpt2'
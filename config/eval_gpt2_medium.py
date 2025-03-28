# evaluate the base gpt2-medium
# n_layer=24, n_head=16, n_embd=1024
# 350M parameters
# 评估基础 gpt2-medium
# n_layer=24, n_head=16, n_embd=1024
# 350M 参数
batch_size = 8
eval_iters = 500 # use more iterations to get good estimate  # 使用更多迭代以获得良好的估计
eval_only = True
wandb_log = False
init_from = 'gpt2-medium'
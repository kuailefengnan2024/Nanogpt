import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

# 读取名字数据
words = open('names.txt', 'r').read().splitlines()
print(words[:8])  # 显示前8个名字

# 数据集大小
print(len(words))

# 构建字符词汇表和字符到整数的映射
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
print(itos)  # 显示整数到字符的映射

# 构建数据集
block_size = 3  # 上下文长度：用3个字符预测下一个字符

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # 滑动窗口更新上下文
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

# 划分数据集：训练集80%，验证集10%，测试集10%
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])  # 训练集
Xdev, Ydev = build_dataset(words[n1:n2])  # 验证集
Xte, Yte = build_dataset(words[n2:])  # 测试集

# 初始化模型参数
g = torch.Generator().manual_seed(2147483647)  # 固定随机种子
C = torch.randn((27, 10), generator=g)  # 字符嵌入矩阵
W1 = torch.randn((30, 200), generator=g)  # 第一层权重
b1 = torch.randn(200, generator=g)  # 第一层偏置
W2 = torch.randn((200, 27), generator=g)  # 第二层权重
b2 = torch.randn(27, generator=g)  # 第二层偏置
parameters = [C, W1, b1, W2, b2]

# 计算总参数量
print(sum(p.nelement() for p in parameters))

# 启用梯度计算
for p in parameters:
    p.requires_grad = True

# 训练循环
stepi, lossi = [], []
for i in range(200000):
    # 小批量采样
    ix = torch.randint(0, Xtr.shape[0], (32,))
    
    # 前向传播
    emb = C[Xtr[ix]]  # (32, 3, 10) 嵌入
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 200) 隐藏层
    logits = h @ W2 + b2  # (32, 27) 输出
    loss = F.cross_entropy(logits, Ytr[ix])  # 计算损失
    
    # 反向传播
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # 更新参数
    lr = 0.1 if i < 100000 else 0.01  # 学习率调度
    for p in parameters:
        p.data += -lr * p.grad
    
    # 记录训练步数和损失
    stepi.append(i)
    lossi.append(loss.log10().item())

# 绘制损失曲线
plt.plot(stepi, lossi)

# 计算训练集损失
emb = C[Xtr]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print(loss)

# 计算验证集损失
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(loss)

# 可视化字符嵌入
plt.figure(figsize=(8, 8))
plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')

# 从模型采样生成名字
g = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:  # 遇到终止符
            break
    print(''.join(itos[i] for i in out))
import torch

# 定义变量，requires_grad=True 启用梯度跟踪
a = torch.tensor(-6.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)

# 前向传播
c = a + b
d = a * b + b**3
c = c + c + 1
c = c + 1 + c + (-a)
d = d + d * 2 + torch.relu(b + a)
d = d + 3 * d + torch.relu(b - a)
e = c - d
f = e**2
g = f / 2.0 + 10.0 / f

# 启用e的梯度跟踪
e.retain_grad()

# 启用反向传播
g.backward()

# 输出梯度 梯度值是对应张量属性grad
print(f'{e.grad:.4f}')  # dg/da
print(f'{b.grad:.4f}')  # dg/db